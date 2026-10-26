"""Independent planar fixture gate, requested-mask costs and SM80 audit."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Request,baseline,monomials
from warpSPHBoundaries.fluxbridge import planar
from warpSPHBoundaries.fluxbridge.planar import PlanarEvaluator
from warpSPHBoundaries.fluxbridge.radial import RadialEvaluator
from fluxbridge_radial_acceptance import timed,PRECISIONS

ROOT=Path(__file__).resolve().parents[2]


def sources():
    paths=list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))+list((ROOT/'tests/fluxbridge').glob('*.py'))
    paths+=[Path(__file__),ROOT/'scripts/studies/fluxbridge_radial_acceptance.py',
            ROOT/'scripts/make_fluxbridge_planar_fixtures.py',ROOT/'tests/fixtures/fluxbridge-planar-v1.json']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def accuracy(device):
    cases=json.loads((ROOT/'tests/fixtures/fluxbridge-planar-v1.json').read_text())['cases'];rows=[]
    for precision in PRECISIONS:
        budget=1e-10 if precision=='float64' else 1e-5
        for dimension in (2,3):
            for kernel in ('w2','w4','w6','cubic'):
                errors=[]
                for c in cases:
                    if c['dimension']!=dimension or c['kernel']!=kernel:continue
                    ev=PlanarEvaluator(c['kind'],dimension,[Request(**r) for r in c['requests']],
                                       kernel=kernel,precision=precision,device=device)
                    kw={} if c['half_width'] is None else dict(half_widths=float(c['half_width']))
                    out=ev.evaluate(float(c['distance']),c['normal'],1,**kw)[0]
                    ref=np.array(c['values'],float);errors.extend((abs(out-ref)/(1+abs(ref))).tolist())
                error=np.array(errors)
                rows.append(dict(precision=precision,dimension=dimension,kernel=kernel,channels=len(error),budget=budget,
                    worst_mixed_error=float(error.max()),failing_channels=int(np.sum(~np.isfinite(error)|(error>budget))),
                    rejected_cases=0,fallback_count=0,method='analytic-support-clipping'))
    return rows


def runtime(device):
    rows=[]
    for precision in PRECISIONS:
        for dimension in (2,3):
            for kind in ('plane','strip'):
                for mask in ('occupancy','baseline','degree9'):
                    requests=([Request((0,)*dimension)] if mask=='occupancy' else baseline(dimension) if mask=='baseline'
                              else [Request(a,ds) for a in monomials(dimension,9) for ds in
                                    [()]+[(i,) for i in range(dimension)]+[(i,j) for i in range(dimension) for j in range(i,dimension)]])
                    count=1 if mask=='degree9' else 128
                    d=np.linspace(-.35,.9,count);normal=[0]*(dimension-1)+[1]
                    kwargs={} if kind=='plane' else dict(half_widths=.2)
                    ev,prepare=timed(lambda:PlanarEvaluator(kind,dimension,requests,precision=precision,device=device),device,1)
                    batch,allocate=timed(lambda:ev.prepare(d,normal,1,**kwargs),device,3)
                    _,first=timed(batch.launch,device,1);_,warm=timed(batch.launch,device)
                    _,readback=timed(batch.numpy,device);_,complete=timed(lambda:ev.evaluate(d,normal,1,**kwargs),device,3)
                    row=dict(precision=precision,kind=kind,dimension=dimension,kernel='w2',mask=mask,queries=count,
                        outputs=len(requests),scratch_width=ev.width,facets_per_query=ev.api.face_count,batch_bytes=batch.bytes,
                        plan_preparation=prepare,allocation_upload=allocate,first_launch=first,warm=warm,
                        readback=readback,complete=complete,graph_replay=None)
                    if mask=='baseline':
                        rotated=[.6,.8] if dimension==2 else [2/3,-1/3,2/3]
                        rb=ev.prepare(d,rotated,1,**kwargs);rb.launch()
                        _,row['rotated_warm']=timed(rb.launch,device)
                        _,row['rotated_complete']=timed(lambda:ev.evaluate(d,rotated,1,**kwargs),device,3)
                    if dimension==3 and kind=='plane' and mask=='baseline':
                        radial=RadialEvaluator('plane',precision=precision,device=device)
                        rb=radial.prepare(d,normal,1);rb.launch();_,row['matched_axial_baseline_warm']=timed(rb.launch,device)
                        row['axial_mixed_difference']=float(np.max(abs(rb.numpy()-batch.numpy())/(1+abs(rb.numpy()))))
                    if device.is_cuda:
                        with wp.ScopedCapture(device=device) as capture:batch.launch()
                        _,row['graph_replay']=timed(lambda:wp.capture_launch(capture.graph),device)
                    rows.append(row)
    return rows


def compile_only(precision,output,arch):
    import warp._src.build as build
    for dimension in (2,3):planar.functions(dimension,precision)
    module=wp.get_module(planar.__name__);options=dict(module.options);options['block_dim']=256
    source,extension,_,ltoirs,fatbins=module._run_codegen(options,is_cpu=False)
    path=output/f'planar-{precision}.{extension}';path.write_text(source);ptx=output/f'planar-{precision}.ptx'
    build.build_cuda(str(path),arch,str(ptx),pch_dir=None,use_precompiled_headers=False,
                     fast_math=False,fuse_fp=False,ltoirs=ltoirs,fatbins=fatbins)
    instructions=[s.strip() for s in ptx.read_text().splitlines() if '.f64' in s and not s.strip().startswith(('.', '//'))]
    if precision!='float64' and instructions:raise RuntimeError('unexpected fp64 device instructions')
    report=dict(precision=precision,arch=arch,warp=wp.__version__,compile_passed=True,execution_tested=False,
        cuda_available=wp.is_cuda_available(),ptx_bytes=ptx.stat().st_size,fp64_instructions=len(instructions),source_sha256=sources())
    (output/f'compile-{precision}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--compile-only',choices=PRECISIONS)
    parser.add_argument('--arch',type=int,default=80);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);wp.init()
    if args.compile_only:compile_only(args.compile_only,args.output,args.arch);return
    device=wp.get_device(args.device)
    with (open('/tmp/fluxbridge-gpu.lock','w') if device.is_cuda else nullcontext(None)) as lock:
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_EX)
        cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown')
        report=dict(schema=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources(),
            device=str(device),cpu=cpu,warp=wp.__version__,fresh_cache=False,cuda_available=wp.is_cuda_available(),
            cuda_runtime_tested=device.is_cuda,accuracy=accuracy(device),runtime=runtime(device),
            limitations=['finite fixtures do not certify a universal bound','raw fp32 remains experimental',
                'exact clipped geometry uses general recurrences; this is not a new axial closed form',
                'normal distances and normals are supplied; world geometry search is not timed',
                'thin-band conditioning policy rejects half_width/support below 1e-4 fp64 or 1e-2 fp32',
                'no pressure, porosity, solver or permeability law'])
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(args.output/'report.json')


if __name__=='__main__':main()
