"""Explicit conditioned 2D numerical integration: accuracy, costs, PTX audit."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry,Evaluator,Request,baseline,monomials,PolynomialField,FieldEvaluator
from warpSPHBoundaries.fluxbridge.box import BoxEvaluator
from warpSPHBoundaries.fluxbridge.planar import PlanarEvaluator
from fluxbridge_radial_acceptance import timed

ROOT=Path(__file__).resolve().parents[2]


def sources():
    paths=list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))+list((ROOT/'tests/fluxbridge').glob('*.py'))
    paths+=[Path(__file__),ROOT/'scripts/studies/fluxbridge_fp32_diagnose.py',ROOT/'scripts/studies/fluxbridge_radial_acceptance.py']
    paths+=list((ROOT/'tests/fixtures').glob('fluxbridge-*.json'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def fixture(c,family,precision,nodes,device):
    req=[Request(**r) for r in c['requests']]
    kw=dict(kernel=c['kernel'],precision=precision,method='conditioned_boundary',nodes=nodes,device=device)
    if family=='v1':
        ev=Evaluator(Geometry([c['vertices']],2),req,**kw)
        return ev.evaluate([c['query']],c['h'])[0],ev.host_plan.amplification
    if family=='diskbox-v1':
        ev=BoxEvaluator([.2,-.05],[.7,.65],req,**kw)
        return ev.evaluate([c['query']],1)[0],ev.host_plan.amplification
    ev=PlanarEvaluator(c['kind'],2,req,**kw)
    value=ev.evaluate(float(c['distance']),c['normal'],1,**({} if c['half_width'] is None else dict(half_widths=float(c['half_width']))))[0]
    return value,ev.host_plan.amplification


def accuracy(device):
    rows=[]
    for precision in ('float32','float64'):
        for family in ('v1','diskbox-v1','planar-v1'):
            cases=json.loads((ROOT/f'tests/fixtures/fluxbridge-{family}.json').read_text())['cases']
            for kernel in ('w2','w4','w6','cubic'):
                errors=[];convergence=[];amp=0;count=0
                for c in cases:
                    if c['kernel']!=kernel or (c['kind']!='box' if family=='diskbox-v1' else c['dimension']!=2 or c.get('measure','volume')!='volume'):continue
                    v,a=fixture(c,family,precision,16,device);higher,_=fixture(c,family,precision,32,device)
                    ref=np.array(c['values'],float);errors.extend((abs(v-ref)/(1+abs(ref))).tolist())
                    convergence.extend((abs(v-higher)/(1+abs(higher))).tolist());amp=max(amp,a);count+=1
                error=np.array(errors);budget=1e-5 if precision=='float32' else 1e-10
                rows.append(dict(precision=precision,family=family,kernel=kernel,cases=count,channels=len(error),nodes=16,
                    worst_mixed_error=float(error.max()),failing_channels=int(np.sum(~np.isfinite(error)|(error>budget))),
                    budget=budget,convergence_nodes=32,mixed_convergence_difference=max(convergence),
                    max_chebyshev_profile_amplification=amp,rejected_cases=0,fallback_count=0,method='conditioned-boundary-quadrature'))
    return rows


def runtime(device):
    rows=[];geo=Geometry([[[0,0],[2,0],[0,2]]],2)
    modes=[('conditioned_boundary','float32'),('conditioned_boundary','float64'),('analytic','float32_compensated'),('analytic','float32')]
    for method,precision in modes:
        for kernel in ('w6','cubic'):
            for mask in ('baseline','degree9'):
                requests=baseline(2) if mask=='baseline' else [Request(a,ds) for a in monomials(2,9) for ds in [(),(0,),(1,),(0,0),(0,1),(1,1)]]
                count=128 if mask=='baseline' else 1
                q=np.stack([np.linspace(-.1,.8,count),np.full(count,.2)],axis=-1)
                ev,prepare=timed(lambda:Evaluator(geo,requests,kernel=kernel,precision=precision,method=method,device=device),device,1)
                batch,allocate=timed(lambda:ev.prepare(q,1),device,3)
                _,first=timed(batch.launch,device,1);_,warm=timed(batch.launch,device)
                _,readback=timed(batch.numpy,device);_,complete=timed(lambda:ev.evaluate(q,1),device,3)
                reference=Evaluator(geo,requests,kernel=kernel,device=device).evaluate(q,1)
                difference=float(np.max(abs(batch.numpy()-reference)/(1+abs(reference))))
                row=dict(method=method,precision=precision,kernel=kernel,mask=mask,queries=count,outputs=len(requests),
                    batch_bytes=batch.bytes,scratch_width=ev.width,plan_preparation=prepare,allocation_upload=allocate,
                    first_launch=first,warm=warm,readback=readback,complete=complete,
                    mixed_difference_from_fp64_analytic=difference,finite_workload_gate_passed=difference<1e-5,graph_replay=None)
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:batch.launch()
                    _,row['graph_replay']=timed(lambda:wp.capture_launch(capture.graph),device)
                rows.append(row)
    return rows


def assembly(device):
    centres=np.array([[3.*i,0.] for i in range(64)]);q=np.array([[3.*(i%64)+.4,.2] for i in range(128)])
    rows=[]
    for precision in ('float32','float64'):
        ev=BoxEvaluator(centres,[.3,.2],kernel='w6',precision=precision,method='conditioned_boundary',device=device)
        batch,allocate=timed(lambda:ev.prepare(q,1),device,3);batch.launch()
        _,warm=timed(batch.launch,device);_,complete=timed(lambda:ev.evaluate(q,1),device,3)
        full=ev.prepare(q,1,exhaustive=True);full.launch();_,exhaustive=timed(full.launch,device,3)
        reference=BoxEvaluator(centres,[.3,.2],kernel='w6',device=device).evaluate(q,1)
        rows.append(dict(precision=precision,kernel='w6',queries=128,elements=64,outputs=15,
            interactions=batch.count,exhaustive_interactions=full.count,search_allocation_upload=allocate,
            warm=warm,complete=complete,exhaustive_warm=exhaustive,batch_bytes=batch.bytes,
            localization_max_abs_difference=float(np.max(abs(batch.numpy()-full.numpy()))),
            mixed_difference_from_fp64_analytic=float(np.max(abs(batch.numpy()-reference)/(1+abs(reference))))))
    return rows


def compile_only(precision,output,arch):
    import warp._src.build as build
    geo=Geometry([[[0,0],[2,0],[0,2]]],2)
    field=FieldEvaluator(PolynomialField.basis(geo,1),precision=precision,method='conditioned_boundary')
    field.prepare([[.2,.2]],1)  # instantiate the common deterministic reduction
    PlanarEvaluator('plane',2,precision=precision,method='conditioned_boundary')
    rows=[]
    for name in ('conditioned2d','fields','planar','evaluate'):
        module=wp.get_module('warpSPHBoundaries.fluxbridge.'+name);options=dict(module.options);options['block_dim']=256
        source,extension,_,ltoirs,fatbins=module._run_codegen(options,is_cpu=False)
        path=output/f'{name}-{precision}.{extension}';path.write_text(source);ptx=output/f'{name}-{precision}.ptx'
        build.build_cuda(str(path),arch,str(ptx),pch_dir=None,use_precompiled_headers=False,fast_math=False,fuse_fp=False,ltoirs=ltoirs,fatbins=fatbins)
        instructions=[s.strip() for s in ptx.read_text().splitlines() if '.f64' in s and not s.strip().startswith(('.', '//'))]
        if precision=='float32' and instructions:raise RuntimeError('unexpected fp64 instructions')
        rows.append(dict(module=name,ptx_bytes=ptx.stat().st_size,fp64_instructions=len(instructions)))
    report=dict(precision=precision,arch=arch,warp=wp.__version__,compile_passed=True,execution_tested=False,
        cuda_available=wp.is_cuda_available(),modules=rows,source_sha256=sources())
    (output/f'compile-{precision}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--compile-only',choices=['float32','float64']);parser.add_argument('--arch',type=int,default=80)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);wp.init()
    if args.compile_only:compile_only(args.compile_only,args.output,args.arch);return
    device=wp.get_device(args.device)
    with (open('/tmp/fluxbridge-gpu.lock','w') if device.is_cuda else nullcontext(None)) as lock:
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_EX)
        cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown')
        report=dict(schema=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),device=str(device),cpu=cpu,warp=wp.__version__,
            source_sha256=sources(),fresh_cache=False,cuda_available=wp.is_cuda_available(),cuda_runtime_tested=device.is_cuda,
            accuracy=accuracy(device),runtime=runtime(device),assembly=assembly(device),
            limitations=['explicit numerical angle quadrature; finite convergence checks, no universal bound',
                '2D polygon volumes only; 3D and surface integration rejected',
                'analytic raw-fp32 recurrences remain separately experimental; this is not a hidden replacement',
                'whole-element search uses the host; native exhaustive disjoint rejection retains all faces of intersecting elements',
                'no pressure, porosity or fluid solver'])
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(args.output/'report.json')


if __name__=='__main__':main()
