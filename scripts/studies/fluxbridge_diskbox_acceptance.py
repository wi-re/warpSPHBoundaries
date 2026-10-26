"""Native disk/box accuracy, quadrature convergence, assembly and compiler audit."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import baseline,Request,Evaluator,Geometry
from warpSPHBoundaries.fluxbridge import disk,disks
from warpSPHBoundaries.fluxbridge.disk import DiskQuadratureEvaluator
from warpSPHBoundaries.fluxbridge.disks import DiskArrayEvaluator
from warpSPHBoundaries.fluxbridge.box import BoxEvaluator
from fluxbridge_radial_acceptance import timed,PRECISIONS

ROOT=Path(__file__).resolve().parents[2]


def sources():
    paths=list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))+list((ROOT/'tests/fluxbridge').glob('*.py'))
    paths+=[Path(__file__),ROOT/'scripts/studies/fluxbridge_radial_acceptance.py',
        ROOT/'scripts/make_fluxbridge_diskbox_fixtures.py',ROOT/'tests/fixtures/fluxbridge-diskbox-v1.json']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def accuracy(device):
    cases=json.loads((ROOT/'tests/fixtures/fluxbridge-diskbox-v1.json').read_text())['cases'];rows=[]
    for precision in PRECISIONS:
        budget=1e-10 if precision=='float64' else 1e-5
        for kernel in ('w2','w4','w6','cubic'):
            selected=[c for c in cases if c['kind']=='disk' and c['kernel']==kernel]
            d=np.array([c['distance'] for c in selected],float);r=np.array([c['radius'] for c in selected],float)
            ev=DiskQuadratureEvaluator(kernel=kernel,precision=precision,device=device)
            value=ev.evaluate(d,[.6,.8],1,radii=r);ref=np.array([c['values'] for c in selected],float)
            higher=DiskQuadratureEvaluator(kernel=kernel,precision=precision,nodes=48,device=device).evaluate(d,[.6,.8],1,radii=r)
            error=abs(value-ref)/(1+abs(ref))
            rows.append(dict(kind='disk',precision=precision,kernel=kernel,channels=error.size,budget=budget,
                worst_mixed_error=float(np.max(error)),failing_channels=int(np.sum(~np.isfinite(error)|(error>budget))),
                nodes=24,convergence_nodes=48,mixed_convergence_difference=float(np.max(abs(value-higher)/(1+abs(higher)))),
                rejected_cases=0,fallback_count=0,method='circle-quadrature'))
            errors=[]
            for c in cases:
                if c['kind']!='box' or c['kernel']!=kernel:continue
                req=[Request(**r) for r in c['requests']]
                value=BoxEvaluator([.2,-.05],[.7,.65],req,kernel=kernel,precision=precision,device=device).evaluate([c['query']],1)[0]
                ref=np.array(c['values'],float);errors.extend((abs(value-ref)/(1+abs(ref))).tolist())
            errors=np.array(errors)
            rows.append(dict(kind='box',precision=precision,kernel=kernel,channels=len(errors),budget=budget,
                worst_mixed_error=float(np.max(errors)),failing_channels=int(np.sum(~np.isfinite(errors)|(errors>budget))),
                rejected_cases=0,fallback_count=0,method='analytic-polygon'))
    return rows


def runtime(device):
    rows=[]
    for precision in PRECISIONS:
        for kernel in ('w2','cubic'):
            for count in (1,128):
                d=np.linspace(-.25,.9,count);r=.3
                ev,prep=timed(lambda:DiskQuadratureEvaluator(kernel=kernel,precision=precision,device=device),device,3)
                batch,allocate=timed(lambda:ev.prepare(d,[1.,0.],1,radii=r),device,3)
                _,first=timed(batch.launch,device,1);_,warm=timed(batch.launch,device)
                _,readback=timed(batch.numpy,device);_,complete=timed(lambda:ev.evaluate(d,[1.,0.],1,radii=r),device,3)
                higher=DiskQuadratureEvaluator(kernel=kernel,precision=precision,nodes=48,device=device).prepare(d,[1,0],1,radii=r)
                higher.launch();_,high_cost=timed(higher.launch,device,3)
                rows.append(dict(kind='disk',precision=precision,kernel=kernel,queries=count,outputs=15,radius_support=r,
                    batch_bytes=batch.bytes,plan_preparation=prep,allocation_upload=allocate,first_launch=first,
                    warm=warm,complete=complete,readback=readback,nodes=24,higher_order_warm=high_cost,
                    higher_order_mixed_difference=float(np.max(abs(batch.numpy()-higher.numpy())/(1+abs(higher.numpy()))))))
    return rows


def assembly(device):
    rows=[];centres=np.array([[3.*i,0.] for i in range(64)])
    q=np.array([[3.*(i%64)+.4,.2] for i in range(128)])
    for precision in PRECISIONS:
        for kind in ('disk','box'):
            factory=(lambda:DiskArrayEvaluator(centres,.3,precision=precision,device=device)) if kind=='disk' else (lambda:BoxEvaluator(centres,[.3,.2],precision=precision,device=device))
            ev,prepare=timed(factory,device,3)
            batch,adjacency_allocation=timed(lambda:ev.prepare(q,1),device,3)
            batch.launch();_,warm=timed(batch.launch,device)
            _,complete=timed(lambda:ev.evaluate(q,1),device,3)
            full=ev.prepare(q,1,exhaustive=True);full.launch();_,exhaustive=timed(full.launch,device,3)
            delta=float(np.max(abs(batch.numpy()-full.numpy())))
            graph=None
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:batch.launch()
                _,graph=timed(lambda:wp.capture_launch(capture.graph),device)
            rows.append(dict(kind=kind,precision=precision,queries=128,elements=64,outputs=15,
                interactions=batch.count,exhaustive_interactions=full.count,batch_bytes=batch.bytes,
                geometry_plan_preparation=prepare,search_allocation_upload=adjacency_allocation,
                warm_evaluation_reduction=warm,complete=complete,exhaustive_warm=exhaustive,
                localization_max_abs_difference=delta,graph_replay=graph))
    return rows


def polygon_comparison(device):
    r=.3;q=np.array([[.4,.1]])
    exact=DiskArrayEvaluator([[0,0]],r,device=device).evaluate(q,1)[0]
    rows=[]
    for count in (32,128,512):
        theta=np.linspace(0,2*np.pi,count,endpoint=False)
        vertices=r*np.stack([np.cos(theta),np.sin(theta)],axis=-1)
        ev=Evaluator(Geometry([vertices],2),baseline(2),device=device,compact=True)
        batch=ev.prepare(q,1);batch.launch();_,warm=timed(batch.launch,device,3)
        error=float(np.max(abs(batch.numpy()[0]-exact)/(1+abs(exact))))
        rows.append(dict(vertices=count,queries=1,outputs=15,warm=warm,
            mixed_geometry_difference_from_circle=error,
            note='straight polygon changes the geometry; this is not equal-accuracy timing'))
    return rows


def compile_only(precision,output,arch):
    import warp._src.build as build
    DiskArrayEvaluator([[0,0]],.3,precision=precision,device='cpu')
    rows=[]
    for name in (disk.__name__,disks.__name__):
        module=wp.get_module(name);options=dict(module.options);options['block_dim']=256
        source,extension,_,ltoirs,fatbins=module._run_codegen(options,is_cpu=False)
        stem=name.rsplit('.',1)[-1]+'-'+precision
        path=output/(stem+'.'+extension);path.write_text(source);ptx=output/(stem+'.ptx')
        build.build_cuda(str(path),arch,str(ptx),pch_dir=None,use_precompiled_headers=False,
            fast_math=False,fuse_fp=False,ltoirs=ltoirs,fatbins=fatbins)
        instructions=[s.strip() for s in ptx.read_text().splitlines() if '.f64' in s and not s.strip().startswith(('.', '//'))]
        rows.append(dict(module=name,ptx_bytes=ptx.stat().st_size,fp64_instructions=len(instructions)))
        if precision!='float64' and instructions:raise RuntimeError('unexpected fp64 instructions')
    report=dict(precision=precision,arch=arch,warp=wp.__version__,compile_passed=True,execution_tested=False,
        cuda_available=wp.is_cuda_available(),modules=rows,source_sha256=sources())
    (output/f'compile-{precision}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--arch',type=int,default=80)
    parser.add_argument('--compile-only',choices=PRECISIONS)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);wp.init()
    if args.compile_only:compile_only(args.compile_only,args.output,args.arch);return
    device=wp.get_device(args.device)
    with (open('/tmp/fluxbridge-gpu.lock','w') if device.is_cuda else nullcontext(None)) as lock:
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_EX)
        cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown') if Path('/proc/cpuinfo').exists() else 'unknown'
        report=dict(schema=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),device=str(device),cpu=cpu,warp=wp.__version__,
            source_sha256=sources(),fresh_cache=False,cuda_available=wp.is_cuda_available(),cuda_runtime_tested=device.is_cuda,
            accuracy=accuracy(device),runtime=runtime(device),assembly=assembly(device),polygon_comparison=polygon_comparison(device),
            limitations=['disk is explicit numerical circle integration; no universal quadrature error bound',
                'disk only advertises degree-two combined baseline; no higher field products',
                'box uses shared analytic polygon arithmetic; no new lookup is certified',
                'raw box fp32 remains experimental','array search/preparation is on the host',
                'disk radial timings expect accurate distances/normals supplied; array timings include construction',
                'no pressure, porosity or fluid solver is implemented'])
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(args.output/'report.json')


if __name__=='__main__':main()
