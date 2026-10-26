"""CPU/CUDA evidence for the native 3D plane/sphere baseline.

Compiler-only mode audits PTX without claiming a GPU execution. Runtime mode
compares all requested channels, includes preparation, and serializes CUDA use.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import fcntl
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import baseline,Geometry,Evaluator
from warpSPHBoundaries.fluxbridge import radial

ROOT=Path(__file__).resolve().parents[2]
PRECISIONS=("float64","float32","float32_compensated")


def sources():
    paths=list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))
    paths+=list((ROOT/'tests/fluxbridge').glob('*.py'))
    paths+=[Path(__file__),ROOT/'scripts/make_fluxbridge_radial_fixtures.py',ROOT/'tests/fixtures/fluxbridge-radial-v1.json']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def timed(call,device,repeats=5):
    samples=[];result=None
    for _ in range(repeats):
        wp.synchronize_device(device)
        start=perf_counter();result=call();wp.synchronize_device(device)
        samples.append((perf_counter()-start)*1e3)
    return result,dict(median_ms=float(np.median(samples)),samples_ms=samples)


def accuracy(device):
    cases=json.loads((ROOT/'tests/fixtures/fluxbridge-radial-v1.json').read_text())['cases']
    rows=[]
    for precision in PRECISIONS:
        for kind in ('plane','sphere'):
            for kernel in ('w2','w4','w6','cubic'):
                selected=[c for c in cases if c['kind']==kind and c['kernel']==kernel]
                kw={} if kind=='plane' else dict(radii=np.array([c['radius'] for c in selected],float))
                ev=radial.RadialEvaluator(kind,kernel=kernel,precision=precision,device=device)
                out=ev.evaluate(np.array([c['distance'] for c in selected],float),np.array(selected[0]['normal'],float),1,**kw)
                ref=np.array([c['values'] for c in selected],float)
                error=np.abs(out-ref)/(1+np.abs(ref));budget=1e-10 if precision=='float64' else 1e-5
                rows.append(dict(precision=precision,kind=kind,kernel=kernel,channels=error.size,
                    worst_mixed_error=float(np.max(error)),budget=budget,
                    failing_channels=int(np.sum(~np.isfinite(error)|(error>budget))),rejected_cases=0,fallback_count=0))
    return rows


def plane_mesh():
    from scipy.spatial import ConvexHull
    vertices=np.array(list(product([-4.,4.],[-4.,4.],[-4.,0.])))
    hull=ConvexHull(vertices);faces=[]
    for f,eq in zip(hull.simplices,hull.equations):
        f=list(f);a,b,c=vertices[f]
        if np.dot(np.cross(b-a,c-a),eq[:3])<0:f[1],f[2]=f[2],f[1]
        faces.append(f)
    return Geometry.closed_surface(vertices,faces)


def runtime(device):
    from warpSPHBoundaries.mesh3d.radial import axisymmetric,select
    geometry=plane_mesh();rows=[]
    for precision in PRECISIONS:
        for kind in ('plane','sphere'):
            for kernel in ('w2','w4','w6','cubic'):
                for count in (1,128):
                    d=np.linspace(-.45,.9,count);normal=np.array([0.,0.,1.])
                    kw={} if kind=='plane' else dict(radii=1.)
                    ev,prepare=timed(lambda:radial.RadialEvaluator(kind,kernel=kernel,precision=precision,device=device),device,3)
                    batch,allocate=timed(lambda:ev.prepare(d,normal,1,**kw),device,3)
                    _,first=timed(batch.launch,device,1)
                    _,warm=timed(batch.launch,device)
                    _,readback=timed(batch.numpy,device)
                    _,complete=timed(lambda:ev.evaluate(d,normal,1,**kw),device,3)
                    row=dict(precision=precision,kind=kind,kernel=kernel,queries=count,outputs=28,
                        batch_bytes=batch.bytes,profile_terms=len(ev.plan.powers),plan_preparation=prepare,allocation_upload=allocate,
                        first_launch=first,warm_evaluation=warm,readback=readback,complete=complete)
                    if precision=='float32' and kernel=='cubic':
                        fixed=radial.functions('float32',beta_early_exit=False);plan=fixed.Plan()
                        for name in ('powers','radial_powers','profiles','bounds','coefficients'):
                            setattr(plan,name,getattr(ev.plan,name))
                        work=wp.empty_like(batch.work);values=wp.empty_like(batch.values)
                        def fixed_launch():
                            wp.launch(fixed.batch,batch.count,inputs=[batch.distances,batch.radii,batch.normals,
                                batch.supports,int(kind=='sphere'),plan,ev.metadata,work,values],device=device)
                        fixed_launch()
                        _,row['matched_fixed_64_beta_warm']=timed(fixed_launch,device)
                        row['fixed_beta_bitwise_equal']=bool(np.array_equal(batch.numpy(),values.numpy()))
                        if not row['fixed_beta_bitwise_equal']:raise RuntimeError('beta early exit changed raw-fp32 result')
                    # Existing CPU radial quadrature is a comparison, never a device fallback.
                    def comparison():
                        inv=axisymmetric(d,kernel,radius=None if kind=='plane' else 1.)
                        directions=np.broadcast_to(normal,(count,3));columns=[]
                        for req in baseline(3):
                            ids=[i for i,a in enumerate(req.alpha) for _ in range(a)]
                            if not req.derivative:
                                name='m00' if not ids else 'm'+''.join(map(str,req.alpha))
                            elif not ids and len(req.derivative)==1:
                                name='G'+'xyz'[req.derivative[0]]
                            else:
                                name=('g1' if len(req.derivative)==1 else 'h')+''.join('xyz'[i] for i in ids+list(req.derivative))
                            columns.append(select(inv,directions,name))
                        return np.stack(columns,axis=-1)
                    reference,reference_cost=timed(comparison,device,3)
                    row['existing_cpu_radial_and_rotation']=reference_cost
                    value=batch.numpy()
                    row['existing_radial_max_mixed_difference']=float(np.max(abs(value-reference)/(1+abs(reference))))
                    if kind=='plane':
                        mesh=Evaluator(geometry,baseline(3),kernel=kernel,precision=precision,device=device,compact=True)
                        q=np.stack([np.zeros(count),np.zeros(count),d],axis=-1)
                        old=mesh.prepare(q,1);old.launch()
                        _,row['matched_mesh_warm']=timed(old.launch,device)
                        _,row['matched_mesh_complete']=timed(lambda:mesh.evaluate(q,1),device,3)
                        row['mesh_max_mixed_difference']=float(np.max(abs(value-old.numpy())/(1+abs(value))))
                    row['graph_replay']=None
                    if device.is_cuda:
                        with wp.ScopedCapture(device=device) as capture:batch.launch()
                        _,row['graph_replay']=timed(lambda:wp.capture_launch(capture.graph),device)
                    rows.append(row)
    return rows


def compile_only(precision,output,arch):
    import warp._src.build as build
    radial.functions(precision)
    module=wp.get_module(radial.__name__)
    options=dict(module.options);options['block_dim']=256
    source,extension,_,ltoirs,fatbins=module._run_codegen(options,is_cpu=False)
    path=output/f'radial-{precision}.{extension}';path.write_text(source)
    ptx=output/f'radial-{precision}.ptx'
    build.build_cuda(str(path),arch,str(ptx),pch_dir=None,use_precompiled_headers=False,
        fast_math=False,fuse_fp=False,ltoirs=ltoirs,fatbins=fatbins)
    fp64=[s.strip() for s in ptx.read_text().splitlines() if '.f64' in s]
    instructions=[s for s in fp64 if not s.startswith(('.', '//'))]
    report=dict(precision=precision,warp=wp.__version__,arch=arch,compile_passed=True,
        execution_tested=False,cuda_available=wp.is_cuda_available(),ptx_bytes=ptx.stat().st_size,
        fp64_instructions=len(instructions),source_sha256=sources())
    (output/f'compile-{precision}.json').write_text(json.dumps(report,indent=2)+'\n')
    if precision!='float64' and instructions:raise RuntimeError('unexpected fp64 device instructions')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compile-only',choices=PRECISIONS)
    parser.add_argument('--arch',type=int,default=80)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);wp.init()
    if args.compile_only:
        compile_only(args.compile_only,args.output,args.arch);return
    device=wp.get_device(args.device)
    with (open('/tmp/fluxbridge-gpu.lock','w') if device.is_cuda else nullcontext(None)) as lock:
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_EX)
        cpu=platform.processor()
        if Path('/proc/cpuinfo').exists():
            cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
        report=dict(schema=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources(),
            device=str(device),cpu=cpu,warp=wp.__version__,fresh_cache=False,
            cuda_available=wp.is_cuda_available(),cuda_runtime_tested=device.is_cuda,
            accuracy=accuracy(device),runtime=runtime(device),
            limitations=['28-channel 3D baseline and subsets only; no higher field products',
                'finite fixture gates; CPU random checks are separate; no universal accuracy bound',
                'cubic outer Laurent primitive uses a 64-term analytic beta series with bounded tail',
                'radial comparison includes NumPy rotation; mesh comparison is the same 28-channel device contract'])
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(args.output/'report.json')


if __name__=='__main__':main()
