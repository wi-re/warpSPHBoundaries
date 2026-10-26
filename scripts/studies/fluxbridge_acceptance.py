"""Reproducible native accuracy and serialized runtime evidence; no solver.

--output overrides the external evidence convention (use /tmp in a sandbox).
GPU timings are never substituted by browser or CPU timings.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
from math import comb
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile
from time import perf_counter
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry,Evaluator,Request,baseline,adjacency
from warpSPHBoundaries.fluxbridge.fields import PolynomialField,FieldEvaluator
from warpSPHBoundaries.fluxbridge.material import ResistanceMaterial
from warpSPHBoundaries.fluxbridge.affine import AffineResistanceMaterial
from warpSPHBoundaries.validation.paths import evidence_path


def timed(call,device,repeats=5):
    samples=[]
    result=None
    for _ in range(repeats):
        wp.synchronize_device(device)
        start=perf_counter()
        result=call()
        wp.synchronize_device(device)
        samples.append((perf_counter()-start)*1e3)
    return result,dict(median_ms=float(np.median(samples)),samples_ms=samples)


def accuracy(device):
    path=Path(__file__).resolve().parents[2]/"tests/fixtures/fluxbridge-v1.json"
    data=json.loads(path.read_text())
    rows=[]
    for precision,budget in (("float64",1e-10),("float32",1e-5),("float32_compensated",1e-5)):
        for dimension in (2,3):
            for kernel in ("w2","w4","w6","cubic"):
                for compact in (False,True):
                    errors=[]; count=0; failures=0; rejected=0
                    for case in data['cases']:
                        if case['dimension']!=dimension or case['kernel']!=kernel:
                            continue
                        g=Geometry([case['vertices']],dimension,case['measure'])
                        req=[Request(**r) for r in case['requests']]
                        e=Evaluator(g,req,kernel=kernel,precision=precision,device=device,compact=compact)
                        try:
                            value=e.evaluate([case['query']],case['h'],exhaustive=True)[0]
                        except ValueError:
                            rejected+=1
                            continue
                        reference=np.array(case['values'],float)
                        err=np.abs(value-reference)/(1+np.abs(reference))
                        errors.extend(err.tolist())
                        failures+=int(np.sum(~np.isfinite(err)|(err>budget)))
                        count+=len(err)
                    rows.append(dict(precision=precision,arithmetic="paired-fp32" if precision=="float32_compensated" else precision,dimension=dimension,kernel=kernel,compact=compact,
                                     channels=count,worst_mixed_error=max(errors,default=0),budget=budget,
                                     failing_channels=failures,rejected_cases=rejected,fallback_count=0))
    return rows


def quadrature(vertices,query,dimension,order):
    """Independent Duffy rule, never used by the analytic evaluator."""
    from warpSPHBoundaries.mesh3d.radial import kernel_jet
    from warpSPHBoundaries.edge.kernels import KERNELS
    nodes,weights=np.polynomial.legendre.leggauss(order)
    u=(nodes+1)/2; w=weights/2
    if dimension==2:
        a,b=np.meshgrid(u,u,indexing='ij')
        wa,wb=np.meshgrid(w,w,indexing='ij')
        points=vertices[0]+a[...,None]*(vertices[1]-vertices[0])+(1-a[...,None])*b[...,None]*(vertices[2]-vertices[0])
        jac=abs(np.linalg.det((vertices[1:]-vertices[0]).T))*(1-a)
        radius=np.linalg.norm(points-query,axis=-1)
        kernel=np.zeros_like(radius)
        for lo,hi,c in KERNELS['w2'].pieces:
            kernel+=np.where((radius>=float(lo))&(radius<float(hi)),np.polynomial.polynomial.polyval(radius,[float(t)/np.pi for t in c]),0)
        return float(np.sum(wa*wb*jac*kernel))
    a,b,c=np.meshgrid(u,u,u,indexing='ij')
    wa,wb,wc=np.meshgrid(w,w,w,indexing='ij')
    points=vertices[0]+a[...,None]*(vertices[1]-vertices[0])+(1-a[...,None])*b[...,None]*(vertices[2]-vertices[0])+(1-a[...,None])*(1-b[...,None])*c[...,None]*(vertices[3]-vertices[0])
    jac=abs(np.linalg.det((vertices[1:]-vertices[0]).T))*(1-a)**2*(1-b)
    radius=np.linalg.norm(points-query,axis=-1)
    return float(np.sum(wa*wb*wc*jac*kernel_jet(radius,'w2')[0]))


def runtime(device):
    rows=[]
    for dimension in (2,3):
        for precision in ('float64','float32','float32_compensated'):
            for count in (1,128):
                primitive=np.vstack([np.zeros(dimension),np.eye(dimension)])
                elements=[primitive+np.array([3.*i]+[0.]*(dimension-1)) for i in range(64)]
                geometry,prep=timed(lambda:Geometry(elements,dimension),device,repeats=3)
                q=np.array([np.full(dimension,.2)+np.array([3.*(i%64)]+[0.]*(dimension-1)) for i in range(count)])
                ev,plan=timed(lambda:Evaluator(geometry,baseline(dimension),device=str(device),precision=precision,compact=True),device,repeats=3)
                neighbors,search=timed(lambda:adjacency(geometry,q,1),device)
                batch,allocation=timed(lambda:ev.prepare(q,1,neighbors=neighbors),device,repeats=3)
                _,cold=timed(batch.launch,device,repeats=1)
                _,warm=timed(batch.launch,device)
                _,readback=timed(batch.numpy,device)
                _,complete=timed(lambda:ev.evaluate(q,1),device,repeats=3)
                full=ev.prepare(q,1,exhaustive=True)
                full.launch()
                _,exhaustive=timed(full.launch,device,repeats=3)
                delta=float(np.max(np.abs(batch.numpy()-full.numpy())))
                basis=PolynomialField.basis(geometry,1)
                field=FieldEvaluator(basis,device=str(device),precision=precision)
                fused=field.prepare(q,1,neighbors=neighbors)
                fused.launch()
                _,consumer=timed(fused.launch,device)
                _,field_complete=timed(lambda:field.evaluate(q,1),device,repeats=3)
                loads=np.ones((len(q),field.components))
                _,transpose=timed(lambda:fused.transpose(loads,np.ones(len(q))),device,repeats=3)
                material=ResistanceMaterial(geometry,np.broadcast_to(np.eye(dimension),(64,1,dimension,dimension)),np.zeros((64,dimension+1,dimension)))
                material_batch=material.prepare(q,1,device=str(device),precision=precision,neighbors=neighbors)
                velocity=wp.array(np.ones((count,dimension)),dtype=ev.api.real,device=device)
                masses=wp.ones(count,dtype=ev.api.real,device=device)
                material_batch.apply_device(velocity,masses)
                _,material_cost=timed(lambda:material_batch.apply_device(velocity,masses),device)
                _,material_complete=timed(lambda:material_batch.apply(np.ones((count,dimension)),np.ones(count)),device,repeats=3)
                affine=AffineResistanceMaterial(geometry,np.broadcast_to(np.eye(dimension),(64,1,dimension,dimension)),np.zeros((64,dimension+1,dimension)))
                affine_batch=affine.prepare(q,1,device=str(device),precision=precision,neighbors=neighbors)
                gradients=wp.zeros((count,dimension,dimension),dtype=ev.api.real,device=device)
                affine_batch.apply_device(velocity,masses,gradients)
                _,affine_cost=timed(lambda:affine_batch.apply_device(velocity,masses,gradients),device)
                graph=None
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        fused.launch()
                    _,graph=timed(lambda:wp.capture_launch(capture.graph),device)
                rows.append(dict(dimension=dimension,precision=precision,arithmetic='paired-fp32' if precision=='float32_compensated' else precision,kernel='w2',queries=count,elements=64,
                    interactions=batch.count,exhaustive_interactions=full.count,requested_outputs=len(baseline(dimension)),
                    scratch_and_batch_bytes=batch.bytes,geometry=prep,plan_upload=plan,adjacency=search,allocation=allocation,
                    first_launch=cold,warm_evaluation_and_reduction=warm,readback=readback,complete=complete,exhaustive=exhaustive,
                    localization_max_abs_difference=delta,fused_p1=consumer,fused_complete=field_complete,
                    transpose_with_upload_and_readback=transpose,material_device=material_cost,material_with_upload_readback=material_complete,affine_material_device=affine_cost,graph_replay=graph))
    quadrature_rows=[]
    for d in (2,3):
        vertices=np.vstack([np.zeros(d),.3*np.eye(d)])
        query=np.full(d,-.1)
        ev=Evaluator(Geometry([vertices],d),[Request((0,)*d)],device=str(device))
        batch=ev.prepare([query],1); batch.launch()
        native=batch.numpy()[0,0]
        _,native_warm=timed(batch.launch,device)
        _,native_complete=timed(lambda:ev.evaluate([query],1),device,repeats=3)
        reference=quadrature(vertices,query,d,64)
        for order in (16,32):
            value,cost=timed(lambda:quadrature(vertices,query,d,order),device,repeats=3)
            quadrature_rows.append(dict(dimension=d,order=order,nodes=order**d,query=query.tolist(),
                quadrature_mixed_error=abs(value-reference)/(1+abs(reference)),native_mixed_error=abs(native-reference)/(1+abs(reference)),timing=cost,native_warm=native_warm,native_complete=native_complete))
    higher=[]
    for d in (2,3):
        geometry=Geometry([np.vstack([np.zeros(d),2*np.eye(d)])],d)
        nodes=comb(3+d,d)
        rng=np.random.default_rng(52)
        factors=rng.normal(size=(1,nodes,d,d))
        resistance=factors@factors.swapaxes(-1,-2)+np.eye(d)*.2
        controls=rng.normal(size=(1,nodes,d))
        u=rng.normal(size=(1,d)); L=rng.normal(size=(1,d,d))
        reference=None
        for precision in ('float64','float32_compensated'):
            def prepare():
                material=AffineResistanceMaterial(geometry,resistance,controls,resistance_degree=3,velocity_degree=3)
                return material.prepare([[.2]*d],1,device=str(device),precision=precision)
            material_batch,preparation=timed(prepare,device,repeats=3)
            ev=material_batch.batch.evaluator
            velocity=wp.array(u,dtype=ev.api.real,device=device)
            masses=wp.ones(1,dtype=ev.api.real,device=device)
            gradient=wp.array(L,dtype=ev.api.real,device=device)
            material_batch.apply_device(velocity,masses,gradient)
            _,cost=timed(lambda:material_batch.apply_device(velocity,masses,gradient),device,repeats=3)
            result=material_batch.apply(u,[1.],L)
            _,complete_cost=timed(lambda:material_batch.apply(u,[1.],L),device,repeats=3)
            value=np.r_[result.fluid_force.ravel(),result.gradient_load.ravel(),result.reaction.ravel(),result.dissipation]
            if reference is None:reference=value
            higher.append(dict(dimension=d,precision=precision,queries=1,interactions=material_batch.batch.count,
                resistance_degree=3,velocity_degree=3,test_degree=3,moment_degree=material_batch.material.field.degree,
                requested_moments=len(ev.host_plan.requests),contracted_outputs=material_batch.batch.columns,
                batch_bytes=material_batch.batch.bytes,preparation=preparation,device_action=cost,upload_action_readback=complete_cost,
                mixed_difference_from_fp64=float(np.max(np.abs(value-reference)/(1+np.abs(reference))))))
    return rows,quadrature_rows,higher


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=evidence_path('fluxbridge-native'))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--fresh-cache',action='store_true',help='measure an initial compile in a new cache directory')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.fresh_cache:
        wp.config.kernel_cache_dir=tempfile.mkdtemp(prefix='warp-cache-',dir=args.output)
    wp.init()
    device=wp.get_device(args.device)
    lock=open('/tmp/fluxbridge-cuda.lock','a') if device.is_cuda else nullcontext()
    with lock as handle:
        if device.is_cuda:
            fcntl.flock(handle,fcntl.LOCK_EX)
        start=perf_counter()
        probe_start=perf_counter()
        probe=Evaluator(Geometry([[[0,0],[2,0],[0,2]]],2),baseline(2),device=str(device))
        probe_batch=probe.prepare([[.2,.2]],1)
        first_prepare_ms=(perf_counter()-probe_start)*1e3
        _,first_execution=timed(probe_batch.launch,device,repeats=1)
        accuracy_rows=accuracy(device)
        runtime_rows,quad,higher=runtime(device)
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
    source_root=Path(__file__).resolve().parents[2]
    sources=list((source_root/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))+list((source_root/'tests/fluxbridge').glob('*.py'))+[Path(__file__).resolve(),source_root/'scripts/make_fluxbridge_fixtures.py',source_root/'scripts/studies/fluxbridge_compile_cuda.py',source_root/'tests/fixtures/fluxbridge-v1.json']
    hashes={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report=dict(schema=1,source_sha256=hashes,timestamp_utc=datetime.now(timezone.utc).isoformat(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_dirty=True,fresh_cache=args.fresh_cache,cache_directory=str(wp.config.kernel_cache_dir),first_prepare_ms=first_prepare_ms,first_operator_execution=first_execution,device=str(device),cpu=cpu,platform=platform.platform(),warp=wp.__version__,
                cuda_available=wp.is_cuda_available(),cuda_validation='executed' if device.is_cuda else 'unavailable; not validated',
                accuracy=accuracy_rows,runtime=runtime_rows,quadrature_baseline=quad,cubic_affine=higher,elapsed_seconds=perf_counter()-start,
                limitations=['CUDA execution is verified only when cuda_validation is executed','raw fp32 W4/W6 and higher weighted derivatives not accepted at 1e-5',
                             'native special-form ports remain incomplete','CPU search and upload are not CUDA-native adjacency',
                             'point-only coupling is noncentral; affine coupling requires the supplied gradient transpose','no pressure or porosity integration'])
    path=args.output/'report.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(path)
    print('Worst fp64:',max(r['worst_mixed_error'] for r in accuracy_rows if r['precision']=='float64'))
    print('Worst fp32:',max(r['worst_mixed_error'] for r in accuracy_rows if r['precision']=='float32'))
    print('Worst compensated fp32:',max(r['worst_mixed_error'] for r in accuracy_rows if r['precision']=='float32_compensated'))
    for row in runtime_rows:
        print(row['dimension'],row['precision'],row['queries'],'warm ms',row['warm_evaluation_and_reduction']['median_ms'],'complete ms',row['complete']['median_ms'])

if __name__=='__main__':
    main()
