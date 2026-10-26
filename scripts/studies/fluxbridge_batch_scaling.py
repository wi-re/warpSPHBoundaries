"""Serialized real material batch scaling, including checked preparation/rejection.

Queries vary within one tetrahedron: every row has one interaction and its own
query-dependent rule. No repeated-query integral cache is used. Static material
construction is reported separately from per-batch allocation/preparation.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry,AffineResistanceMaterial
from fluxbridge_shell_acceptance import timed

ROOT=Path(__file__).resolve().parents[2]


def source_hashes():
    paths=list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))+[Path(__file__),ROOT/'scripts/studies/fluxbridge_shell_acceptance.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def build_material(elements=1):
    g=Geometry([np.vstack([np.zeros(3),2*np.eye(3)])+np.array([4*i,0,0]) for i in range(elements)],3)
    rng=np.random.default_rng(52);f=rng.normal(size=(elements,20,3,3))
    controls=rng.normal(size=(elements,20,3))
    return (AffineResistanceMaterial(g,f@f.swapaxes(-1,-2)+np.eye(3)*.2,
        controls,resistance_degree=3,velocity_degree=3),controls)


def run(device,counts,kernels,analytic_max,preparation,reuse_material=False,search="host",elements=1,nodes=16):
    (material,controls),construction=timed(lambda:build_material(elements),device,1)
    rows=[];compiled={};static_cost={}
    for kernel in kernels:
        for count in counts:
            rng=np.random.default_rng(852)
            q=.2+rng.uniform(-.015,.015,(count,3))
            if elements>1:q[:,0]+=4*(np.arange(count)%elements)
            u=rng.normal(size=(count,3));L=rng.normal(size=(count,3,3));m=np.full(count,.7/count)
            references={}
            methods=[('analytic','float64')] if count<=analytic_max else []
            methods += [('tet_quadrature','float64'),('tet_quadrature','float32')]
            for method,precision in methods:
                options=dict(kernel=kernel,method=method,precision=precision,device=device,
                    preparation=preparation if method=='tet_quadrature' else 'host',search=search,nodes=nodes)
                key=(kernel,method,precision)
                if reuse_material and key not in compiled:
                    compiled[key],static_cost[key]=timed(lambda:material.compile(**options),device,1)
                prepare=(lambda:compiled[key].prepare(q,1)) if reuse_material else (lambda:material.prepare(q,1,**options))
                start=time.perf_counter()
                try:
                    b,prep=timed(prepare,device,1)
                except ValueError as error:
                    row=dict(kernel=kernel,queries=count,method=method,precision=precision,status='rejected',
                        reason=str(error),preparation_until_rejection_ms=1000*(time.perf_counter()-start))
                    if 'storage exceeds' not in str(error):raise
                    rows.append(row);print(row,flush=True);gc.collect();continue
                real=b.batch.evaluator.api.real
                ud=wp.array(u,dtype=real,device=device);Ld=wp.array(L,dtype=real,device=device)
                md=wp.array(m,dtype=real,device=device)
                apply=lambda:b.apply_device(ud,md,Ld)
                _,first=timed(apply,device,1)
                _,warm=timed(apply,device,3,events=True)
                result=b.apply(u,m,L)
                packed=b.batch.numpy().copy()
                action=np.r_[result.fluid_force.ravel(),result.gradient_load.ravel(),result.reaction.ravel(),result.dissipation]
                work=float(np.sum(result.fluid_force*u)+np.sum(result.gradient_load*L)+np.sum(result.reaction*controls))
                balance=float(np.linalg.norm(result.fluid_force.sum(axis=0)+result.reaction.sum(axis=(0,1))))
                force_scale=1+float(np.linalg.norm(result.fluid_force,axis=-1).sum()+np.linalg.norm(result.reaction,axis=-1).sum())
                normalized_balance=balance/force_scale
                normalized_work=abs(work+result.dissipation)/(1+abs(result.dissipation))
                reference=references.get('analytic') if method=='tet_quadrature' else None
                if reference is None and precision=='float32':reference=references.get('tet_quadrature')
                mixed_action=None if reference is None else float(np.max(abs(action-reference[0])/(1+abs(reference[0]))))
                mixed_packed=None if reference is None else float(np.max(abs(packed-reference[1])/(1+abs(reference[1]))))
                budget=1e-5 if precision=='float32' else 1e-10
                if max(normalized_work,normalized_balance,0 if mixed_action is None else mixed_action,0 if mixed_packed is None else mixed_packed)>budget:
                    raise RuntimeError(dict(work=normalized_work,balance=balance,normalized_balance=normalized_balance,force_scale=force_scale,action=mixed_action,packed=mixed_packed))
                if precision=='float64':references[method]=(action,packed)
                _,complete=timed(lambda:prepare().apply(u,m,L),device,1)
                full_complete=complete
                if reuse_material:
                    _,full_complete=timed(lambda:material.prepare(q,1,**options).apply(u,m,L),device,1)
                graph=None
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:apply()
                    before=[a.numpy() for a in (b.force,b.gradient_load,b.reaction,b.power)]
                    wp.capture_launch(capture.graph)
                    if not all(np.array_equal(a.numpy(),v) for a,v in zip((b.force,b.gradient_load,b.reaction,b.power),before)):
                        raise RuntimeError('graph result changed')
                    _,graph=timed(lambda:wp.capture_launch(capture.graph),device,3,events=True)
                    del capture
                row=dict(kernel=kernel,queries=count,interactions=b.batch.count,method=method,precision=precision,
                    status='passed',preparation=preparation if method=='tet_quadrature' else 'host',requested_moments=len(b.batch.evaluator.host_plan.requests),outputs=b.batch.columns,
                    static_material_reused=reuse_material,search=search,elements=elements,requested_nodes=nodes,static_material_compilation=static_cost.get(key),
                    checked_preparation=prep,first_action=first,warm_action=warm,reprepare_and_apply=complete,
                    full_static_rebuild_and_apply=full_complete,
                    graph_replay=graph,batch_bytes=b.bytes,field_batch_bytes=b.batch.bytes,material_action_bytes=b.bytes-b.batch.bytes,
                    field_weight_workspace_bytes=b.batch.field_weights.capacity+b.batch.field_weight_errors.capacity,
                    static_field_bytes=b.batch.owner.coefficients.capacity,
                    search_static_array_bytes=0 if b.batch.evaluator.device_search is None else b.batch.evaluator.device_search.array_bytes,
                    quadrature=getattr(b.batch,'quadrature',None),normalized_work_defect=normalized_work,
                    force_balance_norm=balance,force_scale=force_scale,normalized_force_defect=normalized_balance,dissipation=result.dissipation,mixed_action_difference=mixed_action,
                    mixed_packed_difference=mixed_packed,accuracy_budget=budget,
                    host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                rows.append(row)
                print(kernel,count,method,precision,'warm',round(warm['median_ms'],3),'complete',round(complete['median_ms'],3),
                    'MB',round(b.bytes/1e6,3),flush=True)
                del b,ud,Ld,md,apply,result,packed,prepare
                gc.collect()
    return construction,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--counts',default='1,8,32,128')
    p.add_argument('--kernels',default='w2,w6');p.add_argument('--analytic-max',type=int,default=8)
    p.add_argument('--preparation',choices=['host','device'],default='host')
    p.add_argument('--reuse-material',action='store_true')
    p.add_argument('--search',choices=['host','device'],default='host')
    p.add_argument('--elements',type=int,default=1)
    p.add_argument('--nodes',type=int,default=16,choices=range(8,65))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    counts=[int(v) for v in args.counts.split(',')];kernels=args.kernels.split(',')
    if not 1<=args.elements<=64 or not counts or min(counts)<1 or max(counts)>256 or any(k not in ('w2','w4','w6','cubic') for k in kernels):p.error('invalid counts/kernels')
    args.output.mkdir(parents=True,exist_ok=True);wp.init();device=wp.get_device(args.device)
    with (open('/tmp/fluxbridge-gpu.lock','w') if device.is_cuda else nullcontext(None)) as lock:
        if lock:fcntl.flock(lock,fcntl.LOCK_EX)
        construction,rows=run(device,counts,kernels,args.analytic_max,args.preparation,args.reuse_material,args.search,args.elements,args.nodes)
        report=dict(schema=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),device=str(device),device_name=device.name,
            cuda_runtime_tested=device.is_cuda,warp=wp.__version__,source_sha256=source_hashes(),material_construction=construction,
            rows=rows,limitations=['separated tetrahedra with varied query positions, one interaction per row; element count recorded per row',
                'compiled static material cost is separate when --reuse-material is selected; full rebuild cost is also reported',
                'P3 resistance, P3 skeleton and P3 tests, degree-nine products, 220 moments and 910 outputs',
                'mixed differences use analytic fp64 when present, otherwise numerical fp64; independent fixture evidence is separate',
                'explicit host/device angular preparation, fp64 geometry, selected-dtype evaluation, fixed caches and memory rejection',
                'force gate uses absolute norm/(1+sum fluid force norms+sum nodal reaction norms); absolute defect also recorded',
                'shared machine, serialized measurements, three warm samples and one complete-cost sample',
                'ordinary selected-dtype moments and coefficients; shared field translation/dot products retain explicit same-dtype FMA residual workspace',
                'batch bytes include field/rule/adjacency and material action arrays; static coefficients/BVH and comparison temporaries are separate',
                'reported search arrays exclude opaque BVH storage and scan/sort allocator scratch; host RSS is a process high-water mark'])
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(args.output/'report.json',flush=True)


if __name__=='__main__':main()
