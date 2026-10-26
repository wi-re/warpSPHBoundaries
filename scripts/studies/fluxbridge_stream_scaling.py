"""Serialized complete streamed P3 cost, rejection and allocator measurements.

Use a fresh --memory-only process per case for a CUDA-pool high-water mark
uncontaminated by a double-precision reference or failed full preparation.
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
from fluxbridge_batch_scaling import build_material
from fluxbridge_shell_acceptance import timed

ROOT=Path(__file__).resolve().parents[2]


def hashes():
    paths=list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))
    paths += [Path(__file__),ROOT/'scripts/studies/fluxbridge_batch_scaling.py',ROOT/'scripts/studies/fluxbridge_shell_acceptance.py',ROOT/'tests/fluxbridge/test_streaming.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def pool(device):
    if not device.is_cuda:return None
    return dict(current_used_bytes=wp.get_mempool_used_mem_current(device),
        process_high_water_used_bytes=wp.get_mempool_used_mem_high(device),free_device_bytes=device.free_memory)


def checks(result,u,L,controls,reference=None):
    work=float(np.sum(result.fluid_force*u)+np.sum(result.gradient_load*L)+np.sum(result.reaction*controls))
    balance=float(np.linalg.norm(result.fluid_force.sum(axis=0)+result.reaction.sum(axis=(0,1))))
    scale=1+float(np.linalg.norm(result.fluid_force,axis=-1).sum()+np.linalg.norm(result.reaction,axis=-1).sum())
    defects=dict(normalized_work_defect=abs(work+result.dissipation)/(1+abs(result.dissipation)),
        absolute_force_defect=balance,force_scale=scale,normalized_force_defect=balance/scale,dissipation=result.dissipation)
    if reference is not None:
        defects['mixed_differences']={name:float(np.max(abs(np.asarray(getattr(result,name))-getattr(reference,name))/(1+abs(np.asarray(getattr(reference,name))))))
            for name in ['fluid_force','gradient_load','reaction','resistance','skeleton_product','dissipation']}
    worst=max(defects['normalized_work_defect'],defects['normalized_force_defect'],
        max(defects.get('mixed_differences',{}).values(),default=0))
    if worst>1e-5 or result.dissipation<0:raise RuntimeError(defects)
    return defects


def run(device,counts,sizes,kernels,repeats,memory_only,nodes):
    (material,controls),construction=timed(lambda:build_material(16),device,1)
    rows=[]
    for kernel in kernels:
        provider,static=timed(lambda:material.compile(device=device,kernel=kernel,precision='float32',
            method='tet_quadrature',preparation='device',search='device',nodes=nodes),device,1)
        reference_provider=None;reference_static=None
        if not memory_only:
            reference_provider,reference_static=timed(lambda:material.compile(device=device,kernel=kernel,precision='float64',
                method='tet_quadrature',preparation='device',search='device'),device,1)
        # Initialize the runtime kernels outside steady complete-driver timings.
        _,first=timed(lambda:provider.apply_streamed([[.2]*3],1,[[.1,.2,.3]],1,[np.eye(3)*.1],chunk_size=1),device,1)
        if reference_provider is not None:reference_provider.apply_streamed([[.2]*3],1,[[.1,.2,.3]],1,[np.eye(3)*.1],chunk_size=1)
        for n in counts:
            rng=np.random.default_rng(852);q=.2+rng.uniform(-.015,.015,(n,3));q[:,0]+=4*(np.arange(n)%16)
            u=rng.normal(size=q.shape);L=rng.normal(size=(n,3,3));mass=np.full(n,.7/n)
            values=[];pending=[]
            for size in sizes:
                before=pool(device)
                result,cost=timed(lambda:provider.apply_streamed(q,1,u,mass,L,chunk_size=size),device,repeats)
                after=pool(device)
                defects=checks(result,u,L,controls)
                row=dict(kernel=kernel,queries=n,chunk_size=size,requested_nodes=nodes,status='passed_common_work',precision='float32',
                    material_construction=construction,static_compilation=static,initial_one_row_action=first,
                    complete_streamed_action=cost,execution=result.execution,pool_before=before,pool_after=after,
                    static_coefficient_bytes=provider.evaluator.coefficients.capacity,
                    host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**defects)
                pending.append(row);values.append(result)
                print(kernel,n,size,'complete',round(cost['median_ms'],3),'owned MB',round(result.execution['peak_owned_batch_bytes']/1e6,3),flush=True)
            reference_cost=None
            if reference_provider is not None:
                reference,reference_cost=timed(lambda:reference_provider.apply_streamed(q,1,u,mass,L,chunk_size=32),device,1)
                for result,row in zip(values,pending):
                    row.update(checks(result,u,L,controls,reference));row['status']='passed'
                    row['fp64_reference_cost']=reference_cost;row['fp64_static_compilation']=reference_static
                del reference
            full=None
            if not memory_only:
                began=time.perf_counter()
                try:
                    b=provider.prepare(q,1);r=b.apply(u,mass,L)
                    full=dict(status='passed',complete_ms=1000*(time.perf_counter()-began),owned_batch_bytes=b.bytes,
                        checks=checks(r,u,L,controls))
                    del b,r
                except ValueError as error:
                    if 'storage exceeds' not in str(error):raise
                    full=dict(status='rejected',reason=str(error),until_rejection_ms=1000*(time.perf_counter()-began))
                for row in pending:row['single_batch']=full
                print('single batch',kernel,n,full['status'],flush=True)
            rows.extend(pending);del values,pending;gc.collect()
        del provider,reference_provider;gc.collect()
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--counts',default='128,512,2048')
    p.add_argument('--chunk-sizes',default='32,64');p.add_argument('--kernels',default='w2,w6');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--nodes',type=int,default=16,choices=range(8,65));p.add_argument('--memory-only',action='store_true');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    counts=[int(v) for v in args.counts.split(',')];sizes=[int(v) for v in args.chunk_sizes.split(',')];kernels=args.kernels.split(',')
    if min(counts+sizes)<1 or max(counts)>8192 or max(sizes)>128 or not 1<=args.repeats<=10 or any(k not in ('w2','w4','w6','cubic') for k in kernels):p.error('invalid study bounds')
    if args.memory_only and (len(counts)!=1 or len(sizes)!=1 or len(kernels)!=1):p.error('--memory-only requires one count, chunk size and kernel per fresh process')
    args.output.mkdir(parents=True,exist_ok=True);wp.init();device=wp.get_device(args.device)
    with (open('/tmp/fluxbridge-gpu.lock','w') if device.is_cuda else nullcontext(None)) as lock:
        if lock:fcntl.flock(lock,fcntl.LOCK_EX)
        rows=run(device,counts,sizes,kernels,args.repeats,args.memory_only,args.nodes)
        report=dict(schema=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),device=str(device),device_name=device.name,
            cuda_runtime_tested=device.is_cuda,warp=wp.__version__,source_sha256=hashes(),memory_only=args.memory_only,rows=rows,
            limitations=['16 separated tetrahedra, one interaction per varied query, P3 fields, 220 moments and 910 packed outputs',
                'complete host driver cost includes preparation, convergence, allocation, native actions, transfers and deterministic host reaction aggregation',
                'static providers and initial runtime/module warmup are reported separately; every timed action recomputes integrals',
                'streamed results compare with streamed fp64; independent 70-digit moment/material acceptance is separate',
                'owned-byte maximum excludes comparison temporaries, static plans/BVH and allocator reservation',
                'CUDA pool current/high-water are actual application-pool usage, not total GPU reservation or context/opaque allocation',
                'pool high-water is cumulative for this process; --memory-only excludes fp64 references and rejected full batches',
                'no graph capture for the host streaming driver; ordinary prepared batches retain graph-compatible actions'])
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(args.output/'report.json',flush=True)


if __name__=='__main__':main()
