"""Weighted disk accuracy, complete direct/solid-array costs and clean pool use."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Request,baseline,WeightedDiskEvaluator
from warpSPHBoundaries.fluxbridge.disks import DiskArrayEvaluator
from fluxbridge_shell_acceptance import timed

ROOT=Path(__file__).resolve().parents[2]


def sources():
    paths=list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))+[Path(__file__),
        ROOT/'scripts/studies/fluxbridge_shell_acceptance.py',ROOT/'scripts/make_fluxbridge_weighted_disk_fixtures.py',
        ROOT/'tests/fixtures/fluxbridge-weighted-disk-v1.json',ROOT/'tests/fluxbridge/test_weighted_disk.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def pool(device):
    if not device.is_cuda:return None
    return dict(current_used_bytes=wp.get_mempool_used_mem_current(device),
        process_high_water_used_bytes=wp.get_mempool_used_mem_high(device),free_device_bytes=device.free_memory)


def accuracy(device,cases,nodes,precisions):
    rows=[]
    for precision in precisions:
        for kernel in ['w2','w4','w6','cubic']:
            budget=1e-5 if precision=='float32' else 1e-14
            worst=0.;channels=0;promotions=0;orders={}
            for c in cases:
                if c['kernel']!=kernel:continue
                e=WeightedDiskEvaluator([Request(**r) for r in c['requests']],kernel=kernel,precision=precision,device=device,nodes=nodes)
                b=e.prepare(float(c['distance']),c['normal'],1,radii=float(c['radius']))
                value=b.numpy()[0];reference=np.array(c['values'],float)
                error=float(np.max(abs(value-reference)/(1+abs(reference))))
                if not np.isfinite(value).all() or error>budget:raise RuntimeError((precision,kernel,c['label'],error))
                worst=max(worst,error);channels+=len(value);promotions+=b.quadrature['promotions']
                order=str(b.quadrature['nodes']);orders[order]=orders.get(order,0)+1
            rows.append(dict(precision=precision,kernel=kernel,requested_nodes=nodes,channels=channels,
                worst_mixed_error=worst,budget=budget,promotions=promotions,accepted_orders=orders,
                failing_channels=0,rejected_cases=0,fallbacks=0))
            print('accuracy',precision,kernel,channels,worst,orders,flush=True)
    return rows


def workload(device,requests,precision,nodes,count,scope,stream_chunk=0):
    rng=np.random.default_rng(812);normal=rng.normal(size=(count,2));normal/=np.linalg.norm(normal,axis=1)[:,None]
    d=rng.uniform(-.2,.3,count)
    options=dict(precision=precision,device=device,nodes=nodes)
    if scope=='direct':
        provider,static=timed(lambda:WeightedDiskEvaluator(requests,**options),device,1)
        prepare=lambda:provider.prepare(d,normal,1,radii=.5)
        evaluate=lambda:provider.evaluate(d,normal,1,radii=.5)
    else:
        centres=np.c_[4*np.arange(16),np.zeros(16)]
        q=centres[np.arange(count)%16]+(.5+d)[:,None]*normal
        provider,static=timed(lambda:DiskArrayEvaluator(centres,.5,requests,method='volume_quadrature',search='device',**options),device,1)
        prepare=lambda:provider.prepare(q,1)
        evaluate=(lambda:provider.evaluate_streamed(q,1,chunk_size=stream_chunk,return_metrics=True)) if stream_chunk else (lambda:provider.evaluate(q,1))
    return provider,prepare,evaluate,static


def runtime(device,requests,precision,nodes,count,scope,memory_only=False,stream_chunk=0):
    provider,prepare,evaluate,static=workload(device,requests,precision,nodes,count,scope,stream_chunk)
    if memory_only:
        # No retained accepted batch, fp64 reference or preceding large case.
        result,cost=timed(evaluate,device,3)
        return dict(precision=precision,requested_nodes=nodes,queries=count,scope=scope,outputs=len(requests),
            complete_evaluation=cost,static_preparation=static,pool_after=pool(device),fresh_memory_worker=True,
            stream_chunk=stream_chunk,execution=result[1] if stream_chunk else None,
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    b,first=timed(prepare,device,1)
    _,warm=timed(b.launch,device,3,events=device.is_cuda)
    _,complete=timed(evaluate,device,3)
    graph=None
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:b.launch()
        before=b.numpy().copy();wp.capture_launch(capture.graph)
        if not np.array_equal(before,b.numpy()):raise RuntimeError('disk graph output changed')
        _,graph=timed(lambda:wp.capture_launch(capture.graph),device,3,events=True)
    result=dict(precision=precision,requested_nodes=nodes,queries=count,scope=scope,outputs=len(requests),kernel='w2',
        static_preparation=static,first_checked_preparation=first,warm_action=warm,complete_evaluation=complete,
        graph_replay=graph,batch_owned_bytes=b.bytes,quadrature=b.quadrature,pool_after=pool(device))
    if scope=='array':
        result.update(elements=16,interactions=b.count,search='device',
            static_search_array_bytes=provider.device_search.array_bytes,
            static_search_opaque_bvh_bytes='not included in array bytes')
    print('runtime',precision,scope,len(requests),count,round(warm['median_ms'],3),round(complete['median_ms'],3),flush=True)
    return result


def compile_only(precision,output,nodes):
    import warp._src.build as build
    WeightedDiskEvaluator(precision=precision,nodes=nodes).prepare([], [0,1],1,radii=.5)
    module=wp.get_module('warpSPHBoundaries.fluxbridge.weighted_disk')
    options=dict(module.options);options['block_dim']=256
    code,extension,_,ltoirs,fatbins=module._run_codegen(options,is_cpu=False)
    path=output/f'disk-{precision}.{extension}';path.write_text(code)
    ptx=output/f'disk-{precision}.ptx'
    build.build_cuda(str(path),120,str(ptx),pch_dir=None,use_precompiled_headers=False,
        fast_math=False,fuse_fp=False,ltoirs=ltoirs,fatbins=fatbins)
    instructions=[s.strip() for s in ptx.read_text().splitlines() if '.f64' in s and not s.strip().startswith(('.','//'))]
    if precision=='float32' and instructions:raise RuntimeError('unexpected fp64 instructions')
    return dict(precision=precision,arch=120,fp64_instructions=len(instructions),compile_passed=True,execution_tested=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nodes',type=int,default=24,choices=range(8,65));p.add_argument('--precisions',default='float32,float64')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--compile-only',choices=['float32','float64'])
    mode.add_argument('--memory-only',action='store_true');mode.add_argument('--accuracy-only',action='store_true')
    p.add_argument('--memory-count',type=int,default=128);p.add_argument('--memory-scope',choices=['direct','array'],default='direct')
    p.add_argument('--memory-stream-chunk',type=int,default=0)
    args=p.parse_args();precisions=args.precisions.split(',')
    if any(v not in ('float32','float64') for v in precisions) or not 1<=args.memory_count<=8192:p.error('invalid precision/count')
    if args.memory_only and len(precisions)!=1:p.error('fresh memory worker needs one precision')
    if args.memory_stream_chunk and (not args.memory_only or args.memory_scope!='array' or not 1<=args.memory_stream_chunk<=8192):p.error('stream chunk requires a memory-only array worker')
    args.output.mkdir(parents=True,exist_ok=True);wp.init();device=wp.get_device(args.device)
    if args.compile_only:report=dict(compilation=compile_only(args.compile_only,args.output,args.nodes),source_sha256=sources())
    else:
        cases=json.loads((ROOT/'tests/fixtures/fluxbridge-weighted-disk-v1.json').read_text())['cases']
        full=[Request(**r) for r in cases[0]['requests']]
        with (open('/tmp/fluxbridge-gpu.lock','w') if device.is_cuda else nullcontext(None)) as lock:
            if lock:fcntl.flock(lock,fcntl.LOCK_EX)
            values=[] if args.memory_only else accuracy(device,cases,args.nodes,precisions)
            costs=[]
            if args.memory_only:costs=[runtime(device,full,precisions[0],args.nodes,args.memory_count,args.memory_scope,True,args.memory_stream_chunk)]
            elif not args.accuracy_only:
                for precision in precisions:
                    for req in (baseline(2),full):
                        for count in (1,128):costs.append(runtime(device,req,precision,args.nodes,count,'direct'))
                    for count in (128,512):costs.append(runtime(device,full,precision,args.nodes,count,'array'))
            report=dict(schema=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),warp=wp.__version__,device=str(device),
                device_name=device.name,cuda_runtime_tested=device.is_cuda,source_sha256=sources(),accuracy=values,runtime=costs,
                memory_only=args.memory_only,limitations=['exact 2D disk geometry, host fp64 radial preparation, selected-dtype native evaluation',
                    'two-order rule comparison and finite fixtures are not certified global error/positivity bounds',
                    'complete direct costs include preparation, both rules, allocation, storage validation and readback',
                    'array complete costs additionally include adjacency and deterministic reduction; static BVH construction is separate',
                    'owned bytes exclude comparison batch, static plans/BVH, host geometry temporaries and allocator reservation',
                    'pool high-water is cumulative; only --memory-only is a fresh one-workload worker without retained accepted batch',
                    'CUDA pool usage is not total context/reservation/opaque memory; no porous drag or solver is attached to resolved fibres'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(args.output/'report.json',flush=True)


if __name__=='__main__':main()
