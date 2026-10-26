"""Sustained matched lifetime/cap configurations with rotating order."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import warp as wp
import fluxbridge_shared_jk_lifetime as impl
import fluxbridge_shared_jk_stability as paired
from fluxbridge_shared_jk_full_study import arrays
from fluxbridge_shared_jk_consumers import scene
from fluxbridge_single_term_study import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--configurations',default='control:256,control@128:256,serial:128')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);wp.init();device=wp.get_device('cuda:0')
        report=dict(device=device.name,pairs=1048576,launches_per_batch=32,rounds=3,samples_per_round=7,rows=[],
            scope='Per-launch CUDA-event duration of 32 sequential graph launches; preallocated identical buffers; rotated variant order each round; no profiler',
            source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py')})
        configuration=[(v,int(b)) for v,b in (c.split(':') for c in a.configurations.split(','))];variants=[v for v,b in configuration];blocks=dict(configuration);n=report['pairs'];report['configurations']=configuration
        if len(variants)!=len(set(variants)) or any(v not in impl.VARIANTS or b not in (32,64,128,256) for v,b in configuration):raise ValueError('invalid configurations')
        for mixed in [False,True]:
            vertices=scene(mixed);g=paired.prepare_geometry(vertices,device,'floatpair-inline');rng=np.random.default_rng(852)
            ids=rng.integers(0,len(vertices),n,dtype=np.int32) if mixed else np.zeros(n,int)
            if mixed:
                edges=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1)
                q=vertices[ids,0]+np.einsum('nij,nj->ni',edges[ids],rng.uniform(-.5,1.5,(n,3)));h=rng.uniform(.3,2,n)
            else:q=.2+rng.uniform(-.015,.015,(n,3));h=np.ones(n)
            args,out=arrays(impl.functions('control'),g,q,h,ids,device);graphs={};initial={}
            for v in variants:
                api=impl.functions(v);wp.launch(api.kernel,n,inputs=args,device=device,block_dim=blocks[v]);wp.synchronize_device(device)
                initial[v]=out.numpy()
                with wp.ScopedCapture(device=device) as cap:
                    for _ in range(report['launches_per_batch']):wp.launch(api.kernel,n,inputs=args,device=device,block_dim=blocks[v])
                graphs[v]=cap.graph
            # Warm every graph, keeping both kernel and clocks active before
            # collecting the rotating-order rounds.
            for v in variants:wp.capture_launch(graphs[v])
            wp.synchronize_device(device)
            for round in range(report['rounds']):
                for v in variants[round:]+variants[:round]:
                    samples=[]
                    for _ in range(report['samples_per_round']):
                        start=wp.Event(device,enable_timing=True);end=wp.Event(device,enable_timing=True)
                        wp.record_event(start);wp.capture_launch(graphs[v]);wp.record_event(end);wp.synchronize_event(end)
                        samples.append(wp.get_event_elapsed_time(start,end)/report['launches_per_batch'])
                    stable=bool(np.array_equal(out.numpy(),initial[v]));assert stable
                    report['rows'].append(dict(mixed=mixed,round=round,variant=v,block_dim=blocks[v],event_samples_ms_per_launch=samples,median_ms=statistics.median(samples),graph_bitwise_equal=stable))
                    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                    print(mixed,round,v,statistics.median(samples),flush=True)


if __name__=='__main__':main()
