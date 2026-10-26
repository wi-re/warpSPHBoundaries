"""Actual-input stress and first matched CUDA timings for solid-only bundles."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge.solid import solid_functions,prepare_solid_geometry
from fluxbridge_solid_study import evaluate,metrics,ROOT
from fluxbridge_shell_acceptance import timed


def scene(d,mixed):
    if not mixed:return np.array([np.vstack([np.zeros(d),2*np.eye(d)])])
    rng=np.random.default_rng(729);result=[]
    for _ in range(256):
        rotation=np.linalg.qr(rng.normal(size=(d,d)))[0]
        edges=rotation@(np.diag(rng.uniform(.5,2,d))+np.triu(rng.uniform(-.2,.2,(d,d)),1))
        origin=rng.uniform(-10,10,d);result.append(np.vstack([origin,origin+edges.T]))
    return np.array(result)


def inputs(d,vertices,n,mixed):
    rng=np.random.default_rng(852);ids=rng.integers(0,len(vertices),n,dtype=np.int32) if mixed else np.zeros(n,np.int32)
    if mixed:
        edges=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1)
        q=vertices[ids,0]+np.einsum('nij,nj->ni',edges[ids],rng.uniform(-.5,1.5,(n,d)));h=rng.uniform(.3,2,n)
    else:q=.2+rng.uniform(-.015,.015,(n,d));h=np.ones(n)
    return q.astype(np.float32).astype(float),h.astype(np.float32).astype(float),ids


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:0');p.add_argument('--benchmark',action='store_true');p.add_argument('--mixed',action='store_true');p.add_argument('--count',type=int,default=8192);p.add_argument('--output',type=Path,required=True);p.add_argument('--dimensions',default='2,3')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not 1<=a.count<=1048576:p.error('count outside [1,1048576]')
    if a.benchmark and not a.device.startswith('cuda'):p.error('timings require actual CUDA')
    paths=[Path(__file__),ROOT/'scripts/studies/fluxbridge_solid_study.py',ROOT/'scripts/studies/fluxbridge_shell_acceptance.py']+list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*solid*.py'))
    report=dict(device=a.device,pairs=a.count,mixed=a.mixed,benchmark=a.benchmark,rows=[],geometry=[],block_dim=256,
        reference='Native analytic fp64 at the actual fp32 vertex/query/support inputs; fixture accuracy is recorded separately',
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
    save=lambda:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device)
        for d in map(int,a.dimensions.split(',')):
            if d not in (2,3):raise ValueError('dimension must be 2/3')
            vertices=scene(d,a.mixed).astype(np.float32).astype(float);q,h,ids=inputs(d,vertices,a.count,a.mixed)
            geometries={}
            for precision,conditioning in [('float64','raw'),('float32','raw'),('float32','protected')]:
                begin=time.perf_counter();g=prepare_solid_geometry(vertices,device,precision,conditioning);wp.synchronize_device(device)
                geometries[(precision,conditioning)]=g
                report['geometry'].append(dict(dimension=d,precision=precision,conditioning=conditioning,prepare_upload_ms=(time.perf_counter()-begin)*1000,bytes=g.data.capacity+g.edges.capacity+(g.data_low.capacity if conditioning=='protected' else 0)))
            for k in ('w2','w4'):
                reference=evaluate(solid_functions(d,k,'float64','lean'),geometries[('float64','raw')],q,h,ids,device).astype(float)
                for precision,conditioning in [('float64','raw'),('float32','raw'),('float32','protected')]:
                    g=geometries[(precision,conditioning)]
                    for bundle in ('density','pressure','lean'):
                        api=solid_functions(d,k,precision,bundle,conditioning=conditioning)
                        positions=wp.array(np.pad(q,((0,0),(0,3-d))),dtype=api.vec,device=device);supports=wp.array(h,dtype=api.real,device=device);indices=wp.array(ids,dtype=int,device=device)
                        out=wp.empty((api.channels,a.count),dtype=api.real,device=device);args=[positions,supports,indices,g,out]
                        action=lambda:wp.launch(api.kernel,a.count,inputs=args,device=device,block_dim=256)
                        action();value=out.numpy();budget=1e-10 if precision=='float64' else 1e-5
                        row=dict(dimension=d,kernel=k,precision=precision,conditioning=conditioning,bundle=bundle,channels=api.channels,
                            metrics=metrics(value.T.astype(float),reference[:,:api.channels],budget),input_output_bytes=sum(x.capacity for x in [positions,supports,indices,out]))
                        if device.is_cuda:row['kernel_properties']=wp.get_cuda_kernel_properties(api.kernel,device)
                        if a.benchmark:
                            _,row['warm']=timed(action,device,7,events=True)
                            with wp.ScopedCapture(device=device) as cap:action()
                            _,row['graph']=timed(lambda:wp.capture_launch(cap.graph),device,7,events=True)
                            row['graph_bitwise_equal']=bool(np.array_equal(value,out.numpy()));assert row['graph_bitwise_equal']
                        else:np.savez_compressed(a.output/f'{d}-{k}-{precision}-{conditioning}-{bundle}.npz',actual=value.T,reference=reference[:,:api.channels])
                        report['rows'].append(row);save();print(d,k,precision,conditioning,bundle,row['metrics'],row.get('warm',{}).get('cuda_event_median_ms'),flush=True)
        report['protected_passed']=all(r['metrics']['passed'] for r in report['rows'] if r['conditioning']=='protected');save()


if __name__=='__main__':main()
