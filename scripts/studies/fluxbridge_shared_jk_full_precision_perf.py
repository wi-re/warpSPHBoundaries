"""Speed diagnostic for the failing raw-fp32 full-28 analytic prototype.

Same logical workloads/output contract for both precisions. Accuracy failures
are documented separately; no production use or fp32 acceptance is implied.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
import fluxbridge_shared_jk_full_precision as precision
import fluxbridge_shared_jk_layout as geometry
from fluxbridge_shared_jk_full_study import arrays
from fluxbridge_shared_jk_consumers import scene
from fluxbridge_shell_acceptance import timed
from fluxbridge_single_term_study import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts',default='512,8192,131072,1048576')
    p.add_argument('--blocks',default='64,256')
    p.add_argument('--precisions',default='float32,float64')
    p.add_argument('--mixed',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();counts=list(map(int,a.counts.split(',')));blocks=list(map(int,a.blocks.split(',')));precisions=a.precisions.split(',')
    if any(n<1 or n>1048576 for n in counts) or any(b not in (32,64,128,256) for b in blocks) or any(t not in ('float32','float64') for t in precisions):p.error('invalid counts, blocks or precision')
    if a.profile and (len(counts)!=1 or len(blocks)!=1 or len(precisions)!=1):p.error('profile requires one count/block/precision')
    a.output.mkdir(parents=True,exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device('cuda:0');vertices=scene(a.mixed)
        paths=list((ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py'))+[ROOT/'scripts/studies/fluxbridge_shell_acceptance.py']
        report=dict(device_name=device.name,mixed=a.mixed,accuracy_status='raw fp32 fails 1e-5 gate; performance diagnostic only',rows=[],static_setup={},
            source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
        save=lambda:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        geometries={}
        for dtype in precisions:
            geometries[dtype],timing=timed(lambda:geometry.prepare_geometry(vertices,device,dtype),device,1)
            report['static_setup'][dtype]=dict(timing=timing,bytes=geometry.geometry_bytes(geometries[dtype]))
            (a.output/(dtype+'-native.cpp')).write_text(precision.source(dtype))
        for count in counts:
            rng=np.random.default_rng(852)
            indices=rng.integers(0,len(vertices),count,dtype=np.int32) if a.mixed else np.zeros(count,int)
            if a.mixed:
                edges=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1)
                q=vertices[indices,0]+np.einsum('nij,nj->ni',edges[indices],rng.uniform(-.5,1.5,(count,3)));h=rng.uniform(.3,2,count)
            else:q=.2+rng.uniform(-.015,.015,(count,3));h=np.ones(count)
            for dtype in precisions:
                api=precision.functions(dtype);g=geometries[dtype];args,out=arrays(api,g,q,h,indices,device)
                reference=None
                for block in blocks:
                    action=lambda:wp.launch(api.kernel,count,inputs=args,device=device,block_dim=block)
                    action();wp.synchronize_device(device);value=out.numpy()
                    if reference is None:reference=value
                    row=dict(pairs=count,block_dim=block,precision=dtype,outputs=28,input_output_bytes=sum(x.capacity for x in args if isinstance(x,wp.array)),
                        kernel_properties=wp.get_cuda_kernel_properties(api.kernel,device),finite=bool(np.isfinite(value).all()),
                        bitwise_equal_across_launch_sizes=bool(np.array_equal(reference,value)))
                    if not(row['finite'] and row['bitwise_equal_across_launch_sizes']):raise RuntimeError('nonfinite or changed launch outputs')
                    if a.profile:
                        wp.cuda_profiler_start(device);action();wp.synchronize_device(device);wp.cuda_profiler_stop(device)
                    else:
                        _,row['warm']=timed(action,device,7,events=True)
                        with wp.ScopedCapture(device=device) as cap:action()
                        _,row['graph']=timed(lambda:wp.capture_launch(cap.graph),device,7,events=True)
                        wp.synchronize_device(device)
                        row['graph_bitwise_equal']=bool(np.array_equal(reference,out.numpy()))
                        if not row['graph_bitwise_equal']:raise RuntimeError('graph changed outputs')
                        if block==blocks[-1]:
                            def transferred():
                                new,dst=arrays(api,g,q,h,indices,device);wp.launch(api.kernel,count,inputs=new,device=device,block_dim=block);return dst.numpy()
                            _,row['upload_kernel_readback']=timed(transferred,device,3)
                    report['rows'].append(row);save();print(row,flush=True)


if __name__=='__main__':main()
