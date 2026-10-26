"""Matched launch-size experiment for the complete analytic W6 contract.

Keep the same inputs, output allocation and generated device body for every
launch. No register cap, changed arithmetic, or reduced output contract.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
import fluxbridge_shared_jk_full as impl
import fluxbridge_shared_jk_layout as geometry
from fluxbridge_shared_jk_full_study import arrays
from fluxbridge_shared_jk_consumers import scene
from fluxbridge_shell_acceptance import timed
from fluxbridge_single_term_study import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts',default='8192,131072,1048576')
    p.add_argument('--blocks',default='32,64,128,256')
    p.add_argument('--mixed',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();counts=list(map(int,a.counts.split(',')));blocks=list(map(int,a.blocks.split(',')))
    if any(n<1 or n>1048576 for n in counts) or any(b not in (32,64,128,256) for b in blocks):p.error('invalid counts or blocks')
    if a.profile and (len(counts)!=1 or len(blocks)!=1):p.error('profile needs one count and block size')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device('cuda:0');api=impl.functions();vertices=scene(a.mixed)
        g=geometry.prepare_geometry(vertices,device)
        paths=list((ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py'))
        report=dict(mixed=a.mixed,device_name=device.name,rows=[],
            source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
        save=lambda:a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for count in counts:
            rng=np.random.default_rng(852)
            indices=rng.integers(0,len(vertices),count,dtype=np.int32) if a.mixed else np.zeros(count,int)
            if a.mixed:
                edges=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1)
                q=vertices[indices,0]+np.einsum('nij,nj->ni',edges[indices],rng.uniform(-.5,1.5,(count,3)))
                h=rng.uniform(.3,2,count)
            else:q=.2+rng.uniform(-.015,.015,(count,3));h=np.ones(count)
            args,out=arrays(api,g,q,h,indices,device)
            wp.launch(api.kernel,count,inputs=args,device=device,block_dim=256)
            reference=out.numpy()
            for block in blocks:
                action=lambda:wp.launch(api.kernel,count,inputs=args,device=device,block_dim=block)
                action();wp.synchronize_device(device)
                row=dict(pairs=count,block_dim=block,kernel_properties=wp.get_cuda_kernel_properties(api.kernel,device),
                    bitwise_equal=bool(np.array_equal(reference,out.numpy())))
                if not row['bitwise_equal']:raise RuntimeError('launch size changed outputs')
                if a.profile:
                    wp.cuda_profiler_start(device);action();wp.synchronize_device(device);wp.cuda_profiler_stop(device)
                else:
                    _,row['warm']=timed(action,device,7,events=True)
                    with wp.ScopedCapture(device=device) as cap:action()
                    _,row['graph']=timed(lambda:wp.capture_launch(cap.graph),device,7,events=True)
                report['rows'].append(row);save();print(row,flush=True)


if __name__=='__main__':main()
