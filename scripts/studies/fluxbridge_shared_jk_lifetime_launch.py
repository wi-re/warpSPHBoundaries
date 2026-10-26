"""Matched launch-size, live-state and register-cap measurements."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
import fluxbridge_shared_jk_lifetime as impl
import fluxbridge_shared_jk_fp64_sanity as native64
import fluxbridge_shared_jk_stability as paired
from fluxbridge_shared_jk_full_study import arrays
from fluxbridge_shared_jk_consumers import scene
from fluxbridge_shell_acceptance import timed
from fluxbridge_single_term_study import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',default='control,serial,staged,serial-staged,control@160,control@128,serial@128,staged@128,serial-staged@128,native64-series')
    p.add_argument('--blocks',default='64,128,256');p.add_argument('--mixed',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();variants=a.variants.split(',');blocks=list(map(int,a.blocks.split(',')))
    if any(v not in (*impl.VARIANTS,'native64-series') for v in variants) or any(b not in (32,64,128,256) for b in blocks):p.error('invalid variants/blocks')
    if a.profile and (len(variants)!=1 or len(blocks)!=1):p.error('profile requires one variant/block')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);wp.init();device=wp.get_device('cuda:0');n=1048576
        vertices=scene(a.mixed);g=paired.prepare_geometry(vertices,device,'floatpair-inline');rng=np.random.default_rng(852)
        ids=rng.integers(0,len(vertices),n,dtype=np.int32) if a.mixed else np.zeros(n,int)
        if a.mixed:
            edges=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1)
            q=vertices[ids,0]+np.einsum('nij,nj->ni',edges[ids],rng.uniform(-.5,1.5,(n,3)));h=rng.uniform(.3,2,n)
        else:q=.2+rng.uniform(-.015,.015,(n,3));h=np.ones(n)
        args,out=arrays(impl.functions('control'),g,q,h,ids,device)
        wp.launch(impl.functions('control').kernel,n,inputs=args,device=device,block_dim=256);reference=out.numpy()
        report=dict(mixed=a.mixed,pairs=n,device=device.name,rows=[],scope='Shared buffers; same 28 output contract; block-only changes require bitwise equality',
            source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py')})
        save=lambda:a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for v in variants:
            api=native64.functions(v) if v=='native64-series' else impl.functions(v);variant_reference=None
            for block in blocks:
                action=lambda:wp.launch(api.kernel,n,inputs=args,device=device,block_dim=block)
                action();wp.synchronize_device(device);value=out.numpy()
                if variant_reference is None:variant_reference=value
                error=abs(value.astype(float)-reference)/(1+abs(reference))
                row=dict(variant=v,block_dim=block,kernel_properties=wp.get_cuda_kernel_properties(api.kernel,device),finite=bool(np.isfinite(value).all()),
                    worst_difference_vs_control=float(error.max()),failed_outputs_vs_control=int((error>1e-5).sum()),block_bitwise_equal=bool(np.array_equal(value,variant_reference)))
                assert row['finite'] and row['block_bitwise_equal'] and not row['failed_outputs_vs_control']
                if a.profile:
                    wp.cuda_profiler_start(device);action();wp.synchronize_device(device);wp.cuda_profiler_stop(device)
                else:
                    _,row['warm']=timed(action,device,7,events=True)
                    with wp.ScopedCapture(device=device) as cap:action()
                    _,row['graph']=timed(lambda:wp.capture_launch(cap.graph),device,7,events=True)
                    row['graph_bitwise_equal']=bool(np.array_equal(value,out.numpy()));assert row['graph_bitwise_equal']
                report['rows'].append(row);save();print(row,flush=True)


if __name__=='__main__':main()
