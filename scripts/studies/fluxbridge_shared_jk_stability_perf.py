"""Serialized cost of fp32 conditioning ablations; no acceptance implication."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
import fluxbridge_shared_jk_stability as impl
import fluxbridge_shared_jk_layout as geometry
import fluxbridge_shared_jk_full as fp64
from fluxbridge_shared_jk_full_study import arrays
from fluxbridge_shared_jk_consumers import scene
from fluxbridge_shell_acceptance import timed
from fluxbridge_single_term_study import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',default='raw,products,floatpair,floatpair-math,floatpair-geometry')
    p.add_argument('--counts',default='131072,1048576');p.add_argument('--block',type=int,default=64)
    p.add_argument('--mixed',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();variants=a.variants.split(',');counts=list(map(int,a.counts.split(',')))
    if any(v not in (*impl.VARIANTS,'fp64') for v in variants) or any(n<1 or n>1048576 for n in counts) or a.block not in (32,64,128,256):p.error('invalid variants/counts/block')
    if a.profile and (len(variants)!=1 or len(counts)!=1):p.error('profile requires one variant and count')
    a.output.mkdir(parents=True,exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);wp.init();device=wp.get_device('cuda:0');vertices=scene(a.mixed)
        paths=list((ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py'))+[ROOT/'scripts/studies/fluxbridge_shell_acceptance.py']
        report=dict(device_name=device.name,mixed=a.mixed,block_dim=a.block,rows=[],static_setup={},
            source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
        save=lambda:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        geometries={}
        for variant in variants:
            prepare=lambda:geometry.prepare_geometry(vertices,device,'float64') if variant=='fp64' else impl.prepare_geometry(vertices,device,variant)
            geometries[variant],timing=timed(prepare,device,1)
            g=geometries[variant]
            report['static_setup'][variant]=dict(timing=timing,bytes=geometry.geometry_bytes(g)+(g.data_low.capacity if hasattr(g,'data_low') else 0))
            (a.output/(variant+'-native.cpp')).write_text(fp64.source() if variant=='fp64' else impl.source(variant))
        for count in counts:
            rng=np.random.default_rng(852)
            indices=rng.integers(0,len(vertices),count,dtype=np.int32) if a.mixed else np.zeros(count,int)
            if a.mixed:
                edges=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1)
                q=vertices[indices,0]+np.einsum('nij,nj->ni',edges[indices],rng.uniform(-.5,1.5,(count,3)));h=rng.uniform(.3,2,count)
            else:q=.2+rng.uniform(-.015,.015,(count,3));h=np.ones(count)
            for variant in variants:
                api=fp64.functions() if variant=='fp64' else impl.functions(variant)
                args,out=arrays(api,geometries[variant],q,h,indices,device)
                action=lambda:wp.launch(api.kernel,count,inputs=args,device=device,block_dim=a.block)
                action();wp.synchronize_device(device)
                row=dict(pairs=count,variant=variant,kernel_properties=wp.get_cuda_kernel_properties(api.kernel,device),input_output_bytes=sum(x.capacity for x in args if isinstance(x,wp.array)))
                if a.profile:
                    wp.cuda_profiler_start(device);action();wp.synchronize_device(device);wp.cuda_profiler_stop(device)
                else:
                    _,row['warm']=timed(action,device,7,events=True)
                    with wp.ScopedCapture(device=device) as cap:action()
                    _,row['graph']=timed(lambda:wp.capture_launch(cap.graph),device,7,events=True)
                    row['finite']=bool(np.isfinite(out.numpy()).all())
                    if not row['finite']:raise RuntimeError('nonfinite output')
                report['rows'].append(row);save();print(row,flush=True)


if __name__=='__main__':main()
