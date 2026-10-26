"""Matched shared/repeated P1 field contractions, including mixed geometry.

Both strategies use the same cached geometry, nodal data, query and output
storage. The repeated strategy launches the validated scalar device function
once per component; the shared strategy evaluates four moments once, then
streams contractions. CUDA graph timing removes intervening Python launch gaps.
These are P1 field components, not the 28 distinct moment/derivative operators.
"""
import argparse
import fcntl
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

import fluxbridge_shared_jk_layout as impl
from fluxbridge_shared_jk_layout_study import launch_inputs,ROOT
from fluxbridge_shell_acceptance import timed

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'max_unroll':1})


@lru_cache(None)
def repeated_kernel(layout='packed',reload=True):
    api=impl.functions(layout=layout,reload=reload)
    vec,real,field_vec,Geometry=api.vec,api.real,api.field_vec,api.Geometry
    interaction=api.interaction

    @wp.kernel
    def separate_components(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
                            tet:wp.array(dtype=int),fields:wp.array2d(dtype=field_vec),
                            geometry:Geometry,output:wp.array2d(dtype=real),component:int):
        i=wp.tid()
        output[component,i]=interaction(particle[i],support[i],tet[i],fields[tet[i],component],geometry)

    return separate_components


def scene(mixed):
    if not mixed:
        return np.array([np.vstack([np.zeros(3),2*np.eye(3)])])
    rng=np.random.default_rng(729);edges=[]
    for _ in range(256):
        rotation=np.linalg.qr(rng.normal(size=(3,3)))[0]
        stretch=np.diag(rng.uniform(.5,2,3))+np.triu(rng.uniform(-.2,.2,(3,3)),1)
        edges.append(rotation@stretch)
    edges=np.asarray(edges);origin=rng.uniform(-10,10,(256,3))
    return np.concatenate([origin[:,None,:],origin[:,None,:]+edges.transpose(0,2,1)],axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',choices=('separate','packed'),default='packed')
    p.add_argument('--reload',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--counts',default='512,131072,1048576')
    p.add_argument('--components',default='3,28')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    counts=list(map(int,a.counts.split(',')));components=list(map(int,a.components.split(',')))
    if any(n<1 or n>1048576 for n in counts) or any(k<1 or k>28 for k in components):p.error('invalid counts')
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device('cuda:0');api=impl.functions(layout=a.layout,reload=a.reload)
        scalar=repeated_kernel(a.layout,a.reload)
        paths=list((ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py'))
        report=dict(layout=a.layout,reload=a.reload,precision='float64',rows=[],static=[],
                    source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
                    limitations=['No solver, gradient/Hessian or full 28-operator contract',
                        'Same [tet,component,4] nodal coefficients for both strategies; output [component,pair]',
                        'Warm and graph times exclude JIT and separately timed static geometry preparation',
                        'Transfer timings include host validation/allocation/upload/launch/readback',
                        'GPU serialized; clocks/caches/desktop activity uncontrolled'])
        save=lambda:a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for mixed in (False,True):
            vertices=scene(mixed)
            geometry,setup=timed(lambda:impl.prepare_geometry(vertices,device,layout=a.layout),device,1)
            report['static'].append(dict(mixed=mixed,tets=len(vertices),setup=setup,bytes=impl.geometry_bytes(geometry)))
            for count in counts:
                rng=np.random.default_rng(852)
                indices=rng.integers(0,len(vertices),count,dtype=np.int32) if mixed else np.zeros(count,np.int32)
                if mixed:
                    local=rng.uniform(-.5,1.5,(count,3));edges=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1)
                    q=vertices[indices,0]+np.einsum('nij,nj->ni',edges[indices],local);h=rng.uniform(.3,2,count)
                else:
                    q=.2+rng.uniform(-.015,.015,(count,3));h=np.ones(count)
                for nf in components:
                    field=np.random.default_rng(317).normal(size=(len(vertices),nf,4))
                    kernel,args,out=launch_inputs(api,q,h,indices,field,geometry,device,'fields')
                    reference=None
                    for strategy in ('shared','repeated'):
                        def action():
                            if strategy=='shared':wp.launch(kernel,count,inputs=args,device=device)
                            else:
                                for component in range(nf):wp.launch(scalar,count,inputs=args+[component],device=device)
                        action();wp.synchronize_device(device);values=out.numpy()
                        if reference is None:reference=values.copy()
                        err=float(np.max(abs(values-reference)/(1+abs(reference))))
                        if not np.isfinite(values).all() or err>1e-10:raise RuntimeError('shared/repeated contraction mismatch')
                        _,warm=timed(action,device,7,events=True)
                        with wp.ScopedCapture(device=device) as capture:action()
                        wp.capture_launch(capture.graph);wp.synchronize_device(device)
                        bitwise=bool(np.array_equal(values,out.numpy()))
                        if not bitwise:raise RuntimeError('graph output changed')
                        _,graph=timed(lambda:wp.capture_launch(capture.graph),device,7,events=True)
                        def transferred():
                            k,new,dst=launch_inputs(api,q,h,indices,field,geometry,device,'fields')
                            if strategy=='shared':wp.launch(k,count,inputs=new,device=device)
                            else:
                                for component in range(nf):wp.launch(scalar,count,inputs=new+[component],device=device)
                            return dst.numpy()
                        _,transfer=timed(transferred,device,3)
                        row=dict(mixed=mixed,pairs=count,components=nf,strategy=strategy,
                            mixed_difference=err,graph_bitwise_equal=bitwise,warm=warm,graph=graph,
                            upload_kernel_readback=transfer,
                            kernel_properties=wp.get_cuda_kernel_properties(kernel if strategy=='shared' else scalar,device),
                            input_output_bytes=sum(x.capacity for x in args if isinstance(x,wp.array)),
                            global_pair_scratch_bytes=0)
                        report['rows'].append(row);save()
                        print(mixed,count,nf,strategy,graph['cuda_event_median_ms'],row['kernel_properties'],flush=True)


if __name__=='__main__':main()
