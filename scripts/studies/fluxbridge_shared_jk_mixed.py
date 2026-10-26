"""Measure the fused fp64 prototype with heterogeneous per-pair inputs."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from fluxbridge_shared_jk import functions, prepare_geometry
from fluxbridge_shared_jk_study import inputs
from fluxbridge_shell_acceptance import timed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device('cuda:0');api=functions('float64')
        rng=np.random.default_rng(729)
        edges=[]
        for _ in range(256):
            rotation=np.linalg.qr(rng.normal(size=(3,3)))[0]
            stretch=np.diag(rng.uniform(.5,2,3))+np.triu(rng.uniform(-.2,.2,(3,3)),1)
            edges.append(rotation@stretch)
        edges=np.asarray(edges);origins=rng.uniform(-10,10,(256,3))
        vertices=np.concatenate([origins[:,None,:],origins[:,None,:]+edges.transpose(0,2,1)],axis=1)
        geometry,setup=timed(lambda:prepare_geometry(vertices,device),device,1)
        static_bytes=sum(getattr(geometry,k).capacity for k in ('vertices','normals','tangents','outward','gradients','edges'))
        report=dict(scope='fp64; 256 rotated/stretched/sheared tets; random position, support, tet index and four nodal coefficients for every pair',
                    setup=setup,static_geometry_bytes=static_bytes,rows=[],
                    limitations=['No search or scatter; static geometry reused',
                                 'Different branch mix from the canonical benchmark; speeds are workload-specific',
                                 'GPU timings serialized; clocks/caches and desktop activity uncontrolled'],
                    source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (Path(__file__),Path(__file__).with_name('fluxbridge_shared_jk.py'),Path(__file__).with_name('fluxbridge_shared_jk_study.py'))})
        for count in (512,8192,131072,1048576):
            tet=rng.integers(0,256,count,dtype=np.int32)
            local=rng.uniform(-.5,1.5,(count,3))
            q=origins[tet]+np.einsum('nij,nj->ni',edges[tet],local)
            h=rng.uniform(.3,2,count);field=rng.normal(size=(count,4))
            arrays=inputs(api,q,h,field,tet,device);dst=wp.empty(count,dtype=api.real,device=device)
            action=lambda:wp.launch(api.kernel,count,inputs=arrays+[geometry,dst],device=device)
            action();wp.synchronize_device(device)
            _,warm=timed(action,device,7,events=True)
            def transferred():
                new=inputs(api,q,h,field,tet,device);out=wp.empty(count,dtype=api.real,device=device)
                wp.launch(api.kernel,count,inputs=new+[geometry,out],device=device)
                return out.numpy()
            values,complete=timed(transferred,device,3)
            assert np.isfinite(values).all()
            report['rows'].append(dict(pairs=count,warm=warm,upload_kernel_readback=complete,
                                       input_output_bytes=sum(x.capacity for x in arrays)+dst.capacity,
                                       global_pair_scratch_bytes=0))
            a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(count,warm,flush=True)


if __name__=='__main__':
    main()
