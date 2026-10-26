"""One warmed solid-only interaction kernel for external CUDA profiling."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge.solid import solid_functions,prepare_solid_geometry
from fluxbridge_solid_probe import scene,inputs,ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dimension',type=int,choices=(2,3),required=True);p.add_argument('--precision',choices=('float32','float64'),required=True);p.add_argument('--conditioning',choices=('raw','protected'),default='raw');p.add_argument('--kernel',choices=('w2','w4'),default='w4');p.add_argument('--bundle',choices=('density','pressure','lean'),default='lean');p.add_argument('--mixed',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);wp.init();device=wp.get_device('cuda:0');n=1048576
        vertices=scene(a.dimension,a.mixed).astype(np.float32).astype(float);q,h,ids=inputs(a.dimension,vertices,n,a.mixed)
        api=solid_functions(a.dimension,a.kernel,a.precision,a.bundle,conditioning=a.conditioning);g=prepare_solid_geometry(vertices,device,a.precision,a.conditioning)
        args=[wp.array(np.pad(q,((0,0),(0,3-a.dimension))),dtype=api.vec,device=device),wp.array(h,dtype=api.real,device=device),wp.array(ids,dtype=int,device=device),g,wp.empty((api.channels,n),dtype=api.real,device=device)]
        action=lambda:wp.launch(api.kernel,n,inputs=args,device=device,block_dim=256)
        action();wp.synchronize_device(device)
        wp.cuda_profiler_start(device);action();wp.synchronize_device(device);wp.cuda_profiler_stop(device)
        assert np.isfinite(args[-1].numpy()).all()
        paths=[Path(__file__),ROOT/'scripts/studies/fluxbridge_solid_probe.py']+list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*solid*.py'))
        report=dict(dimension=a.dimension,profile=a.kernel,precision=a.precision,conditioning=a.conditioning,bundle=a.bundle,pairs=n,block_dim=256,mixed=a.mixed,kernel_properties=wp.get_cuda_kernel_properties(api.kernel,device),source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
        a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
