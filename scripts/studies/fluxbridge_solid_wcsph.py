"""Serialized actual-input WCSPH accuracy, warm pair rates and CSR bridge cost."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import SolidEvaluator
from warpSPHBoundaries.fluxbridge.solid import solid_functions,prepare_solid_geometry
from fluxbridge_solid_probe import scene,inputs
from fluxbridge_solid_study import evaluate,metrics,ROOT
from fluxbridge_shell_acceptance import timed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=1048576)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(pair_count=a.count,rows=[],pipeline=[],source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__)]+list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*solid*.py'))})
    save=lambda:a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);wp.init();device=wp.get_device('cuda:0');assert device.is_cuda
        report['device']=device.name
        for d in (2,3):
            for mixed in (False,True):
                vertices=scene(d,mixed).astype(np.float32).astype(float);q,h,ids=inputs(d,vertices,a.count,mixed)
                for k in ('w2','w4'):
                    for bundle in ('lean','wcsph','wcsph_extended'):
                        # Small distinct actual-input comparison; whole-million output checked for finiteness.
                        reference=evaluate(solid_functions(d,k,'float64',bundle),prepare_solid_geometry(vertices,device,'float64'),q[:8192],h[:8192],ids[:8192],device)
                        for precision in ('float64','float32'):
                            api=solid_functions(d,k,precision,bundle);g=prepare_solid_geometry(vertices,device,precision)
                            args=[wp.array(np.pad(q,((0,0),(0,3-d))),dtype=api.vec,device=device),wp.array(h,dtype=api.real,device=device),wp.array(ids,dtype=int,device=device),g]
                            out=wp.empty((api.channels,a.count),dtype=api.real,device=device)
                            action=lambda:wp.launch(api.kernel,a.count,inputs=args+[out],device=device,block_dim=256)
                            action();value=out.numpy();assert np.isfinite(value).all()
                            m=metrics(value[:,:8192].T.astype(float),reference,1e-10 if precision=='float64' else 1e-5);assert m['passed'],m
                            _,warm=timed(action,device,7,events=True)
                            row=dict(dimension=d,mixed=mixed,kernel=k,bundle=bundle,precision=precision,channels=api.channels,accuracy=m,warm=warm,
                                million_pairs_per_s=a.count/warm['cuda_event_median_ms']/1000,properties=wp.get_cuda_kernel_properties(api.kernel,device),
                                io_bytes=sum(x.capacity for x in args[:3])+out.capacity,geometry_bytes=g.data.capacity+g.edges.capacity+(g.data_low.capacity if precision=='float32' else 0))
                            report['rows'].append(row);save();print(d,mixed,k,bundle,precision,row['million_pairs_per_s'],flush=True)
                            del args,out,value,g
        # A dense static simplex lattice, with particles fixed in its interior.
        from scipy.spatial import Delaunay
        from itertools import product
        for d in (2,3):
            axis=np.linspace(0,1,9 if d==2 else 5);points=np.array(list(product(axis,repeat=d)));tri=Delaunay(points)
            vertices=points[tri.simplices]
            # Qhull may emit zero-volume tets on the regular lattice.
            det=np.linalg.det((vertices[:,1:]-vertices[:,0,None]).transpose(0,2,1));vertices=vertices[abs(det)>1e-14]
            rng=np.random.default_rng(912)
            for n in (4096,16384,65536):
                q=rng.uniform(.05,.95,(n,d));h=np.full(n,.13 if d==2 else .24)
                for search in ('host','device'):
                    before=perf_counter();e=SolidEvaluator(vertices,device=device,precision='float32',bundle='wcsph',search=search);wp.synchronize_device(device)
                    static_ms=(perf_counter()-before)*1000
                    before=perf_counter();b=e.prepare(q,h);wp.synchronize_device(device);prepare_ms=(perf_counter()-before)*1000
                    b.evaluate();b.aggregate.numpy() # compile/warm outside timing
                    _,warm=timed(b.evaluate,device,7,events=True)
                    before=perf_counter();result=b.aggregate.numpy();readback_ms=(perf_counter()-before)*1000;assert np.isfinite(result).all()
                    row=dict(dimension=d,search=search,elements=len(vertices),queries=n,pairs=b.pair_count,static_prepare_upload_ms=static_ms,
                        query_prepare_search_upload_ms=prepare_ms,warm_evaluate_assemble=warm,readback_ms=readback_ms,
                        batch_array_bytes=b.array_bytes,geometry_array_bytes=e.prepared.data.capacity+e.prepared.data_low.capacity+e.prepared.edges.capacity,
                        search_array_bytes=e.search.array_bytes if e.search else 0,search_metrics=e.search.last_metrics if e.search else None)
                    report['pipeline'].append(row);save();print('PIPELINE',d,n,search,b.pair_count,prepare_ms,warm['cuda_event_median_ms'],flush=True)
        report['passed']=True;save()


if __name__=='__main__':main()
