"""Serialized fp32 CUDA rates for individual volume-term bundles.

One interaction means one particle/simplex pair, independent of output count.
Warm launches recompute integrals and reduce pairs into particle outputs. Complete
calls include host search, validation, allocation/upload, convergence checks
(where implemented), native evaluation, reduction and readback; static geometry/
plan construction and compilation are separate. This is not solver throughput.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry, Evaluator, Request, baseline
from warpSPHBoundaries.fluxbridge.contract import monomials
from fluxbridge_shell_acceptance import timed

ROOT=Path(__file__).resolve().parents[2]


def masks(d):
    z=(0,)*d;eye=np.eye(d,dtype=int)
    return dict(
        occupancy=[Request(z)],
        gradient=[Request(z,(j,)) for j in range(d)],
        first_second=[Request(a) for a in monomials(d,2) if sum(a)],
        first_gradient=[Request(tuple(a),(j,)) for a in eye for j in range(d)],
        hessian=[Request(z,(i,j)) for i in range(d) for j in range(i,d)],
        laplacian_inputs=[Request(z,(j,j)) for j in range(d)],
        baseline=list(baseline(d)))


def run(output,quick=False,large_3d=False):
    wp.init();device=wp.get_device('cuda:0');assert device.is_cuda
    paths=list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))+[Path(__file__),ROOT/'scripts/studies/fluxbridge_shell_acceptance.py']
    report=dict(schema=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),device=str(device),device_name=device.name,
        warp=wp.__version__,precision='float32',rows=[],references=[],source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        workload='16 separated 2*unit simplices; varied queries around local coordinate .2; h=1; one selected element per particle',
        limitations=['not full solver throughput','GPU action includes pair-to-particle reduction','wall timers include Python/driver submission and synchronization',
                     'CUDA events measure the submitted multi-kernel action, including possible host submission gaps',
                     'Laplacian row computes diagonal Hessian entries; their final trace is not included',
                     'complete cost reuses static geometry/plan; no static BVH/material rebuild',
                     'term masks are separate evaluations, not additive costs of a fused application',
                     'baseline-subset checks and sampled fp64 comparisons are finite checks, not independent high-precision fixtures'])
    def save():
        output.mkdir(parents=True,exist_ok=True)
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    for d in ((3,) if large_3d else (2,3)):
        elements=[np.vstack([np.zeros(d),2*np.eye(d)])+np.array([4*i]+[0]*(d-1)) for i in range(16)]
        geo=Geometry(elements,d)
        for kernel in ('w2','w6'):
            counts=[512] if large_3d else [128] if d==3 or quick else [128,2048]
            orders=[8] if large_3d else [16] if d==2 or quick else [16,8]
            for count in counts:
                rng=np.random.default_rng(852)
                q=.2+rng.uniform(-.015,.015,(count,d));q[:,0]+=4*(np.arange(count)%16)
                reference_ev=Evaluator(geo,baseline(d),kernel=kernel,precision='float64',method='conditioned_boundary' if d==2 else 'tet_quadrature',
                    device=device,preparation='host',nodes=16)
                reference=reference_ev.evaluate(q[:8],1)
                report['references'].append(dict(dimension=d,kernel=kernel,queries=8,method=reference_ev.method,precision='float64'))
                del reference_ev;gc.collect()
                for nodes in orders:
                    expected=None
                    for name,requests in [('baseline',masks(d)['baseline'])]+[(k,v) for k,v in masks(d).items() if k!='baseline']:
                        options=dict(kernel=kernel,precision='float32',method='conditioned_boundary' if d==2 else 'tet_quadrature',
                            device=device,preparation='host' if d==2 else 'device',search='device',nodes=nodes)
                        ev,static=timed(lambda:Evaluator(geo,requests,**options),device,1)
                        batch,prepare=timed(lambda:ev.prepare(q,1),device,1)
                        assert batch.count==count,(batch.count,count)
                        batch.launch();wp.synchronize_device(device)
                        _,warm=timed(batch.launch,device,5,events=True)
                        values=batch.numpy()
                        lookup={r:i for i,r in enumerate(baseline(d))};indices=[lookup[r] for r in requests]
                        error=float(np.max(abs(values[:8]-reference[:,indices])/(1+abs(reference[:,indices]))))
                        if expected is None:expected=values.copy()
                        subset_error=float(np.max(abs(values-expected[:,indices])/(1+abs(expected[:,indices]))))
                        if max(error,subset_error)>1e-5:raise RuntimeError((d,kernel,name,error,subset_error))
                        _,complete=timed(lambda:ev.evaluate(q,1),device,3)
                        row=dict(dimension=d,kernel=kernel,term=name,queries=count,interactions=batch.count,outputs=len(requests),
                            method=ev.method,preparation=ev.preparation,search=ev.search,requested_nodes=nodes,
                            quadrature=getattr(batch,'quadrature',None),static_setup=static,checked_preparation=prepare,
                            warm_action=warm,complete=complete,batch_owned_bytes=batch.bytes,
                            sampled_mixed_error_vs_fp64=error,mixed_error_vs_fused_baseline=subset_error,
                            warm_interactions_per_second=count*1000/warm['median_ms'],
                            event_interactions_per_second=count*1000/warm['cuda_event_median_ms'],
                            complete_interactions_per_second=count*1000/complete['median_ms'])
                        report['rows'].append(row);save()
                        print(d,kernel,count,nodes,name,'warm',round(warm['median_ms'],3),'ms','complete',round(complete['median_ms'],3),'ms',flush=True)
                        del batch,ev;gc.collect()
                    del expected
    save();return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--large-3d',action='store_true',help='Measure 512-pair tetrahedral batches with explicit order 8')
    args=parser.parse_args()
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);run(args.output,args.quick,args.large_3d)
