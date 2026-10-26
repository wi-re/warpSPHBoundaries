"""Additional deterministic random and near-face seed stress checks."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
import fluxbridge_shared_jk_shortmath as impl
import fluxbridge_shared_jk_stability as paired
import fluxbridge_shared_jk_full as fp64
import fluxbridge_shared_jk_layout as geometry
from fluxbridge_shared_jk_full_study import evaluate
from fluxbridge_shared_jk_full_precision_study import metrics
from fluxbridge_shared_jk_consumers import scene
from fluxbridge_single_term_study import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device);vertices=scene(True)
        g=paired.prepare_geometry(vertices,device,'floatpair-inline');g64=geometry.prepare_geometry(vertices,device,'float64')
        report=dict(device=str(device),seed=28491,budget=1e-5,rows=[],
            reference='Existing analytic fp64 full-28 with exact fp32 query/support values; not an independent oracle',
            source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py')})
        rng=np.random.default_rng(report['seed'])
        for name,n in [('random',65536),('near_faces',16384)]:
            ids=rng.integers(0,len(vertices),n,dtype=np.int32)
            if name=='random':
                edges=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1)
                q=vertices[ids,0]+np.einsum('nij,nj->ni',edges[ids],rng.uniform(-.5,1.5,(n,3)))
            else:
                bary=rng.dirichlet(np.ones(4),n);face=np.arange(n)%4
                bary[np.arange(n),face]=0;bary/=bary.sum(axis=1)[:,None]
                q=np.einsum('ni,nij->nj',bary,vertices[ids])
                # Exact face positions and signed offsets; final rounding is
                # included explicitly in the analytic reference inputs below.
                normal=g64.data.numpy()[ids,4+face]
                q+=normal*rng.choice([0,1e-8,-1e-8,1e-5,-1e-5],n)[:,None]
            h=rng.uniform(.3,2,n)
            q=q.astype(np.float32).astype(float);h=h.astype(np.float32).astype(float)
            reference=evaluate(fp64.functions(),g64,q,h,ids,device)
            outputs={}
            for variant in ['control','strict','balanced']:
                actual=evaluate(impl.functions(variant),g,q,h,ids,device)
                outputs[variant]=actual
                m=metrics(actual,reference,list(range(n)),list(range(28)))
                worst=m['worst']['case'];case=dict(vertices=vertices[ids[worst]].tolist(),query=q[worst].tolist(),h=float(h[worst]))
                report['rows'].append(dict(dataset=name,variant=variant,metrics=m,worst_input=case))
                print(name,variant,m['failed_outputs'],m['worst'],flush=True)
            np.savez_compressed(a.output/(name+'.npz'),q=q,h=h,indices=ids,vertices=vertices,reference=reference,**outputs)
            (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        report['passed']=all(r['metrics']['passed'] for r in report['rows'])
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
