"""Independent boundary-input audit of the fused analytic P1/W6 prototype.

Tensor Gauss/Duffy integration is used only as an independent CPU reference.
It is not part of the prototype kernel or its measured runtime. The reference
integrand uses the factored W6 shape, not the analytic edge/face formulas.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from fluxbridge_shared_jk import functions, prepare_geometry
from fluxbridge_shared_jk_study import evaluate
from fluxbridge_single_term_analytic import provider


def shape_cases():
    rng = np.random.default_rng(51)
    cases = []
    for i in range(18):
        rotation = np.linalg.qr(rng.normal(size=(3,3)))[0]
        axes = np.diag(rng.uniform(.5,2,3))
        if i >= 12:
            axes += np.triu(rng.uniform(-.35,.35,(3,3)),1)
        edge = rotation @ axes
        origin = rng.uniform(-1,1,3)
        vertices = np.vstack([origin, origin+edge.T])
        field = rng.normal(size=4)
        samples = dict(interior=vertices.mean(axis=0), vertex0=vertices[0],
                       vertex1=vertices[1], edge=.4*vertices[0]+.6*vertices[1],
                       face=vertices[1:].mean(axis=0))
        for label, query in samples.items():
            cases.append(dict(label=[i,label], vertices=vertices, query=query,
                              h=float(rng.uniform(.5,2)), field=field))
    return cases


def independent_integral(v, q, h, f, order):
    t, w = np.polynomial.legendre.leggauss(order)
    t, w = (t+1)/2, w/2
    u=t[:,None,None]; b=1-u; vv=t[None,:,None]
    c=b*vv; d=b*(1-vv); z=d*t[None,None,:]
    y=((v[0]-q)/h + u[...,None]*(v[1]-v[0])/h
       + c[...,None]*(v[2]-v[0])/h + z[...,None]*(v[3]-v[0])/h)
    r=np.sqrt(np.sum(y*y,axis=-1)); s=np.maximum(1-r,0)
    field=f[0]+u*(f[1]-f[0])+c*(f[2]-f[0])+z*(f[3]-f[0])
    weights=w[:,None,None]*w[None,:,None]*w[None,None,:]*b*b*(1-vv)
    return float(1365/(64*np.pi)*abs(np.linalg.det((v[1:]-v[0]).T))/h**3
                 * np.sum(weights*s**8*(1+r*(8+r*(25+32*r)))*field))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)
    cs=shape_cases()
    # Independent convergence check at every query, including exact boundaries.
    refs=[[independent_integral(c['vertices'],c['query'],c['h'],c['field'],n)
           for n in (48,72,96)] for c in cs]
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):
            fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device);api=functions('float64')
        geometry=prepare_geometry([c['vertices'] for c in cs],device)
        values=evaluate(api,geometry,[c['query'] for c in cs],[c['h'] for c in cs],
                        [c['field'] for c in cs],np.arange(len(cs)),device)
        native=[float(provider(c['vertices'],c['field'],device,True,'float64')
                      .evaluate([c['query']],c['h'])[0]) for c in cs]
    rows=[]
    for c,value,old,ref in zip(cs,values,native,refs):
        rows.append(dict(**{k:(v.tolist() if isinstance(v,np.ndarray) else v)
                            for k,v in c.items()}, value=float(value),
                         native_value=old, quadrature_orders=[48,72,96],
                         independent_values=ref,
                         reference_convergence=abs(ref[-1]-ref[-2])/(1+abs(ref[-1])),
                         mixed_error=abs(float(value)-ref[-1])/(1+abs(ref[-1])),
                         native_mixed_error=abs(old-ref[-1])/(1+abs(ref[-1]))))
    budget=1e-10
    report=dict(device=str(device),budget=budget,rows=rows,
                scope='90 rotated/stretched/sheared tet queries: interior, two vertices, edge, face',
                passed=all(np.isfinite(r['mixed_error']) and r['mixed_error']<=budget
                           and r['reference_convergence']<=budget for r in rows),
                source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__),Path(__file__).with_name('fluxbridge_shared_jk.py'))})
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('passed',report['passed'],'worst error',max(r['mixed_error'] for r in rows),
          'worst reference convergence',max(r['reference_convergence'] for r in rows),
          'native failures',sum(r['native_mixed_error']>budget for r in rows),flush=True)
    if not report['passed']:
        raise RuntimeError('independent shape-input gate failed')


if __name__=='__main__':
    main()
