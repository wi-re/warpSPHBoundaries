"""Independent long-double signed-cone volume reference for the 28 channels.

Decompose the volume into oriented cones from the query to its four faces.
Integrate the radial polynomial exactly, then use a two-dimensional Gauss rule
on each face. This is validation-only, independent of device J/K recurrences.
"""
from fractions import Fraction as F
from math import comb
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from warpSPHBoundaries.fluxbridge import baseline
from fluxbridge_single_term_study import cases,ROOT

LD=np.longdouble
REQUESTS=baseline(3)
SHAPES=ROOT/'docs/work/fluxbridge-shared-jk-shapes-20261026.json'


def profiles():
    phi={}
    for i in range(9):
        for j,c in enumerate((1,8,25,32)):
            phi[i+j]=phi.get(i+j,0)+(-1)**i*comb(8,i)*c
    psi={n-2:n*c for n,c in phi.items() if n>=2 and c}
    chi={n-4:n*(n-2)*c for n,c in phi.items() if n>=4 and c}
    return phi,psi,chi


def integral(c,order):
    vertices=np.asarray(c['vertices'],dtype=LD);query=np.asarray(c['query'],dtype=LD);h=LD(c['h'])
    v=(vertices-query)/h
    nodes,weights=np.polynomial.legendre.leggauss(order)
    u=((nodes+1)/2).astype(LD)[:,None];w=(weights/2).astype(LD)
    vv=((nodes+1)/2).astype(LD)[None,:]
    angular=w[:,None]*w[None,:]*(1-u)
    phi,psi,chi=profiles();result=np.zeros(28,dtype=LD)
    C=LD(1365)/(LD(64)*np.arccos(LD(-1)))
    for opposite in range(4):
        ids=[i for i in range(4) if i!=opposite];a,b,d=v[ids]
        if np.dot(np.cross(b-a,d-a),v[opposite]-a)>0:b,d=d,b
        det=np.dot(a,np.cross(b,d))
        if det==0:continue
        y=a+u[...,None]*(b-a)+((1-u)*vv)[...,None]*(d-a)
        r=np.sqrt(np.sum(y*y,axis=-1))
        cap=np.minimum(LD(1),np.divide(LD(1),r,out=np.ones_like(r),where=r>0));top=r*cap
        def radial(profile,power):
            coefficients=np.zeros(max(profile)+1,dtype=LD)
            for n,coefficient in profile.items():
                f=F(coefficient,n+power);coefficients[n]=LD(f.numerator)/LD(f.denominator)
            return cap**power*np.polynomial.polynomial.polyval(top,coefficients)
        radial_phi=[radial(phi,k+3) for k in range(3)]
        radial_grad=[radial(psi,k+4) for k in range(2)]
        diagonal=radial(psi,3);dyadic=radial(chi,5)
        for k,req in enumerate(REQUESTS):
            degree=sum(req.alpha)
            if not req.derivative:value=radial_phi[degree]
            elif len(req.derivative)==1:value=-radial_grad[degree]*y[...,req.derivative[0]]
            else:
                i,j=req.derivative;value=dyadic*y[...,i]*y[...,j]+int(i==j)*diagonal
            for axis,power in enumerate(req.alpha):
                if power:value=value*y[...,axis]**power
            result[k]+=C*h**req.scale*det*np.sum(angular*value,dtype=LD)
    return np.asarray(result,dtype=float)


def generate(path):
    report=dict(method='Signed face cones; exact radial polynomials in np.longdouble; 2D Gauss face rule',
                target=2e-12,rows=[],fixture_checks=[],shape_sha256=hashlib.sha256(SHAPES.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),longdouble_mantissa_bits=np.finfo(LD).nmant)
    # The oracle also has to agree with the independently tabulated fixtures.
    for c in cases(3):
        table={(tuple(r['alpha']),tuple(r['derivative'])):float(x) for r,x in zip(c['requests'],c['values'])}
        ref=np.array([table[(r.alpha,r.derivative)] for r in REQUESTS]);value=integral(c,96)
        err=float(np.max(abs(value-ref)/(1+abs(ref))))
        report['fixture_checks'].append(dict(label=c['label'],worst=err))
    if max(c['worst'] for c in report['fixture_checks'])>1e-10:raise RuntimeError('cone oracle fixture gate failed')
    for c in json.loads(SHAPES.read_text())['rows']:
        orders=[32,48];old=integral(c,32);value=integral(c,48)
        err=float(np.max(abs(value-old)/(1+abs(value))))
        for order in (64,96,128,160,192,256):
            if err<=report['target']:break
            old=value;value=integral(c,order);orders.append(order)
            err=float(np.max(abs(value-old)/(1+abs(value))))
        row={k:c[k] for k in ('label','vertices','query','h')}
        row.update(values=value.tolist(),orders=orders,convergence=err);report['rows'].append(row)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(c['label'],orders,err,flush=True)
    report['worst_convergence']=max(c['convergence'] for c in report['rows'])
    report['passed']=report['worst_convergence']<=report['target']
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not report['passed']:raise RuntimeError('cone reference convergence failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    generate(p.parse_args().output)
