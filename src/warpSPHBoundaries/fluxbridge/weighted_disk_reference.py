"""Independent high-precision regular-flux circle oracle for weighted disks.

All circle arcs, including those outside support, participate. Homogeneous
Cartesian powers are rotated into axial/transverse powers before independent
mpmath boundary integration. Native evaluation uses interior volume nodes.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import comb
import mpmath as mp
from ..edge.kernels import KERNELS


def _mp(value):
    value=F(value)
    return mp.mpf(value.numerator)/value.denominator


@lru_cache(None)
def primitives(kernel,profile,degree):
    rows=[];acc=F(0)
    for lo,hi,coeff in KERNELS[kernel].pieces:
        poly={}
        for power,c in enumerate(coeff):
            if profile==1:c*=power
            if profile==2:c*=power*(power-2)
            if not c:continue
            exponent=power-2*profile+degree+2
            if exponent==0:raise ValueError('unexpected logarithmic primitive')
            poly[exponent]=c/exponent
        offset=acc-sum(c*lo**p for p,c in poly.items())
        acc=offset+sum(c*hi**p for p,c in poly.items())
        rows.append((lo,hi,poly,offset))
    return rows,acc


def rotated(alpha,normal):
    nx,ny=map(F,normal);out={}
    for i in range(alpha[0]+1):
        for j in range(alpha[1]+1):
            axial=i+j;transverse=sum(alpha)-axial
            coefficient=(comb(alpha[0],i)*comb(alpha[1],j)*nx**i*(-ny)**(alpha[0]-i)
                         *ny**j*nx**(alpha[1]-j))
            out[(axial,transverse)]=out.get((axial,transverse),F(0))+coefficient
    return out


def evaluate(distance,radius,requests,kernel='w2',normal=(F(3,5),F(4,5))):
    """Unit-support moments; derivative axes act only on the kernel."""
    d,R=_mp(distance),_mp(radius);centre=R+d
    if R<=0 or centre<0:raise ValueError('invalid disk radius/distance')
    if sum(F(x)**2 for x in normal)!=1:raise ValueError('exact unit rational normal required')
    if d>=1:return [mp.mpf(0)]*len(requests)
    cuts=[mp.mpf(0),mp.pi]
    if centre:
        for knot in sorted({hi for lo,hi,c in KERNELS[kernel].pieces}):
            z=(_mp(knot)**2-d*d)/(4*centre*R)
            if 0<z<1:cuts.append(2*mp.asin(mp.sqrt(z)))
        # Resolve the nearest-point boundary layer without dropping any arc.
        scale=abs(d)/mp.sqrt(centre*R)
        if scale:
            angle=min(mp.pi,scale)
            while angle<mp.pi:
                cuts.append(angle);angle*=4
    cuts=sorted(set(cuts))

    @lru_cache(None)
    def axial_integral(axial,transverse,profile):
        if transverse%2:return mp.mpf(0)
        degree=axial+transverse
        rows,total=primitives(kernel,profile,degree)
        packed=[(_mp(lo),_mp(hi),{p:_mp(c) for p,c in poly.items()},_mp(off)) for lo,hi,poly,off in rows]
        def integrand(phi):
            sine=mp.sin(phi);half=mp.sin(phi/2)
            z=-d-2*R*half*half;t=R*sine
            r=mp.sqrt(d*d+4*centre*R*half*half)
            if not r:return mp.mpf(0)
            primitive=_mp(total)
            for lo,hi,poly,off in packed:
                if r<=hi:
                    primitive=off+sum(c*r**p for p,c in poly.items());break
            return R*(R-centre*mp.cos(phi))*z**axial*t**transverse*primitive/r**(degree+2)/mp.pi
        return 2*sum(mp.quad(integrand,[a,b]) for a,b in zip(cuts,cuts[1:]))

    results=[]
    for request in requests:
        alpha=tuple(request.alpha);ds=request.derivative
        terms=[]
        if not ds:terms=[(alpha,0,1)]
        elif len(ds)==1:
            a=list(alpha);a[ds[0]]+=1;terms=[(tuple(a),1,-1)]
        else:
            if ds[0]==ds[1]:terms.append((alpha,1,1))
            a=list(alpha);a[ds[0]]+=1;a[ds[1]]+=1;terms.append((tuple(a),2,1))
        value=mp.mpf(0)
        for a,profile,sign in terms:
            for (axial,transverse),coefficient in rotated(a,normal).items():
                if coefficient:value+=sign*_mp(coefficient)*axial_integral(axial,transverse,profile)
        results.append(value)
    return results
