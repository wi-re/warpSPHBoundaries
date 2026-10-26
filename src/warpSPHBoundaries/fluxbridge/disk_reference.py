"""Independent arbitrary-precision volume-shell oracle for 2D solid disks.

The native comparator uses a circle boundary rule, not this radial quadrature.
Normal points outward from the disk centre to the query; h=1 here.
"""
import mpmath as mp
from ..edge.kernels import KERNELS
from .radial_reference import _mp


def invariants(distance,radius,kernel="w2"):
    d,R=_mp(distance),_mp(radius)
    if R<=0 or d < -R:
        raise ValueError("positive disk radius and distance >= -radius required")
    D=R+d
    result=[mp.mpf(0) for _ in range(9)]
    full_end=min(mp.mpf(1),max(mp.mpf(0),-d))
    for a,b,coeff in KERNELS[kernel].pieces:
        coeff=[_mp(c)/mp.pi for c in coeff]
        for lo,hi,full in [( _mp(a),min(_mp(b),full_end),True),
                           (max(_mp(a),abs(d)),min(_mp(b),2*R+d),False)]:
            if hi<=lo:
                continue
            def f(r,column):
                if full:
                    a0,a1,at,an=2*mp.pi,mp.mpf(0),mp.pi,mp.pi
                else:
                    t=(r*r+D*D-R*R)/(2*D*r)
                    t=max(mp.mpf(-1),min(mp.mpf(1),t))
                    theta=mp.acos(t);s=mp.sqrt(1-t*t)
                    a0,a1,at,an=2*theta,2*s,theta-t*s,theta+t*s
                w=sum(c*r**n for n,c in enumerate(coeff) if c)
                ell=sum(n*c*r**(n-2) for n,c in enumerate(coeff) if n and c)
                q=sum(n*(n-2)*c*r**(n-4) for n,c in enumerate(coeff) if n and n!=2 and c)
                values=(r*w*a0,-r*r*w*a1,r**3*w*at,r**3*w*an,
                        r*r*ell*a1,-r**3*ell*at,-r**3*ell*an,
                        r*ell*a0+r**3*q*at,r*ell*a0+r**3*q*an)
                return values[column]
            for column in range(9):
                result[column]+=mp.quad(lambda r:f(r,column),[lo,hi])
    return tuple(result)
