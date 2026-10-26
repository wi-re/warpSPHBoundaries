"""Independent arbitrary-precision shell quadrature for 3D plane/sphere fixtures.

This oracle integrates the angular caps and radial kernel pieces numerically.
It is never called by native evaluators. Normal points outward from the solid.
"""
import mpmath as mp
from ..edge.kernels import KERNELS as BLOCKS
from ..curvbound.kernels import KERNELS as SHAPES


def _mp(value):
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return mp.mpf(int(value.numerator))/int(value.denominator)
    return mp.mpf(str(value))


def invariants(distance, kernel="w2", radius=None):
    """Nine unit-support invariants; distance is signed outward distance."""
    d=_mp(distance)
    radius=None if radius is None else _mp(radius)
    if radius is not None and (radius<=0 or d < -radius):
        raise ValueError("sphere radius/centre distance is invalid")
    ratio=_mp(SHAPES[kernel].c3_pi/SHAPES[kernel].c2_pi)/mp.pi
    results=[mp.mpf(0) for _ in range(9)]
    # Integrate the actual solid angle, not the native primitive expansion.
    full_end=min(mp.mpf(1),max(mp.mpf(0),-d))
    cap_end=mp.mpf(1) if radius is None else min(mp.mpf(1),2*radius+d)
    for lo,hi,coeff in BLOCKS[kernel].pieces:
        lo,hi=_mp(lo),_mp(hi)
        coeff=[_mp(c)*ratio for c in coeff]
        for left,right,full in [(lo,min(hi,full_end),True),
                                (max(lo,abs(d)),min(hi,cap_end),False)]:
            if right<=left:
                continue

            def integrand(r, column):
                if full:
                    cap=mp.mpf(1)
                elif radius is None:
                    cap=-d/r
                else:
                    centre=radius+d
                    cap=(radius**2-centre**2-r**2)/(2*centre*r)
                # Direct angular integration of 1, z, x*x and z*z.
                a0=2*mp.pi*(cap+1)
                a1=mp.pi*(cap**2-1)
                at=mp.pi*(cap-cap**3/3+mp.mpf(2)/3)
                an=2*mp.pi*(cap**3+1)/3
                w=sum(c*r**n for n,c in enumerate(coeff))
                ell=sum(n*c*r**(n-2) for n,c in enumerate(coeff) if n>=1 and c)
                q=sum(n*(n-2)*c*r**(n-4) for n,c in enumerate(coeff) if n>=1 and n!=2 and c)
                entries=(r*r*w*a0, r**3*w*a1, r**4*w*at, r**4*w*an,
                         -r**3*ell*a1, -r**4*ell*at, -r**4*ell*an,
                         r*r*ell*a0+r**4*q*at, r*r*ell*a0+r**4*q*an)
                return entries[column]
            for column in range(9):
                results[column]+=mp.quad(lambda r: integrand(r,column),[left,right])
    return tuple(results)


def select(values, normal, requests):
    """Rotate axial invariants into the Cartesian baseline request contract."""
    n=tuple(_mp(x) for x in normal)
    out=[]
    for req in requests:
        ids=tuple(i for i,power in enumerate(req.alpha) for _ in range(power))
        ds=req.derivative
        if not ds and not ids:
            value=values[0]
        elif not ds and len(ids)==1:
            value=values[1]*n[ids[0]]
        elif len(ds)==1 and not ids:
            value=values[4]*n[ds[0]]
        else:
            i,j=(ids if not ds else ids+ds)
            t,a=(2,3) if not ds else ((5,6) if len(ds)==1 else (7,8))
            value=(values[t] if i==j else 0)+(values[a]-values[t])*n[i]*n[j]
        out.append(value)
    return tuple(out)
