"""Independent 3D plane/band volume shells in exact rational angular algebra.

Tangential monomials integrate over azimuth; the remaining polar polynomial
is integrated on each clipped shell. No mesh/flux/device recurrence is used.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import comb,factorial
import mpmath as mp
from ..edge.kernels import KERNELS as BLOCKS
from ..curvbound.kernels import KERNELS as SHAPES
from .radial_reference import _mp


def evaluate(lower,upper,requests,kernel="w2"):
    """Unit support, normal +z, lower <= source_z-query_z <= upper."""
    lower,upper=F(lower),F(upper)
    if upper<=lower:raise ValueError("positive band width required")
    ratio=SHAPES[kernel].c3_pi/SHAPES[kernel].c2_pi

    @lru_cache(None)
    def moment(alpha,n,lo,hi):
        a,b,c=alpha
        if a%2 or b%2:return mp.mpf(0)
        i,j=a//2,b//2;m=i+j
        azimuth=F(2*factorial(2*i)*factorial(2*j),4**m*factorial(i)*factorial(j)*factorial(m))
        knots=sorted({lo,hi,*[abs(x) for x in (lower,upper) if lo<abs(x)<hi]})
        total=mp.mpf(0)
        for left,right in zip(knots[:-1],knots[1:]):
            mid=(left+right)/2
            if lower>=mid or upper<=-mid:continue
            for k in range(m+1):
                e=c+2*k+1
                coefficient=azimuth*F((-1)**k*comb(m,k),e)
                for bound,sign in ((upper,1),(lower,-1)):
                    clipped=bound>=mid if sign==1 else bound<=-mid
                    value=F(1) if sign==1 else F(-1)
                    exponent=n+sum(alpha)+2
                    if not clipped:
                        value=bound;exponent-=e
                    weight=coefficient*sign*value**e
                    if not weight:continue
                    if exponent==-1:
                        integral=mp.log(_mp(right)/_mp(left))
                    else:
                        integral=_mp((right**(exponent+1)-left**(exponent+1))/F(exponent+1))
                    total+=_mp(weight)*integral
        return total

    def higher(a,axis):return tuple(v+int(j==axis) for j,v in enumerate(a))
    out=[]
    for req in requests:
        if len(req.alpha)!=3:raise ValueError("shell reference is 3D")
        value=mp.mpf(0)
        for lo,hi,coeff in BLOCKS[kernel].pieces:
            for n,c in enumerate(coeff):
                if not c:continue
                c=_mp(c*ratio)
                if not req.derivative:
                    value+=c*moment(req.alpha,n,lo,hi)
                elif len(req.derivative)==1:
                    if n:value-=c*n*moment(higher(req.alpha,req.derivative[0]),n-2,lo,hi)
                else:
                    i,j=req.derivative
                    if n and i==j:value+=c*n*moment(req.alpha,n-2,lo,hi)
                    if n and n!=2:value+=c*n*(n-2)*moment(higher(higher(req.alpha,i),j),n-4,lo,hi)
        out.append(value)
    return out
