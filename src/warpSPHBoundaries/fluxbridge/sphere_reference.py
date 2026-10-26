"""Independent exact angular-cap/Laurent reference for weighted solid spheres.

Cartesian angular moments use rational Wick contractions of the tangent-plane
projector, followed by exact polynomial integration in cos(theta). Substitution
of the sphere's cap boundary produces rational radial Laurent coefficients;
only their final powers/logarithms are evaluated with mpmath. This oracle has
no native sampling rule and is never called by a device provider.
"""
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from math import comb, factorial, prod
import mpmath as mp
from ..edge.kernels import KERNELS as BLOCKS
from ..curvbound.kernels import KERNELS as SHAPES


def _add(poly, exponent, coefficient):
    poly[exponent] = poly.get(exponent, F(0)) + coefficient


@lru_cache(None)
def angular_cap(alpha, normal):
    """Coefficients of integral(omega**alpha, omega.n <= t), divided by pi."""
    n = tuple(F(x) for x in normal)
    if sum(x*x for x in n) != 1:
        raise ValueError('reference normal must be exactly unit rational')
    projection = tuple(tuple(F(i == j)-n[i]*n[j] for j in range(3)) for i in range(3))

    @lru_cache(None)
    def gaussian(beta):
        if not any(beta): return F(1)
        if sum(beta) % 2: return F(0)
        i = next(i for i, b in enumerate(beta) if b)
        rest = list(beta); rest[i] -= 1
        value = F(0)
        for j, count in enumerate(rest):
            if count:
                smaller = rest.copy(); smaller[j] -= 1
                value += count*projection[i][j]*gaussian(tuple(smaller))
        return value

    out = {}
    for beta in product(*(range(a+1) for a in alpha)):
        if sum(beta) % 2: continue
        k = sum(beta)//2
        coefficient = (2*gaussian(beta)*prod(comb(a,b)*nj**(a-b) for a,b,nj in zip(alpha,beta,n))
                       / (2**k*factorial(k)))
        if not coefficient: continue
        axial = sum(alpha)-2*k
        for j in range(k+1):
            exponent = axial+2*j+1
            value = coefficient*((-1)**j)*comb(k,j)/exponent
            _add(out,exponent,value)
            _add(out,0,-value*((-1)**exponent))
    return {p:c for p,c in out.items() if c}


def _mp(value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def evaluate(distance, radius, requests, kernel='w2', normal=(F(2,3),F(-1,3),F(2,3))):
    """Unit-support volume moments and kernel-only derivatives through order 2."""
    d, radius = F(distance), F(radius)
    if radius <= 0 or d < -radius: raise ValueError('invalid sphere distance/radius')
    centre = radius+d
    ratio = SHAPES[kernel].c3_pi/SHAPES[kernel].c2_pi
    n = tuple(F(x) for x in normal)
    cap_a = (radius*radius-centre*centre)/(2*centre) if centre else F(0)
    cap_b = -1/(2*centre) if centre else F(0)
    full_end = min(F(1),max(F(0),-d))
    cap_end = min(F(1),2*radius+d)
    results = []
    for request in requests:
        alpha = tuple(request.alpha); order = len(request.derivative)
        terms = []
        if order == 0: terms.append((alpha,0,F(1)))
        elif order == 1:
            a = list(alpha); a[request.derivative[0]] += 1
            terms.append((tuple(a),1,F(-1)))
        else:
            i,j = request.derivative
            if i == j: terms.append((alpha,1,F(1)))
            a = list(alpha); a[i] += 1; a[j] += 1
            terms.append((tuple(a),2,F(1)))
        exact = F(0); logarithms = {}
        for lo,hi,coefficients in BLOCKS[kernel].pieces:
            for left,right,full in ((lo,min(hi,full_end),True),
                                    (max(lo,abs(d)),min(hi,cap_end),False)):
                if right <= left: continue
                radial = {}
                for angular_alpha, profile, sign in terms:
                    angular = angular_cap(angular_alpha,n)
                    if full:
                        angular_radial = {0:sum(angular.values())}
                    else:
                        angular_radial = {}
                        for power,c in angular.items():
                            for k in range(power+1):
                                _add(angular_radial,2*k-power,c*comb(power,k)*cap_a**(power-k)*cap_b**k)
                    for power,c in enumerate(coefficients):
                        if profile == 1: c *= power
                        elif profile == 2: c *= power*(power-2)
                        if not c: continue
                        base = sum(alpha)+power+2-order
                        for extra,value in angular_radial.items():
                            _add(radial,base+extra,sign*c*ratio*value)
                for power,c in radial.items():
                    if not c: continue
                    if power == -1:
                        key = right/left
                        logarithms[key] = logarithms.get(key,F(0))+c
                    else:
                        exact += c*(right**(power+1)-left**(power+1))/(power+1)
        # Exact rational summation avoids near-centre cancellation. Logarithmic
        # cubic terms get extra precision for the same geometric conditioning.
        extra = 30
        if centre and centre < radius:
            extra += (sum(alpha)+len(request.derivative)+3)*max(0,int(mp.ceil(mp.log10(_mp(radius/centre)))))
        with mp.workdps(mp.mp.dps+extra):
            result = _mp(exact)+sum(_mp(c)*mp.log(_mp(key)) for key,c in logarithms.items() if c)
        results.append(+result)
    return results
