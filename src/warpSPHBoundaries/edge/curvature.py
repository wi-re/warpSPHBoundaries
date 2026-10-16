"""Closed-form line coefficients for the order-two curved-boundary family.

The local solid is z <= -k*t^2/2 - k_s*t^3/6. Count k as order one
and k_s as order two; retain the quadratic Taylor term in the wall shift.
The production tensor implementation is Tier3.curvature_moments. These exact
line primitives provide a precision-refinable reference without new tables.
"""
from fractions import Fraction
from functools import lru_cache

import mpmath as mp

from .kernels import kernel as get_kernel
from .mpq import mpq
from .primitives import S_all


@lru_cache(None)
def whole_plane_variance(kernel):
    return sum(Fraction(c)*(hi**(n+4)-lo**(n+4))/Fraction(n+4)
               for lo, hi, coefficients in get_kernel(kernel).pieces
               for n, c in enumerate(coefficients))


def line_moment(kernel, distance, power=2, dps=50):
    """B_power(d)=integral t^power W(sqrt(t²+d²)) dt in closed form.

    Polynomial radial pieces reduce to existing S/I primitives (rational,
    square root and asinh). Includes the wall and every support/kernel knot.
    Odd tangential powers vanish exactly.
    """
    if power < 0:
        raise ValueError('line moment needs a nonnegative power')
    if power % 2:
        return mp.mpf(0)
    with mp.workdps(dps+15):
        d = abs(mpq(Fraction(distance)))
        result = mp.mpf(0)
        for lo, hi, coefficients in get_kernel(kernel).pieces:
            if d >= mpq(hi):
                continue
            bottom = mp.sqrt(max(mpq(lo)**2-d*d, 0))
            top = mp.sqrt(mpq(hi)**2-d*d)
            for n, c in enumerate(coefficients):
                if c:
                    result += mpq(c)*(S_all(power, n, top, d)-S_all(power, n, bottom, d))
        return +(2*result/mp.pi)


def moments(kernel, distance, curvature=0, variation=0, normal=(0, 1), dps=50):
    """Precision-refinable closed-form evaluation of the twelve-moment model.

    Inputs use support units, variation is relative to tangent=(-ny,nx).
    This evaluates the SAME order-two model, not the original curved geometry.
    The production spline route is a cheaper approximation of these coefficients.
    """
    from .core import moment
    with mp.workdps(dps+20):
        distance = Fraction(distance)
        d, k, b = map(lambda x: mpq(Fraction(x)), (distance, curvature, variation))
        zero, one = mp.mpf(0), mp.mpf(1)
        variance = mpq(whole_plane_variance(kernel))
        if d >= 1:
            return (zero,)*12
        if d <= -1:
            return one, zero, zero, variance, zero, variance, zero, zero, one, zero, zero, one
        nx, ny = map(lambda x: mpq(Fraction(x)), normal)
        length = mp.sqrt(nx*nx+ny*ny)
        if length == 0:
            raise ValueError('active curved moments require a defined normal')
        n = mp.matrix([nx/length, ny/length])
        t = mp.matrix([-n[1], n[0]])
        B0, B2, B4 = [line_moment(kernel, distance, j, dps+10) for j in (0, 2, 4)]
        if d == 0:
            L, T, N = mp.mpf(1)/2, variance/2, variance/2
        else:
            vertices = [(Fraction(-4), Fraction(-4)), (Fraction(4), Fraction(-4)),
                        (Fraction(4), Fraction(0)), (Fraction(-4), Fraction(0))]
            L, T, N = [moment(vertices, (Fraction(0), distance), kernel, alpha, dps=dps+10)
                       for alpha in ((0, 0), (2, 0), (0, 2))]
        mass = L-k*B2/2+3*d*k*k*B2/8
        mt, mn = -b*B4/6, -B2+k*d*B2/2+k*k*(B4-3*d*d*B2)/8
        stt = T-k*B4/2+5*d*k*k*B4/8
        stn = b*d*B4/6
        snn = N-k*d*d*B2/2+k*k*(-2*d*B4+3*d**3*B2)/8
        gt = -b*B2/2
        gn = -B0+k*d*B0/2+3*k*k*(B2-d*d*B0)/8
        ctt = L-3*k*B2/2+15*d*k*k*B2/8
        ctn = b*d*B2/2
        cnn = L+d*B0-k*d*d*B0/2+k*k*(-6*d*B2+3*d**3*B0)/8
        tt, nn, tn = t*t.T, n*n.T, t*n.T+n*t.T
        first = mt*t+mn*n
        second = stt*tt+stn*tn+snn*nn
        gradient = gt*t+gn*n
        cov = ctt*tt+ctn*tn+cnn*nn
        return tuple([mass, *first, second[0, 0], second[0, 1], second[1, 1],
                      *gradient, cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]])
