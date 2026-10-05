"""Tier 4 in 2D (codimension-1 and "point" objects): mean-value series for a strip and a disk, and the exact references.

Strip of half-width a around a straight centreline, x at distance d (d > a) from the centreline:
    value = int ds int_{-a}^{a} du W(sqrt(s^2 + (d+u)^2)) = sum_{k>=0} 2 a^{2k+1}/(2k+1)!  int ds  d_z^{2k} W(sqrt(s^2+z^2))|_{z=d}
Disk of radius a centred at distance D from x (a "2D point / fibre cross-section"):
    value = pi a^2 sum_{k>=0} a^{2k}/(4^k k! (k+1)!) (Delta^k W)(D)                    (2D mean-value series, Delta radial Laplacian)

Exact references: strip = lambda_2(d-a) - lambda_2(d+a) (planar closed forms of the PLAN track; two edges of a polygon);
disk = polar oracle (`oracle.polar_disk_value`) and the circle edge identity (`arc_value`: divergence theorem on the circle).
All series terms use exact kernel polynomials (sympy) evaluated in mpmath; the series is ASYMPTOTIC in a/h and needs a smooth kernel
at the evaluation radii (not at a knot / the support rim).
"""
from fractions import Fraction
from functools import lru_cache

import mpmath as mp
import sympy as sp

from .kernels import kernel as get_kernel
from .mpq import mpq

_r, _s, _z = sp.symbols("r s z", real=True)


def _piece_poly(kname, which):
    lo, hi, c = get_kernel(kname).pieces[which]
    return sum(sp.Rational(cc.numerator, cc.denominator) * _r ** n for n, cc in enumerate(c) if cc != 0), lo, hi


def _which_piece(kname, r):
    for i, (lo, hi, c) in enumerate(get_kernel(kname).pieces):
        if lo <= r <= hi:
            return i
    return None


@lru_cache(maxsize=None)
def _lap_k(kname, which, k):
    """(Delta^k piW)(r) for a radial piece polynomial (exact sympy expression in r)."""
    P, lo, hi = _piece_poly(kname, which)
    f = P
    for _ in range(k):
        f = sp.simplify(sp.diff(f, _r, 2) + sp.diff(f, _r) / _r)
    return f


def disk_series(kname, a, D, K, dps=40):
    """K+1 terms of the 2D mean-value series for a disk of radius a at distance D (kernel smooth on [D-a, D+a])."""
    af, Df = Fraction(a), Fraction(D)
    wlo, whi = _which_piece(kname, Df - af), _which_piece(kname, Df + af)
    with mp.workdps(dps):
        a, D = mpq(af), mpq(Df)
        if wlo != whi:
            raise ValueError("disk straddles a kernel knot / the support rim: series not applicable")
        tot = mp.mpf(0)
        for k in range(K + 1):
            term = sp.lambdify(_r, _lap_k(kname, wlo, k), "mpmath")(D)
            tot += mp.pi * a ** (2 * k + 2) * term / (4 ** k * mp.factorial(k) * mp.factorial(k + 1))
        return tot / mp.pi


def strip_series(kname, a, d, K, dps=40):
    """K+1 terms of the strip series (x at distance d > a from the centreline); chord integrals by quadrature (smooth for d > 0)."""
    with mp.workdps(dps):
        a, d = mpq(Fraction(a)), mpq(Fraction(d))
        pieces = get_kernel(kname).pieces
        # W(sqrt(s^2+z^2)) piecewise: use the piece that contains the radius; pieces split the s-range at the knot radii
        tot = mp.mpf(0)
        for k in range(K + 1):
            fk = []
            for which in range(len(pieces)):
                P, lo, hi = _piece_poly(kname, which)
                expr = sp.diff(P.subs(_r, sp.sqrt(_s ** 2 + _z ** 2)), _z, 2 * k)
                fk.append((sp.lambdify((_s, _z), expr, "mpmath"), mpq(lo), mpq(hi)))

            def integrand(s):
                r = mp.sqrt(s * s + d * d)
                for f, lo, hi in fk:
                    if lo <= r <= hi:
                        return f(s, d)
                return mp.mpf(0)
            # breakpoints: where r hits a knot radius
            br = [mp.mpf(0)]
            for f, lo, hi in fk:
                for R in (lo, hi):
                    if R > d:
                        br.append(mp.sqrt(R * R - d * d))
            br = sorted(set(br))
            val = 2 * sum(mp.quad(integrand, [br[i], br[i + 1]]) for i in range(len(br) - 1))     # symmetric in s
            tot += 2 * a ** (2 * k + 1) / mp.factorial(2 * k + 1) * val
        return tot / mp.pi


def strip_exact(kname, a, d, dps=40):
    """exact strip value for x outside the strip (d > a): lambda_2(d - a) - lambda_2(d + a)  (PLAN-track closed forms)."""
    from curvbound import planar2d
    with mp.workdps(dps + 10):
        a, d = mpq(Fraction(a)), mpq(Fraction(d))
        lam = lambda t: planar2d(kname, t, dps=dps + 10) if t < 1 else mp.mpf(0)
        return +(lam(d - a) - lam(d + a))


def arc_value(kname, center, Rd, x=(0, 0), nodes=40, dps=40):
    """value of the solid DISK via the divergence theorem on the circle (the edge identity on a curved boundary):
        int_disk W = 1[x in disk] + oint (n.y) (M(r) - M(R_support))/r^2 dl   over the arcs inside the support radius,
    arcs split at the angles where the circle crosses a kernel radius; Gauss-Legendre per arc.  Independent of the polar oracle."""
    from .kernels import pint
    with mp.workdps(dps + 10):
        cx, cy = mpq(Fraction(center[0])) - mpq(Fraction(x[0])), mpq(Fraction(center[1])) - mpq(Fraction(x[1]))
        Rr = mpq(Fraction(Rd))
        du = mp.sqrt(cx * cx + cy * cy)
        pieces = get_kernel(kname).pieces
        # M(r) = int_0^r t W dt (piecewise polynomial of the pi*W pieces) / pi ;  Rsup = 1
        Ms = []
        acc = mp.mpf(0)
        for lo, hi, c in pieces:
            tw = [Fraction(0)] + list(c)                       # t * (pi W): shift coefficients by one power
            Pi = pint(tw)                                      # int t pi W dt
            acc_piece_start = sum(mpq(v) * mpq(lo) ** i for i, v in enumerate(Pi))
            acc_piece_end = sum(mpq(v) * mpq(hi) ** i for i, v in enumerate(Pi))
            Ms.append((mpq(lo), mpq(hi), [mpq(v) for v in Pi], acc - acc_piece_start))
            acc += acc_piece_end - acc_piece_start
        Msup = acc                                              # pi * M(1) = 1/2  (normalisation)

        def Mfun(r):
            for lo, hi, Pi, off in Ms:
                if r <= hi:
                    return (sum(v * r ** i for i, v in enumerate(Pi)) + off)
            return Msup

        # angles on the circle where |p| equals a kernel radius (p = x' - x = centre + Rr (cos,sin))
        breaks = [mp.mpf(0), 2 * mp.pi]
        for R in sorted({hi for lo, hi, c in pieces}):
            Rq = mpq(R)
            if du > 0:
                cs = (Rq ** 2 - du ** 2 - Rr ** 2) / (2 * du * Rr)
                # |centre + Rr e(phi)|^2 = du^2 + Rr^2 + 2 du Rr cos(phi - th_c) = R^2,  th_c = atan2(cy, cx)
                if abs(cs) <= 1:
                    thc = mp.atan2(cy, cx)
                    a_ = mp.acos(cs)
                    for b in (thc + a_, thc - a_):
                        breaks.append(b % (2 * mp.pi))
        breaks = sorted(set(breaks))
        total = mp.mpf(0)
        inside = du < Rr
        for t0, t1 in zip(breaks[:-1], breaks[1:]):
            mid = (t0 + t1) / 2
            p = (cx + Rr * mp.cos(mid), cy + Rr * mp.sin(mid))
            if mp.sqrt(p[0] ** 2 + p[1] ** 2) >= mpq(max(hi for lo, hi, c in pieces)):
                continue                                         # arc outside the support: contributes via the indicator only

            def f(ph):
                px, py = cx + Rr * mp.cos(ph), cy + Rr * mp.sin(ph)
                r = mp.sqrt(px * px + py * py)
                nx, ny = mp.cos(ph), mp.sin(ph)                 # outward normal of the disk
                ndoty = nx * px + ny * py
                return ndoty * (Mfun(r) - Msup) / (r * r) * Rr
            total += mp.quad(f, [t0, t1])
        ind = mp.mpf(1) if inside else mp.mpf(0)
        return ind + total / mp.pi
