"""Tier 3 in 2D: closest-point (tubular) curvature expansion of the value integral, lambda(q, alpha) = F0 + kappa F1 + kappa^2 F2 + O(kappa^3).

Derived in `maple/14_tier3_2d.mpl` (J = 1 - kappa s; |y|^2 = (d+s)^2 + (1+kappa d)(1-kappa s) t^2 g(kappa t), g(u) = 2(1-cos u)/u^2) and expanded as a
Taylor series of Phi(sigma) = W(sqrt(sigma)) (sigma = |y|^2) around the planar sigma_0 = (d+s)^2 + t^2:

    F0 = int Phi
    F1 = int [ Phi' y1^2 (2d - y2) - (y2 - d) Phi ]
    F2 = int [ Phi' (-d (y2-d) y1^2 - y1^4/12) + Phi''/2 y1^4 (2d - y2)^2 - (y2-d) Phi' y1^2 (2d - y2) ]

over the half plane y2 >= d (y1 = t, y2 = d + s).  Every term is a half-plane MOMENT of a radial profile (Phi, Phi' = W'/(2r), Phi'' = (W'/r)'/(4r)... as exact
polynomial blocks, with powers down to r^-3 for the cubic spline), so F0, F1, F2 are evaluated in closed form with the edge machinery
(single edge at distance d: stage 1, `core.block_moment`).  kappa > 0 for a convex solid (disk obstacle); kappa < 0: the solid is the complement of a disk and
x is inside the hole (same series, analytic continuation, checked).
"""
from fractions import Fraction
from functools import lru_cache

import mpmath as mp
import sympy as sp

from . import core, geometry
from .kernels import kernel as get_kernel
from .mpq import mpq


def _blocks(kname):
    return [(Fraction(b.R), {n: Fraction(c) for n, c in enumerate(b.coeffs) if c != 0}) for b in get_kernel(kname).blocks]


def _deriv_blocks(blocks, order):
    """profiles of Phi^(order)(sigma), Phi(sigma) = W(sqrt sigma): order 1 -> W'/(2r): n c_n r^(n-2)/2; order 2 -> n(n-2) c_n r^(n-4)/4."""
    out = []
    for R, P in blocks:
        Q = {}
        for n, c in P.items():
            f = {0: Fraction(1), 1: Fraction(n, 2), 2: Fraction(n * (n - 2), 4)}[order]
            if f != 0:
                Q[n - 2 * order] = Q.get(n - 2 * order, 0) + c * f
        out.append((R, Q))
    return out


@lru_cache(maxsize=None)
def _F_terms():
    """integrands of F1, F2 as {(order, a, b): coefficient polynomial in d} (sympy), moments m_{ab}[Phi^(order)]."""
    y1, y2, d = sp.symbols("y1 y2 d")
    P0, P1, P2 = sp.symbols("P0 P1 P2")                    # Phi, Phi', Phi'' placeholders
    s = y2 - d
    F1 = P1 * y1 ** 2 * (2 * d - y2) - s * P0
    F2 = P1 * (-d * s * y1 ** 2 - y1 ** 4 / 12) + P2 / 2 * y1 ** 4 * (2 * d - y2) ** 2 - s * P1 * y1 ** 2 * (2 * d - y2)
    out = {}
    for name, F in (("F1", F1), ("F2", F2)):
        poly = sp.Poly(sp.expand(F), P0, P1, P2, y1, y2)
        terms = {}
        for (e0, e1, e2, a, b), coef in poly.terms():
            order = 0 if e0 else (1 if e1 else 2)
            terms[(order, a, b)] = terms.get((order, a, b), 0) + coef
        out[name] = terms
    out["F0"] = {(0, 0, 0): sp.Integer(1)}
    return out


def halfplane_moment(blocks, alpha, d, dps=40):
    """int_{y2 >= d} y^alpha f(r) dA for a profile given as exact blocks [(R, {n: c})] (pi-free): result * 1/pi, x at the origin."""
    big = [(Fraction(-40), Fraction(d)), (Fraction(40), Fraction(d)), (Fraction(0), Fraction(80))]
    prep = geometry.prepare(big, (Fraction(0), Fraction(0)), 1)
    cache = {}
    tot = mp.mpf(0)
    for R, P in blocks:
        for n, c in P.items():
            tot += mpq(c) * core.block_moment(prep, alpha, n, R, cache)
    return tot / mp.pi


def F_k(kname, d, k, dps=40):
    """F_k(d) (h = 1) in closed form via half-plane moments (k = 0, 1, 2)."""
    d = Fraction(d)
    with mp.workdps(dps + core.GUARD):
        blocks = {0: _blocks(kname)}
        blocks[1] = _deriv_blocks(blocks[0], 1)
        blocks[2] = _deriv_blocks(blocks[0], 2)
        tot = mp.mpf(0)
        for (order, a, b), coef in _F_terms()[f"F{k}"].items():
            cf = sp.Poly(coef, sp.Symbol("d")) if coef.free_symbols else None
            val = sp.Rational(coef.subs(sp.Symbol("d"), sp.Rational(d.numerator, d.denominator))) if coef.free_symbols else sp.Rational(coef)
            c = Fraction(int(val.p), int(val.q))
            if c == 0:
                continue
            tot += mpq(c) * halfplane_moment(blocks[order], (a, b), d, dps)
        return +tot


def lambda_expansion(kname, d, kappa, order=2, dps=40):
    """F0 + kappa F1 + kappa^2 F2 (truncated at `order`), support radius 1."""
    kappa = Fraction(kappa)
    return sum((mpq(kappa) ** k * F_k(kname, d, k, dps) for k in range(order + 1)), mp.mpf(0))


def lambda_exact_disk(kname, d, kappa, dps=40):
    """exact lambda for a convex disk (kappa > 0, R = 1/kappa, x outside at distance d) or for a cavity (kappa < 0: solid = complement of the disk
    of radius R = 1/|kappa|, x inside at distance d from the wall)."""
    from . import tier4
    kappa, d = Fraction(kappa), Fraction(d)
    if kappa > 0:
        R = 1 / kappa
        return tier4.arc_value(kname, (Fraction(0), R + d), R, dps=dps)
    R = 1 / (-kappa)
    # fluid disk of radius R around the centre; x at distance R - d from the centre; solid = everything else -> 1 - (disk integral)
    return 1 - tier4.arc_value(kname, (Fraction(0), Fraction(0)), R, x=(Fraction(0), R - d), dps=dps)


def gradient_expansion(kname, d, kappa, order=2, dps=40, h=Fraction(1, 10**8)):
    """|grad_x lambda| along the surface normal (towards the fluid):  d lambda / d d  (the particle moves along the normal; the closest-point
    curvature of a circle is constant).  4th-order central difference of the closed-form expansion (the F_k are smooth in d)."""
    d = Fraction(d)
    with mp.workdps(dps + 10):
        f = lambda dd: lambda_expansion(kname, dd, kappa, order, dps + 10)
        hh = mpq(h)
        return (8 * (f(d + h) - f(d - h)) - (f(d + 2 * h) - f(d - 2 * h))) / (12 * hh)


def gradient_exact_disk(kname, d, kappa, dps=40):
    """exact grad_x lambda of the disk (polar oracle) projected on the normal pointing towards the fluid (away from the solid)."""
    from .oracle import polar_disk_gradient
    kappa, d = Fraction(kappa), Fraction(d)
    if kappa > 0:
        R = 1 / kappa
        g = polar_disk_gradient((Fraction(0), R + d), R, (Fraction(0), Fraction(0)), kname, dps=dps)
        return -g[1]                                   # normal into the fluid is -y for the centre at +y; grad lambda points into the solid
    R = 1 / (-kappa)
    g = polar_disk_gradient((Fraction(0), Fraction(0)), R, (Fraction(0), R - d), kname, dps=dps)   # fluid disk: lambda = 1 - disk integral
    return g[1]                                        # d(1 - disk)/dx_y ; moving towards the wall (+y) decreases d
