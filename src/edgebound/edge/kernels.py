"""Kernels as truncated monomials  W(r) = (1/pi) sum_j sum_n c_{j,n} r^n 1[r <= R_j]  (h = 1).

Built from the exact piecewise-polynomial shapes of `curvbound.kernels`
(the PLAN track's kernel table; only imported, not modified):

    W_hat piecewise polynomial on knots 0 = k_0 < k_1 < ... < k_m = 1
    blocks: R_m = 1 with the outer piece, and for the inner pieces
            R_j = k_j with (piece_j - piece_{j+1}).

All coefficients are exact Fractions (C2 = c2_pi / pi is folded in; the common
1/pi is applied at evaluation time).  Verified in Maple: `maple/11_*.mpl`.
"""
from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

from curvbound.kernels import KERNELS as _SRC


# ---- tiny exact polynomial arithmetic (coefficient lists, ascending powers) ----
def padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]


def pscale(a, c):
    return [c * x for x in a]


def pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def ppow(a, k):
    out = [Fraction(1)]
    for _ in range(k):
        out = pmul(out, a)
    return out


def pderiv(a):
    return [i * a[i] for i in range(1, len(a))]


def pint(a):
    return [Fraction(0)] + [a[i] / (i + 1) for i in range(len(a))]


def peval(a, x):
    acc = 0
    for c in reversed(a):
        acc = acc * x + c
    return acc


@dataclass(frozen=True)
class Block:
    R: Fraction                 # truncation radius (h = 1)
    coeffs: Tuple[Fraction, ...]  # c_n of r^n, pi-free (W = (1/pi) * sum)


@dataclass(frozen=True)
class EdgeKernel:
    name: str
    blocks: Tuple[Block, ...]        # sorted by R descending
    pieces: Tuple[tuple, ...]        # ((lo, hi, coeffs), ...) of pi*W, ascending
    c2_pi: Fraction

    def W_pi(self, r: Fraction) -> Fraction:
        """pi * W(r) at rational r (exact); used by tests."""
        for lo, hi, c in self.pieces:
            if lo <= r <= hi and r < 1:
                return peval(c, r)
        if r == 1:
            return Fraction(0)
        return Fraction(0)

    @property
    def Rs(self):
        return tuple(b.R for b in self.blocks)


def _pieces_from_source(k):
    c = k.c2_pi
    half = Fraction(1, 2)
    if k.kind == "cubic":
        inner = padd(ppow([Fraction(1), Fraction(-1)], 3),
                     pscale(ppow([half, Fraction(-1)], 3), -4))
        outer = ppow([Fraction(1), Fraction(-1)], 3)
        return [(Fraction(0), half, pscale(inner, c)), (half, Fraction(1), pscale(outer, c))]
    pol = [Fraction(0)] * (max(p for p, _ in k.poly) + 1)
    for p, cf in k.poly:
        pol[p] += cf
    shape = pmul(ppow([Fraction(1), Fraction(-1)], k.n), pol)
    return [(Fraction(0), Fraction(1), pscale(shape, c))]


def _blocks_from_pieces(pieces):
    blocks = []
    # outer piece, R = last knot; inner: difference with the next piece
    for j in range(len(pieces) - 1, -1, -1):
        lo, hi, c = pieces[j]
        if j == len(pieces) - 1:
            coeffs = c
        else:
            coeffs = padd(c, pscale(pieces[j + 1][2], -1))
        blocks.append(Block(hi, tuple(Fraction(x) for x in coeffs)))
    return tuple(blocks)


def _build(name):
    k = _SRC[name]
    pieces = _pieces_from_source(k)
    return EdgeKernel(name, _blocks_from_pieces(pieces), tuple((lo, hi, tuple(c)) for lo, hi, c in pieces), k.c2_pi)


KERNELS = {n: _build(n) for n in _SRC}

# --------------------------------------------------------------------------------------------------------------
# Additional exact piecewise-polynomial kernels of warpSPHCore (kernel_specs.yaml), support radius 1:
#   shape = sum_t c_t (knot_t - q)_+^p_t   (truncated powers)  or an explicit polynomial in q (poly6)
# Blocks are the truncated powers themselves (R = knot_t); the pi * W pieces follow by summing the blocks that cover each interval.
# Not included (not piecewise polynomial in r / not exact): HOCT4 (hard-switched linear core), Gaussian.
# --------------------------------------------------------------------------------------------------------------
def _from_terms(name, terms):
    """terms: [(coef, knot, power)] truncated powers; or for poly6 pass the polynomial via _from_poly."""
    from math import comb
    q = [Fraction(0), Fraction(1)]
    blocks = {}
    for c, R, pw in terms:
        R = Fraction(R)
        base = [R, Fraction(-1)]                       # (R - q)
        poly = ppow(base, pw)
        poly = pscale(poly, Fraction(c))
        blocks[R] = padd(blocks.get(R, [Fraction(0)]), poly)
    return _finish(name, blocks)


def _finish(name, blocks):
    Rs = sorted(blocks)                                                 # ascending knots, last = 1
    # normalisation: 2 * int_0^1 r (pi W) dr = 1 with pi W = C * shape  ->  C = 1 / (2 int r shape)
    def shape_integral():
        tot = Fraction(0)
        lo = Fraction(0)
        for j, hi in enumerate(Rs):
            poly = [Fraction(0)]
            for R in Rs:
                if R >= hi:
                    poly = padd(poly, blocks[R])
            P = pint(pmul([Fraction(0), Fraction(1)], poly))
            tot += peval(P, hi) - peval(P, lo)
            lo = hi
        return tot
    C = 1 / (2 * shape_integral())
    bl = tuple(Block(R, tuple(Fraction(x) * C for x in blocks[R])) for R in sorted(Rs, reverse=True))
    pieces = []
    lo = Fraction(0)
    for hi in Rs:
        poly = [Fraction(0)]
        for R in Rs:
            if R >= hi:
                poly = padd(poly, blocks[R])
        pieces.append((lo, hi, tuple(Fraction(x) * C for x in poly)))
        lo = hi
    return EdgeKernel(name, bl, tuple(pieces), C)


def _poly6():
    return _finish("poly6", {Fraction(1): [Fraction(1), Fraction(0), Fraction(-3), Fraction(0), Fraction(3), Fraction(0), Fraction(-1)]})


_F = Fraction
KERNELS["quartic"] = _from_terms("quartic", [(1, _F(1), 4), (-5, _F(3, 5), 4), (10, _F(1, 5), 4)])
KERNELS["quintic"] = _from_terms("quintic", [(1, _F(1), 5), (-6, _F(2, 3), 5), (15, _F(1, 3), 5)])
KERNELS["b7"] = _from_terms("b7", [(1, _F(1), 6), (-7, _F(5, 7), 6), (21, _F(3, 7), 6), (-35, _F(1, 7), 6)])
KERNELS["b8"] = _from_terms("b8", [(1, _F(1), 7), (-8, _F(3, 4), 7), (28, _F(1, 2), 7), (-56, _F(1, 4), 7)])
KERNELS["poly6"] = _poly6()
# cone: shape (1 - q), i.e. K(r) = (r - H) 1[r <= H] = -H (1 - q) at h = H; used by cover.cover_vector_scene (Q3a); normalisation 3
KERNELS["cone"] = _from_terms("cone", [(1, _F(1), 1)])

# 2D normalisation constants C2 * pi of warpSPHCore (kernel_specs.yaml), the independent check of the exact normalisation above
WARPSPH_C2_PI = {"quartic": Fraction(46875, 2398), "quintic": Fraction(15309, 478), "b7": Fraction(5764801, 113149),
                 "b8": Fraction(589824, 7435), "poly6": Fraction(4)}


def kernel(name: str) -> EdgeKernel:
    try:
        return KERNELS[name]
    except KeyError:
        raise KeyError(f"unknown kernel {name!r}; expected one of {sorted(KERNELS)}")


def disk_moment(kern: EdgeKernel, alpha) -> Fraction:
    """EXACT full-support moment  int_{R^2} y^alpha W dA  (a rational number, h = 1)."""
    a, b = alpha
    if a % 2 or b % 2:
        return Fraction(0)
    k = a + b

    def df(n):
        out = 1
        while n > 1:
            out *= n
            n -= 2
        return out
    ang = Fraction(2 * df(a - 1) * df(b - 1), df(k))   # oint w^alpha / pi
    # int_0^1 r^(k+1) pi*W dr, per piece, exact
    rad = Fraction(0)
    for lo, hi, c in kern.pieces:
        P = pint(pmul([Fraction(0)] * (k + 1) + [Fraction(1)], c))
        rad += peval(P, hi) - peval(P, lo)
    return ang * rad            # (oint w^a)/pi * int r^(k+1) (pi W) dr


# ---- powers W^k of the Wendland families as truncated-power terms (the shifting tensile control T = (1/5) grad int W^5) ----
# (base coeffs in the u = 1-q basis, p0, degree per k): w2 shape (1-q)^4 (1+4q), w4 shape (1-q)^6 (1+6q+35/3 q^2)
POWER_FAMILIES = {"w2": ([_F(5), _F(-4)], 4, 5), "w4": ([_F(56, 3), _F(-88, 3), _F(35, 3)], 6, 8)}


def power_terms(k, fam):
    """[(coef, knot=1, power)] of W^k in the truncated-power basis (1-q)^power (the reviewer probe's construction)."""
    base, p0, _ = POWER_FAMILIES[fam]
    pol = [_F(1)]
    for _ in range(k):
        new = [_F(0)] * (len(pol) + len(base) - 1)
        for i, a in enumerate(pol):
            for j, b in enumerate(base):
                new[i + j] += a * b
        pol = new
    return [(c, _F(1), p0 * k + m) for m, c in enumerate(pol)]


def power_monomials(k, fam):
    """exact Fractions a_j with  shape^k = sum_j a_j q^j   (the u-powers (1-q)^p expanded into monomials q^j)."""
    from math import comb
    amax = max(p for _, _, p in power_terms(k, fam))
    a = [_F(0)] * (amax + 1)
    for c, _, p in power_terms(k, fam):
        for j in range(p + 1):
            a[j] += c * comb(p, j) * (-1) ** j
    return a
