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
