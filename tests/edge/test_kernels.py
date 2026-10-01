"""truncated-monomials.md checks that live in Python (the symbolic ones are maple/11_*.mpl)."""
from fractions import Fraction as F

import mpmath as mp
import pytest

from curvbound.kernels import KERNELS as SRC
from edgebound.kernels import KERNELS, disk_moment, kernel, peval
from edgebound.mpq import mpq
from .conftest import KERNELS as NAMES


@pytest.mark.parametrize("name", NAMES)
def test_blocks_reproduce_kernel_exactly(name):
    k = kernel(name)
    for q in [F(0), F(1, 7), F(1, 4), F(1, 2), F(1, 2) + F(1, 1000), F(3, 4), F(99, 100), F(1)]:
        got = sum(peval(list(b.coeffs), q) for b in k.blocks if q <= b.R)        # sum_j q_j 1[r<=R_j]  = pi*W
        ref = SRC[name].c2_pi * SRC[name].w_hat(q) if q < 1 else 0
        assert got == ref, (name, q)


@pytest.mark.parametrize("name", NAMES)
def test_normalisation_and_even_moments_exact(name):
    assert disk_moment(kernel(name), (0, 0)) == 1
    # odd moments vanish, a few even ones are positive rationals
    assert disk_moment(kernel(name), (1, 0)) == 0 and disk_moment(kernel(name), (1, 2)) == 0
    assert disk_moment(kernel(name), (2, 0)) == disk_moment(kernel(name), (0, 2)) > 0
    # sigma^2 style identity: m_(4,0) = 3 m_(2,2) (angular average cos^4 : cos^2 sin^2 = 3 : 1)
    assert disk_moment(kernel(name), (4, 0)) == 3 * disk_moment(kernel(name), (2, 2))


def test_cubic_block_weights():
    """indicator weights per radius of the cubic spline: 8/7 (R=1) and -1/7 (R=1/2)."""
    k = kernel("cubic")
    w = []
    for b in k.blocks:
        w.append(2 * sum(c * b.R ** (n + 2) / (n + 2) for n, c in enumerate(b.coeffs)))
    assert w == [F(8, 7), F(-1, 7)]


def test_disk_moment_vs_quad():
    for name in NAMES:
        k = kernel(name)
        W = lambda r: sum(peval(list(c), r) for lo, hi, c in k.pieces if mpq(lo) <= r <= mpq(hi) and r < 1) / mp.pi
        a, b = 2, 2
        ang = mp.quad(lambda t: mp.cos(t) ** a * mp.sin(t) ** b, [0, mp.pi / 2, mp.pi, 3 * mp.pi / 2, 2 * mp.pi])
        rad = mp.quad(lambda r: r ** (a + b + 1) * W(r), [0, mp.mpf(1) / 2, 1])
        ref = ang * rad
        assert abs(mpq(disk_moment(k, (a, b))) - ref) < mp.mpf(10) ** -30
