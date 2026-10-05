"""moments-recursion.md checks: (a) vs polar (k <= 4), (a) vs (b), full-support meshes, half-plane."""
from fractions import Fraction as F

import mpmath as mp
import pytest

import edgebound as eb
from edgebound.edge.kernels import disk_moment, kernel
from edgebound.edge.mpq import mpq

from .conftest import KERNELS, rand_point, rand_triangle
from .conftest import big_triangle, tiling

TOL = mp.mpf(10) ** -30
ALPHAS = [(a, k - a) for k in range(1, 5) for a in range(k + 1)]


@pytest.mark.parametrize("name", ["cubic", "w6"])
def test_a_vs_polar_k_le_4(name, rng):
    for _ in range(3):
        T = rand_triangle(rng)
        x = rand_point(rng)
        for al in ALPHAS:
            m = eb.moment(T, x, name, al)
            o = eb.polar_moment(T, x, name, al)
            assert abs(m - o) < TOL, (T, x, al)


@pytest.mark.parametrize("name", KERNELS)
def test_a_vs_b_far_field(name, rng):
    for _ in range(4):
        T = rand_triangle(rng)
        x = rand_point(rng)
        for al in [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), (4, 0), (2, 2)]:
            ma = eb.moment(T, x, name, al, method="a")
            mb = eb.moment(T, x, name, al, method="b")
            assert abs(ma - mb) < mp.mpf(10) ** -33, (T, x, al)


def test_b_is_undefined_on_an_edge_line():
    """finding: the far-field form (b) cannot be used at z = 0 (0 * divergent integral)."""
    T = [(F(0), F(0)), (F(1), F(0)), (F(0), F(1))]
    with pytest.raises(ZeroDivisionError):
        eb.moment(T, (F(1, 3), F(0)), "w4", (1, 1), method="b")
    # while (a) is fine there and agrees with the polar oracle
    for al in [(1, 1), (2, 0), (0, 3)]:
        assert abs(eb.moment(T, (F(1, 3), F(0)), "w4", al) - eb.polar_moment(T, (F(1, 3), F(0)), "w4", al)) < TOL


@pytest.mark.parametrize("name", KERNELS)
def test_full_support_mesh_gives_disk_moments(name):
    """covering mesh: sum over elements = exact rational disk moment (no quadrature)."""
    tris = tiling(F(2), 2)                                   # covers the support disk for |x| < 1
    for x in [(F(1, 7), F(-1, 5)), (F(0), F(0)), (F(3, 4), F(0))]:        # mesh vertex / generic / on a mesh edge
        for al in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (4, 0), (2, 2), (0, 4)]:
            tot = sum(eb.moment(T, x, name, al) for T in tris)
            # moments are about the evaluation point: the disk is centred at x, mesh must cover it
            ref = mpq(disk_moment(kernel(name), al))
            assert abs(tot - ref) < mp.mpf(10) ** -33, (x, al, tot, ref)


@pytest.mark.parametrize("name", KERNELS)
def test_half_plane_second_component_first_moment(name):
    """m_(0,1) of the half plane {y > d}: 2 int_d^1 W r sqrt(r^2-d^2) dr (independent 1D quadrature)."""
    k = kernel(name)

    def W(r):
        return sum(sum(c * r ** i for i, c in enumerate(cc)) for lo, hi, cc in k.pieces
                   if mpq(lo) <= r <= mpq(hi) and r < 1) / mp.pi if False else None

    from edgebound.edge.kernels import peval
    def Wf(r):
        tot = mp.mpf(0)
        for lo, hi, cc in k.pieces:
            if mpq(lo) <= r <= mpq(hi):
                tot = peval([mpq(c) for c in cc], r)
                if r > mpq(lo):
                    break
        return tot / mp.pi
    for d in [F(1, 10), F(2, 5), F(3, 5)]:
        dd = mpq(d)
        pts = [dd, mpq(F(1, 2)), mp.mpf(1)] if dd < mpq(F(1, 2)) else [dd, mp.mpf(1)]
        ref = 2 * mp.quad(lambda r: Wf(r) * r * mp.sqrt(r * r - dd * dd), pts)
        m = eb.moment(big_triangle(d), (F(0), F(0)), name, (0, 1))
        assert abs(m - ref) < mp.mpf(10) ** -28, (d, m, ref)
        assert abs(eb.moment(big_triangle(d), (F(0), F(0)), name, (1, 0))) < mp.mpf(10) ** -35     # symmetry


@pytest.mark.parametrize("name", ["cubic", "w4"])
def test_moment_gradient_vs_finite_difference(name, rng):
    for _ in range(2):
        T = rand_triangle(rng)
        x = rand_point(rng)
        for al in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
            g = eb.moment_gradient(T, x, name, al)
            h = F(1, 10**8)
            fd = []
            for dx, dy in [(h, 0), (0, h)]:
                def v(k):
                    return eb.moment(T, (x[0] + k * dx, x[1] + k * dy), name, al, dps=60)
                fd.append((8 * (v(1) - v(-1)) - (v(2) - v(-2))) / (12 * mpq(h)))
            assert max(abs(g[0] - fd[0]), abs(g[1] - fd[1])) < mp.mpf(10) ** -25, (T, x, al)


def test_h_scaling_of_moments():
    T = [(F(-3, 10), F(-1, 5)), (F(3, 5), F(-1, 10)), (F(1, 10), F(7, 10))]
    x = (F(1, 20), F(1, 10))
    h = F(3, 2)
    Th = [(a * h, b * h) for a, b in T]
    xh = (x[0] * h, x[1] * h)
    for al in [(1, 0), (1, 1), (2, 1), (0, 3)]:
        k = sum(al)
        m1 = eb.moment(T, x, "w2", al)
        mh = eb.moment(Th, xh, "w2", al, h=h)
        assert abs(mh - mpq(h) ** k * m1) < mp.mpf(10) ** -35
        g1 = eb.moment_gradient(T, x, "w2", al)
        gh = eb.moment_gradient(Th, xh, "w2", al, h=h)
        assert abs(gh[0] - mpq(h) ** (k - 1) * g1[0]) < mp.mpf(10) ** -35
