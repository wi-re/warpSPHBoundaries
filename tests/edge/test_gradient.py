"""edge-gradient-identity.md checks."""
from fractions import Fraction as F

import mpmath as mp
import pytest

import edgebound as eb
from edgebound.edge.mpq import mpq
from curvbound import planar2d

from .conftest import KERNELS, rand_point, rand_triangle
from .conftest import big_triangle

TOL = mp.mpf(10) ** -30


@pytest.mark.parametrize("name", KERNELS)
def test_gradient_vs_polar(name, rng):
    for _ in range(4):
        T = rand_triangle(rng)
        x = rand_point(rng)
        g = eb.gradient(T, x, name)
        o = eb.polar_gradient(T, x, name)
        assert max(abs(g[0] - o[0]), abs(g[1] - o[1])) < TOL, (T, x)


@pytest.mark.parametrize("name", KERNELS)
def test_gradient_vs_finite_difference_of_value(name, rng):
    """4th-order central differences of the (edge) value identity at 80 digits."""
    for _ in range(3):
        T = rand_triangle(rng)
        x = rand_point(rng)
        g = eb.gradient(T, x, name)
        h = F(1, 10**8)
        fd = []
        for dx, dy in [(h, 0), (0, h)]:
            def v(k):
                return eb.value(T, (x[0] + k * dx, x[1] + k * dy), name, dps=60)
            fd.append((8 * (v(1) - v(-1)) - (v(2) - v(-2))) / (12 * mpq(h)))
        assert max(abs(g[0] - fd[0]), abs(g[1] - fd[1])) < mp.mpf(10) ** -25, (T, x)


@pytest.mark.parametrize("name", KERNELS)
def test_half_plane_gradient_is_minus_dlambda_dd(name):
    """grad_x lambda along the wall normal = -d lambda_2/d d  (x moving toward the wall, n = (0,-1))."""
    for d in [F(1, 100), F(1, 10), F(3, 10), F(1, 2), F(7, 10), F(95, 100)]:
        g = eb.gradient(big_triangle(d), (F(0), F(0)), name)
        with mp.workdps(70):
            f = lambda t: planar2d(name, t, dps=70)
            h, x0 = mp.mpf(10) ** -12, mpq(d)                       # 4th-order stencil: error ~ h^4
            dl = (8 * (f(x0 + h) - f(x0 - h)) - (f(x0 + 2 * h) - f(x0 - 2 * h))) / (12 * h)
        assert abs(g[0]) < mp.mpf(10) ** -35                      # tangential component vanishes (symmetric chord)
        assert abs(g[1] + dl) < mp.mpf(10) ** -30, (d, g[1], -dl)


@pytest.mark.parametrize("name", KERNELS)
def test_gradient_across_knot_radius_cubic(name, rng):
    """edge of a triangle tangent to the R = 1/2 circle of the cubic spline: block jumps cancel."""
    T = [(F(-2), F(-1, 2)), (F(2), F(-1, 2)), (F(0), F(3))]
    for x in [(F(0), F(0)), (F(1, 10**9), F(0)), (F(1, 3), F(1, 100))]:
        g = eb.gradient(T, x, name)
        o = eb.polar_gradient(T, x, name)
        assert max(abs(g[0] - o[0]), abs(g[1] - o[1])) < TOL
