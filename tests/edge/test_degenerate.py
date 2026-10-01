"""Robustness checks listed in backends-and-verification.md (stage 1):
x on an edge / at a vertex / on an edge line, z -> 0, tiny chords, elements << h, non-convex polygon."""
from fractions import Fraction as F

import mpmath as mp
import pytest

import edgebound as eb
from edgebound.mpq import mpq

from .conftest import KERNELS

TOL = mp.mpf(10) ** -30
T0 = [(F(-3, 10), F(-1, 5)), (F(3, 5), F(-1, 10)), (F(1, 10), F(7, 10))]


def _on_edge(T, i, t):
    p, q = T[i], T[(i + 1) % 3]
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def _all_vs_polar(T, x, name, alphas=((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (3, 0), (2, 1)), tol=TOL):
    v, o = eb.value(T, x, name), eb.polar_value(T, x, name)
    assert abs(v - o) < tol, ("value", x)
    g, og = eb.gradient(T, x, name), eb.polar_gradient(T, x, name)
    assert max(abs(g[0] - og[0]), abs(g[1] - og[1])) < tol, ("grad", x)
    for al in alphas:
        m, om = eb.moment(T, x, name, al), eb.polar_moment(T, x, name, al)
        assert abs(m - om) < tol, ("moment", al, x)


@pytest.mark.parametrize("name", ["cubic", "w4"])
def test_x_in_the_interior_of_an_edge(name):
    for i, t in [(0, F(2, 5)), (1, F(1, 2)), (2, F(1, 10**6))]:
        _all_vs_polar(T0, _on_edge(T0, i, t), name)


@pytest.mark.parametrize("name", ["cubic", "w6"])
def test_x_at_a_vertex(name):
    for v in T0:
        _all_vs_polar(T0, v, name)
    # obtuse and very acute vertices
    obtuse = [(F(0), F(0)), (F(1, 2), F(1, 10)), (F(-1, 2), F(1, 10))]
    acute = [(F(0), F(0)), (F(1, 2), F(0)), (F(1, 2), F(1, 1000))]
    for T in (obtuse, acute):
        for v in T:
            _all_vs_polar(T, v, name, alphas=((1, 0), (1, 1), (2, 0)))


@pytest.mark.parametrize("name", ["cubic", "w4"])
def test_x_on_an_edge_line_outside_the_segment(name):
    """z = 0 but the chord does not contain s = 0 (x on the extension of an edge)."""
    T = [(F(0), F(0)), (F(1, 2), F(0)), (F(0), F(1, 2))]
    for x in [(F(-1, 5), F(0)), (F(7, 10), F(0)), (F(0), F(-1, 3)), (F(0), F(9, 10))]:
        _all_vs_polar(T, x, name, alphas=((1, 0), (0, 1), (1, 1), (2, 0)))


@pytest.mark.parametrize("name", KERNELS)
def test_x_on_edge_equals_limit_from_both_sides(name):
    """continuity: V(on edge) = lim V(x +- eps n) -- sweeps z -> 0 down to 1e-40."""
    p = _on_edge(T0, 0, F(2, 5))
    dx, dy = T0[1][0] - T0[0][0], T0[1][1] - T0[0][1]
    v0 = eb.value(T0, p, name)
    m0 = eb.moment(T0, p, name, (1, 1))
    for e in range(2, 41, 6):
        eps = F(1, 10**e)
        for sign in (1, -1):
            xs = (p[0] - sign * eps * dy, p[1] + sign * eps * dx)
            assert abs(eb.value(T0, xs, name) - v0) < 4 * mpq(eps) + mp.mpf(10) ** -35
            assert abs(eb.moment(T0, xs, name, (1, 1)) - m0) < 4 * mpq(eps) + mp.mpf(10) ** -35


@pytest.mark.parametrize("name", ["cubic", "w2"])
def test_z_tiny_vs_polar(name):
    p = _on_edge(T0, 0, F(2, 5))
    dx, dy = T0[1][0] - T0[0][0], T0[1][1] - T0[0][1]
    for e in (10, 20, 30):
        eps = F(1, 10**e)
        for sign in (1, -1):
            xs = (p[0] - sign * eps * dy, p[1] + sign * eps * dx)
            _all_vs_polar(T0, xs, name, alphas=((1, 0), (1, 1), (2, 0)), tol=mp.mpf(10) ** -28)


@pytest.mark.parametrize("name", ["cubic", "w4"])
def test_tiny_chords(name):
    """edge grazing the support circle (chord ~ 3e-12) and a chord of length 1e-13 at the circle."""
    z = F(1) - F(1, 10**24)
    T = [(F(-5), z), (F(5), z), (F(0), F(9))]            # x = origin; wall at y = z: chord = 2 sqrt(1 - z^2)
    chord = 2 * mp.sqrt(1 - mpq(z) ** 2)
    assert 1e-12 < chord < 1e-11
    _all_vs_polar(T, (F(0), F(0)), name, alphas=((0, 1), (0, 2)))
    # chord clipped by the END of the edge: z = 3/5, L = 4/5 exactly; the edge ends 1e-13 inside the circle
    T2 = [(F(-2), F(3, 5)), (-F(4, 5) + F(1, 10**13), F(3, 5)), (F(-1), F(4))]
    _all_vs_polar(T2, (F(0), F(0)), name, alphas=((1, 0), (0, 1), (1, 1)))
    # the knot circle R = 1/2 of the cubic spline: edge at distance 3/10, L = 2/5
    T3 = [(F(-2), F(3, 10)), (-F(2, 5) + F(1, 10**13), F(3, 10)), (F(-1), F(4))]
    _all_vs_polar(T3, (F(0), F(0)), name, alphas=((1, 0), (0, 1)))


@pytest.mark.parametrize("name", ["cubic", "w4"])
@pytest.mark.parametrize("scale", [F(1, 10**3), F(1, 10**6)])
def test_elements_much_smaller_than_h(name, scale):
    base = [(F(-3), F(-2)), (F(6), F(-1)), (F(1), F(7))]
    T = [(a * scale / 10, b * scale / 10) for a, b in base]
    for x in [(F(0), F(0)), (T[0][0], T[0][1]), (F(1, 10) * scale + F(1, 5), F(-1, 7)), _on_edge(T, 1, F(1, 3))]:
        _all_vs_polar(T, x, name, alphas=((1, 0), (1, 1), (2, 0), (3, 0)), tol=mp.mpf(10) ** -28)


@pytest.mark.parametrize("name", ["cubic", "w4"])
def test_nonconvex_polygon_winding_indicator(name):
    """L-shaped polygon, x inside the notch region / inside / outside: equals sum over a triangulation."""
    L = [(F(0), F(0)), (F(1), F(0)), (F(1), F(1, 2)), (F(1, 2), F(1, 2)), (F(1, 2), F(1)), (F(0), F(1))]
    tri = [[L[0], L[1], L[2]], [L[0], L[2], L[3]], [L[0], L[3], L[4]], [L[0], L[4], L[5]]]
    for x in [(F(1, 4), F(1, 4)), (F(3, 4), F(3, 4)), (F(3, 4), F(1, 4)), (F(1, 2), F(1, 2)), (F(1, 4), F(3, 4)), (F(-1, 5), F(1, 2))]:
        v = eb.value(L, x, name)
        vt = sum(eb.value(T, x, name) for T in tri)
        assert abs(v - vt) < mp.mpf(10) ** -33, x
        for al in [(1, 0), (1, 1), (0, 3)]:
            assert abs(eb.moment(L, x, name, al) - sum(eb.moment(T, x, name, al) for T in tri)) < mp.mpf(10) ** -33
    assert abs(eb.value(L, (F(1, 4), F(1, 4)), name) - eb.polar_value(L, (F(1, 4), F(1, 4)), name)) < TOL


def test_clockwise_input_is_reversed():
    cw = T0[::-1]
    for x in [(F(1, 20), F(1, 10)), (F(-1, 2), F(1, 2))]:
        assert abs(eb.value(cw, x, "w4") - eb.value(T0, x, "w4")) < mp.mpf(10) ** -38
        assert abs(eb.moment(cw, x, "w4", (1, 1)) - eb.moment(T0, x, "w4", (1, 1))) < mp.mpf(10) ** -38


def test_degenerate_triangle_rejected():
    with pytest.raises(ValueError):
        eb.value([(F(0), F(0)), (F(1), F(1)), (F(2), F(2))], (F(0), F(0)), "w2")


def test_support_does_not_reach_polygon():
    T = [(F(5), F(5)), (F(6), F(5)), (F(5), F(6))]
    assert eb.value(T, (F(0), F(0)), "w6") == 0
    assert eb.gradient(T, (F(0), F(0)), "w6") == (0, 0)
    assert eb.moment(T, (F(0), F(0)), "w6", (2, 1)) == 0
