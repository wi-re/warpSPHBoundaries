"""edge-value-identity.md checks (stage 1: mpmath reference vs independent polar oracle)."""
from fractions import Fraction as F

import mpmath as mp
import pytest

import edgebound as eb
from edgebound.edge import geometry as G
from edgebound.edge.mpq import mpq
from curvbound import planar2d

from .conftest import KERNELS, big_triangle, rand_point, rand_triangle, tiling

TOL = mp.mpf(10) ** -30


@pytest.mark.parametrize("name", KERNELS)
def test_random_triangles_vs_polar(name, rng):
    for _ in range(6):
        T = rand_triangle(rng)
        x = rand_point(rng)
        v = eb.value(T, x, name)
        o = eb.polar_value(T, x, name)
        assert abs(v - o) < TOL, (T, x)

@pytest.mark.parametrize("name", KERNELS)
def test_half_plane_equals_planar2d(name):
    """HALF-PLANE cross-check: one edge at distance d == lambda_2(d) of docs/derivation.md s.3."""
    worst = mp.mpf(0)
    for d in [F(0), F(1, 10**12), F(1, 1000), F(1, 10), F(3, 10), F(1, 2), F(1, 2) + F(1, 10**6),
              F(7, 10), F(999, 1000), F(1) - F(1, 10**12)]:
        v = eb.value(big_triangle(d), (F(0), F(0)), name)
        ref = mp.mpf(1) / 2 if d == 0 else planar2d(name, mpq(d), dps=60)
        err = abs(v - ref)
        worst = max(worst, err)
        assert err < mp.mpf(10) ** -35, (d, err)
    # d >= 1: support does not reach the wall; d < 0: x inside the solid: 1 - lambda(|d|)
    assert eb.value(big_triangle(F(1)), (F(0), F(0)), name) == 0
    assert eb.value(big_triangle(F(3, 2)), (F(0), F(0)), name) == 0
    for d in [F(1, 10), F(1, 2), F(9, 10)]:
        vin = eb.value(big_triangle(-d), (F(0), F(0)), name)
        assert abs(vin - (1 - planar2d(name, mpq(d), dps=60))) < mp.mpf(10) ** -35


def test_half_plane_x_on_edge_is_one_half():
    for name in KERNELS:
        v = eb.value(big_triangle(F(0)), (F(0), F(0)), name)
        assert abs(v - mp.mpf(1) / 2) < mp.mpf(10) ** -38

@pytest.mark.parametrize("name", KERNELS)
def test_tiled_mesh_covering_support_sums_to_one(name):
    tris = tiling(F(3, 2), 3)                        # covers the unit disk around any |x| < 1/2
    for x in [(F(1, 7), F(-1, 5)), (F(-1, 2), F(0)), (F(1, 2), F(1, 2)),   # on grid vertex (-1/2... cell corners) / edges
              (F(0), F(0))]:
        tot = sum(eb.value(T, x, name) for T in tris)
        assert abs(tot - 1) < mp.mpf(10) ** -35, x


@pytest.mark.parametrize("name", KERNELS)
def test_fan_around_x_vertex_sum(name):
    """x is the common vertex of a fan of 8 triangles covering the support: indicator = angle / 2 pi each."""
    R = F(2)
    ring = [(R, F(0)), (R, R), (F(0), R), (-R, R), (-R, F(0)), (-R, -R), (F(0), -R), (R, -R)]
    x = (F(0), F(0))
    tot = sum(eb.value([x, ring[i], ring[(i + 1) % 8]], x, name) for i in range(8))
    assert abs(tot - 1) < mp.mpf(10) ** -35


@pytest.mark.parametrize("name", KERNELS)
def test_subdivision_invariance(name, rng):
    for _ in range(3):
        T = rand_triangle(rng)
        x = rand_point(rng)
        a, b, c = T
        mid = lambda p, q: ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        v = eb.value(T, x, name)
        v2 = sum(eb.value(S, x, name) for S in ([a, ab, c], [ab, b, c]))
        v4 = sum(eb.value(S, x, name) for S in ([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]))
        assert abs(v - v2) < mp.mpf(10) ** -35 and abs(v - v4) < mp.mpf(10) ** -35


@pytest.mark.parametrize("name", KERNELS)
def test_crossing_an_edge_total_is_continuous(name):
    T = [(F(-3, 10), F(-1, 5)), (F(3, 5), F(-1, 10)), (F(1, 10), F(7, 10))]
    # point on edge 0 (from T[0] to T[1]) at parameter 2/5, then step off by +-eps along the normal
    px = T[0][0] + F(2, 5) * (T[1][0] - T[0][0])
    py = T[0][1] + F(2, 5) * (T[1][1] - T[0][1])
    on = (px, py)
    v0 = eb.value(T, on, name)
    for eps in [F(1, 10**6), F(1, 10**12), F(1, 10**20), F(1, 10**30)]:
        dx, dy = T[1][0] - T[0][0], T[1][1] - T[0][1]
        # inward normal direction (ccw triangle): (-dy, dx), not normalised: exact rational displacement
        for sign in (1, -1):
            xs = (px - sign * eps * dy, py + sign * eps * dx)
            v = eb.value(T, xs, name)
            assert abs(v - v0) < 100 * mpq(eps), (eps, sign)       # |grad| <= O(1): no jump
    g = eb.gradient(T, on, name)
    assert max(abs(g[0]), abs(g[1])) < 5


def test_indicator_from_signs_equals_exact_winding(rng):
    for _ in range(30):
        T = rand_triangle(rng)
        x = rand_point(rng)
        P = G.prepare(T, x)
        assert abs(G.indicator_signs(P) - P.indicator) < mp.mpf(10) ** -35
    # degenerate placements
    T = [(F(0), F(0)), (F(1), F(0)), (F(0), F(1))]
    for x in [(F(1, 2), F(0)), (F(0), F(0)), (F(1, 2), F(1, 2)), (F(0), F(1, 3)), (F(1, 3), F(1, 3)), (F(2), F(2)), (F(-1), F(0))]:
        P = G.prepare(T, x)
        assert abs(G.indicator_signs(P) - P.indicator) < mp.mpf(10) ** -35, x


def test_h_scaling():
    T = [(F(-3, 10), F(-1, 5)), (F(3, 5), F(-1, 10)), (F(1, 10), F(7, 10))]
    x = (F(1, 20), F(1, 10))
    for h in [F(1, 2), F(7, 5)]:
        Th = [(a * h, b * h) for a, b in T]
        xh = (x[0] * h, x[1] * h)
        assert abs(eb.value(Th, xh, "w4", h=h) - eb.value(T, x, "w4")) < mp.mpf(10) ** -35


@pytest.mark.parametrize("name", KERNELS)
def test_unsplit_form_for_polygons_inside_the_innermost_piece(name):
    """Remedy for tiny elements (edge-value-identity.md s.6): if the whole polygon lies inside
    r <= R_1 (innermost kernel piece), value = sum_e z int M_in(r)/r^2 ds with M_in/r^2 a POLYNOMIAL:
    no indicator, no atan.  (Float64 cancellation of the split form is avoided; this checks the identity.)"""
    from edgebound.edge.kernels import kernel as get_kernel
    from edgebound.edge import primitives as pr
    kern = get_kernel(name)
    lo, hi, c = kern.pieces[0]                       # innermost piece, (pi*W) coefficients, r in [0, hi]
    for T, x in [([(F(-1, 10), F(-1, 20)), (F(1, 5), F(-1, 10)), (F(0), F(1, 4))], (F(1, 50), F(1, 30))),
                 ([(F(-1, 10**5), F(-1, 10**5)), (F(2, 10**5), F(-1, 10**5)), (F(0), F(3, 10**5))], (F(0), F(0))),
                 ([(F(-1, 10), F(-1, 20)), (F(1, 5), F(-1, 10)), (F(0), F(1, 4))], (F(-1, 10), F(-1, 20)))]:   # x at a vertex
        P = G.prepare(T, x)
        assert max(mp.sqrt(mpq(v[0]) ** 2 + mpq(v[1]) ** 2) for v in P.verts_rel) < mpq(hi)   # all vertices inside r <= R_1
        tot = mp.mpf(0)
        for e in P.edges:
            for n, cn in enumerate(c):
                if cn == 0:
                    continue
                tot += e.z * mpq(cn) / (n + 2) * (pr.I_all(n, e.s1, e.z)[n] - pr.I_all(n, e.s0, e.z)[n])
        tot /= mp.pi
        assert abs(tot - eb.value(T, x, name)) < mp.mpf(10) ** -35
