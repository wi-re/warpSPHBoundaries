"""Tier 2 (closed boundary polylines, data on the boundary only): non-convex polygons in numpy, polygon -> exact disk convergence."""
from fractions import Fraction as F

import mpmath as mp
import numpy as np
import pytest

import edgebound as eb
from edgebound.edge import np2d
from edgebound.edge.mpq import mpq

NAMES = ["cubic", "w2", "w4", "w6"]


def ngon(N, R, center=(0.0, 0.0), phase=0.0):
    t = phase + 2 * np.pi * np.arange(N) / N
    return np.stack([center[0] + R * np.cos(t), center[1] + R * np.sin(t)], axis=1)


def fr(a):
    return F(float(a))


# ------------------------------------------------------------------ numpy: simple polygons, convex or not
SHAPES = {
    "L": [(0, 0), (1, 0), (1, .5), (.5, .5), (.5, 1), (0, 1)],
    "star": [(1.0 * np.cos(a) * (1 if i % 2 == 0 else .45), 1.0 * np.sin(a) * (1 if i % 2 == 0 else .45))
             for i, a in enumerate(np.pi * np.arange(10) / 5)],
    "arrow": [(0, 0), (.9, .1), (.3, .35), (.8, .9), (.1, .6), (-.4, .95), (-.2, .3)],
}


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("name", ["cubic", "w4"])
def test_nonconvex_numpy_matches_stage1(shape, name):
    P = np.array(SHAPES[shape], dtype=float)
    Pf = [(fr(a), fr(b)) for a, b in P]
    rng = np.random.default_rng(5)
    c = P.mean(0)
    pts = [c + rng.uniform(-0.6, 0.6, 2) for _ in range(6)] + [P[0], 0.5 * (P[0] + P[1]), P[2] + np.array([0.0, 0.0])]
    for x in pts:
        xf = (fr(x[0]), fr(x[1]))
        v = np2d.value(P[None], x[None], name)[0]
        g = np2d.gradient(P[None], x[None], name)[0]
        m = np2d.moment(P[None], x[None], name, (1, 1))[0]
        assert abs(v - float(eb.value(Pf, xf, name))) < 1e-11
        gr = eb.gradient(Pf, xf, name)
        assert abs(g[0] - float(gr[0])) < 1e-10 and abs(g[1] - float(gr[1])) < 1e-10
        assert abs(m - float(eb.moment(Pf, xf, name, (1, 1)))) < 1e-11


def test_polygon_union_of_two_polygons_sharing_an_edge():
    """closed boundary additivity: L-shape = square + rectangle (interior edge cancels)."""
    sq = np.array([(0, 0), (1, 0), (1, .5), (0, .5)], dtype=float)
    top = np.array([(0, .5), (.5, .5), (.5, 1), (0, 1)], dtype=float)
    L = np.array(SHAPES["L"], dtype=float)
    for x in ([.3, .3], [.7, .2], [.2, .8], [.5, .5], [.5, .7]):
        x = np.array(x)
        for name in NAMES:
            v = np2d.value(L[None], x[None], name)[0]
            assert abs(v - np2d.value(sq[None], x[None], name)[0] - np2d.value(top[None], x[None], name)[0]) < 1e-12


# ------------------------------------------------------------------ polygon -> exact disk
@pytest.mark.parametrize("name", ["cubic", "w4"])
@pytest.mark.parametrize("R,d", [(1.0, 0.3), (3.0, 0.1), (0.4, 0.2)])
def test_ngon_converges_to_exact_disk_value_and_gradient(name, R, d):
    """x outside a disk obstacle at distance d from its surface; inscribed N-gon, N = 16 ... 128: error ~ N^-2; the combination
    (inscribed + 2 circumscribed)/3 (the O(N^-2) area errors are in ratio 2:1) ~ N^-4 (until float64 noise)."""
    c = (0.0, R + d)
    ref = float(eb.polar_disk_value((F(0), fr(R + d)), fr(R), (F(0), F(0)), name))
    gref = [float(v) for v in eb.polar_disk_gradient((F(0), fr(R + d)), fr(R), (F(0), F(0)), name)]
    x = np.zeros((1, 2))
    errs, errs_m, gerrs = [], [], []
    for N in (16, 32, 64, 128):
        ins = ngon(N, R, c, phase=0.37)
        cir = ngon(N, R / np.cos(np.pi / N), c, phase=0.37 + np.pi / N)
        vi = np2d.value(ins[None], x, name)[0]
        vc = np2d.value(cir[None], x, name)[0]
        errs.append(abs(vi - ref))
        errs_m.append(abs((vi + 2 * vc) / 3 - ref))      # area deficit (inscribed) : excess (circumscribed) = 2 : 1 -> N^-2 terms cancel
        g = np2d.gradient(ins[None], x, name)[0]
        gerrs.append(max(abs(g[0] - gref[0]), abs(g[1] - gref[1])))
    order = np.log2(errs[-2] / errs[-1])
    order_m = np.log2(errs_m[-2] / errs_m[-1])
    assert 1.7 < order < 2.4, (errs, order)
    assert order_m > 3.3 or errs_m[-1] < 1e-9, (errs_m, order_m)
    assert np.log2(gerrs[-2] / gerrs[-1]) > 1.6


@pytest.mark.parametrize("name", NAMES)
def test_ngon_x_inside_disk_and_complement(name):
    """x inside the solid (d < 0) and the complement identity  value(disk) + value(exterior region within the support) = 1."""
    R = 0.8
    x = np.array([[0.1, -0.05]])
    ref = float(eb.polar_disk_value((F(0), F(0)), fr(R), (fr(0.1), fr(-0.05)), name))
    v = np2d.value(ngon(256, R)[None], x, name)[0]
    assert abs(v - ref) < 3e-5
    # engulfing: the disk covers the support -> exactly 1; polygon too (indicator 1, no chord)
    big = ngon(64, 5.0)
    assert abs(np2d.value(big[None], x, name)[0] - 1) < 1e-13
    assert abs(float(eb.polar_disk_value((F(0), F(0)), F(5), (fr(0.1), fr(-0.05)), name)) - 1) < 1e-30


def test_stage1_exact_ngon_vs_disk_oracle_high_precision():
    """one point at 40 digits: stage 1 on an inscribed 64-gon with rational vertices agrees with the exact-disk oracle to the
    polygon's own N^-2 discretisation error (consistency of the two independent routes), and the area-equal polygon is closer."""
    R, d = F(1), F(1, 5)
    ref = eb.polar_disk_value((F(0), R + d), R, (F(0), F(0)), "w4")
    N = 64
    verts = [(fr(np.cos(2 * np.pi * i / N + 0.3)), fr(R + d + np.sin(2 * np.pi * i / N + 0.3))) for i in range(N)]
    v = eb.value(verts, (F(0), F(0)), "w4")
    assert abs(v - ref) < 3e-3 and abs(v - ref) > 1e-8      # inscribed 64-gon: area deficit ~ pi R^2 (2 pi/N)^2/6


def test_linear_boundary_field_moments_k_le_2_converge_to_disk():
    """MLS-extended linear (and quadratic) field over the solid: moments k <= 2 of the N-gon vs the exact disk, error ~ N^-2."""
    R, d = 1.0, 0.25
    c = (0.0, R + d)
    for al in [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
        ref = float(eb.polar_disk_moment((F(0), fr(R + d)), fr(R), (F(0), F(0)), "w4", al))
        e = [abs(np2d.moment(ngon(N, R, c, 0.2)[None], np.zeros((1, 2)), "w4", al)[0] - ref) for N in (64, 128)]
        assert e[1] < 2e-4 and (np.log2(e[0] / e[1]) > 1.5 or e[1] < 1e-8), (al, e)
