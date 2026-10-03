"""Tests for the exact Barecasco cover vector (edgebound.cover) against the independent polar oracle.

Tolerances (stated with their reason):
  * mp closed form vs oracle  <= 1e-12 (relative to H^2): both evaluate the SAME smooth integral
    at high precision (mp at dps = 40 + GUARD, the oracle's polar quadrature at dps = 40 with an
    internal +15 guard); the only difference is exact antiderivative vs adaptive quadrature, so the
    residual is at/below float64 round-off of an O(H^2)-magnitude number. 1e-12 is ~6 orders above
    the float64 epsilon (1e-16) -- a comfortable, non-vacuous bound.
  * numpy vs mp               <= 1e-12 (relative to H^2): the numpy closed form is float64, mp is
    high precision; the difference is a few ulps of an O(H^2) number (measured ~1e-16).
"""
import math

import mpmath as mp
import numpy as np
import pytest

from edgebound import cover, geometry as G, oracle
from edgebound.core import GUARD, block_grad

TOL = 1e-12                     # relative to H^2 (see module docstring)

# --------------------------------------------------------------------------- polygons
UNIT_SQ = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
L_SHAPE = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]            # non-convex (reflex at (1,1))
TRI30 = [(0.0, 0.0), (1.0, 0.0), (math.cos(math.radians(30)), math.sin(math.radians(30)))]     # 30-deg acute corner at (0,0)
LARGE_SQ = [(-5.0, -5.0), (5.0, -5.0), (5.0, 0.0), (-5.0, 0.0)]                                # half-plane-like (solid below y = 0)
POLYGONS = {"unit_square": UNIT_SQ, "L_shape": L_SHAPE, "triangle_30": TRI30, "large_square": LARGE_SQ}


def oracle_grad(verts, x, H, dps=40):
    """grad_x int_T K dA for K = (r - H) 1[r <= H] via the independent polar oracle (true units)."""
    gx = -oracle.polar_moment_profile(verts, x, (1, 0), [(0.0, H, {-1: 1})], h=1.0, dps=dps)
    gy = -oracle.polar_moment_profile(verts, x, (0, 1), [(0.0, H, {-1: 1})], h=1.0, dps=dps)
    return (float(gx), float(gy))


def _bbox(verts):
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]
    return min(xs), max(xs), min(ys), max(ys)


def structured_points(verts, H):
    """inside / outside / within 0.05 H of an edge / within 0.05 H of a corner / on an edge / at a vertex."""
    minx, maxx, miny, maxy = _bbox(verts)
    cx = 0.5 * (minx + maxx); cy = 0.5 * (miny + maxy)
    return [(cx, cy), (maxx + 2.0, maxy + 2.0), (cx, miny + 0.03 * H), (minx + 0.03 * H, miny + 0.03 * H),
            (cx, miny), verts[0]]


def random_points(verts, n, seed):
    minx, maxx, miny, maxy = _bbox(verts)
    rng = np.random.default_rng(seed)
    m = 1.5
    return [(float(minx - m + rng.random() * (maxx - minx + 2 * m)), float(miny - m + rng.random() * (maxy - miny + 2 * m))) for _ in range(n)]


def _cmp(mp_val, ref):
    return math.hypot(float(mp_val[0]) - ref[0], float(mp_val[1]) - ref[1])


# --------------------------------------------------------------------------- smoke reference (reviewer's throw-away value)
def test_smoke_reference_unit_square():
    mpv = cover.cover_gradient_mp(UNIT_SQ, (0.3, 0.4), 1.0)
    npv = cover.cover_vector_np(np.array([0.3, 0.4]), UNIT_SQ, 1.0)
    for v in (mpv, npv):
        assert abs(float(v[0]) - (-0.3457904)) < 1e-6
        assert abs(float(v[1]) - (-0.1702250)) < 1e-6


# --------------------------------------------------------------------------- mp vs oracle, np vs mp
def _check_polygon(verts, H, n_random=200, oracle_subset_at_this_H=200, seed=0):
    """Return (worst |mp-ora|/H^2, worst |np-mp|/H^2, n_oracle_checked)."""
    pts = structured_points(verts, H) + random_points(verts, n_random, seed)
    worst_oracle = 0.0
    worst_np = 0.0
    n_ora = 0
    for k, p in enumerate(pts):
        mpv = cover.cover_gradient_mp(verts, p, H)
        npv = cover.cover_vector_np(np.asarray(p, dtype=np.float64), verts, H)
        worst_np = max(worst_np, _cmp(mpv, npv) / (H * H))
        is_struct = k < len(structured_points(verts, H))
        if is_struct or k - len(structured_points(verts, H)) < oracle_subset_at_this_H:
            ov = oracle_grad(verts, p, H)
            worst_oracle = max(worst_oracle, _cmp(mpv, ov) / (H * H))
            n_ora += 1
    return worst_oracle, worst_np, n_ora


@pytest.mark.parametrize("name,verts", list(POLYGONS.items()))
@pytest.mark.parametrize("H", [1.0, 0.3])
def test_mp_vs_oracle_and_np_vs_mp(name, verts, H):
    # all 200 random points checked against the oracle at H = 1; a 50-point subset at H = 0.3 (scaling)
    n_oracle = 200 if H == 1.0 else 50
    worst_oracle, worst_np, n_ora = _check_polygon(verts, H, n_random=200, oracle_subset_at_this_H=n_oracle)
    assert worst_oracle <= TOL, "mp vs oracle worst = %.3e (H=%s, %s, %d oracle pts)" % (worst_oracle, H, name, n_ora)
    assert worst_np <= TOL, "np vs mp worst = %.3e (H=%s, %s)" % (worst_np, H, name)


def test_polygon_completely_inside_and_outside_the_disk():
    for name, verts in POLYGONS.items():
        minx, maxx, miny, maxy = _bbox(verts)
        cx = 0.5 * (minx + maxx); cy = 0.5 * (miny + maxy)
        rmax = max(math.hypot(v[0] - cx, v[1] - cy) for v in verts)
        # polygon entirely inside the support disk: x = centre, H = 3 rmax
        H_in = 3.0 * rmax
        for fn in (lambda: cover.cover_gradient_mp(verts, (cx, cy), H_in), lambda: tuple(cover.cover_vector_np(np.array([cx, cy]), verts, H_in))):
            v = fn(); ov = oracle_grad(verts, (cx, cy), H_in)
            assert _cmp(v, ov) / (H_in * H_in) <= TOL, name
        # polygon entirely outside: x far away, H = 1 -> empty intersection -> zero
        xout = (maxx + 5.0, maxy + 5.0)
        for H in (1.0, 0.3):
            for fn in (lambda: cover.cover_gradient_mp(verts, xout, H), lambda: tuple(cover.cover_vector_np(np.array(xout), verts, H))):
                v = fn(); ov = oracle_grad(verts, xout, H)
                assert _cmp(v, ov) <= 1e-12 and _cmp(v, (0.0, 0.0)) <= 1e-12, name


def test_scaling_with_H():
    """the result scales as H^2 under a uniform scaling of the geometry (and H)."""
    for name, verts in POLYGONS.items():
        base = cover.cover_gradient_mp(verts, (0.3, 0.4), 1.0)
        for lam in (0.5, 2.0):
            poly = [(v[0] * lam, v[1] * lam) for v in verts]
            x = (0.3 * lam, 0.4 * lam)
            v = cover.cover_gradient_mp(poly, x, 1.0 * lam)
            assert _cmp(v, (lam * lam * float(base[0]), lam * lam * float(base[1]))) <= TOL * lam * lam, name


# --------------------------------------------------------------------------- (i) negative control: discontinuous kernel
def _edge_only_mp(verts, x, H, dps=40):
    """the edge-only formula for the DISCONTINUOUS kernel r 1[r <= H]: -sum_e n_e int_chord r ds (misses the circle term)."""
    with mp.workdps(dps + GUARD):
        P = G.prepare(verts, x, h=1)
        g = block_grad(P, 1, H)
        return (float(g[0]), float(g[1]))


def _true_grad_discontinuous(verts, x, H, eps=1e-6, dps=40):
    """true grad_x int_T r 1[r <= H] dA by a central difference of the oracle VALUE (g = r on (0, H))."""
    def V(px, py):
        return float(oracle.polar_moment_profile(verts, (px, py), (0, 0), [(0.0, H, {1: 1})], h=1.0, dps=dps))
    gx = (V(x[0] + eps, x[1]) - V(x[0] - eps, x[1])) / (2.0 * eps)
    gy = (V(x[0], x[1] + eps) - V(x[0], x[1] - eps)) / (2.0 * eps)
    return (gx, gy)


def test_negative_control_discontinuous_kernel():
    """Negative control: the edge-only formula for the DISCONTINUOUS kernel r 1[r <= H]
    must NOT reproduce the correct continuous-K gradient; the gap is the 'circle term'
    (= -H * block_grad(P, 0, H), i.e. the -H*1[r<=H] part of K = r - H), magnitude > 1e-3.

    FINDING (also in LOG-001 / REPORT-001): the work order's literal wording asks for
    'edge-only vs the true gradient of r 1[r <= H] by a central difference of the oracle value'.
    That central difference converges to the edge-only formula itself (the distributional
    gradient of r 1[r<=H] IS -sum n_e int_chord r ds), so that reading gives a mismatch ~0
    and can never exceed 1e-3.  The reviewer's smoke reference instead pins the 'circle term'
    to (0.5929, 0) - (-0.4071, 0) = (1.0, 0) at x=(1.3, 0.5), i.e. continuous-K gradient minus
    the discontinuous-r edge-only.  This test implements that reading (the only one that is
    > 1e-3 and matches the given number) and separately records the central-difference result.
    """
    worst = 0.0
    worst_at = None
    pts = [(1.3, 0.5), (0.3, 0.4), (0.0, 0.5), (0.5, 1.3), (2.0, 0.2)]
    for p in pts:
        cont = cover.cover_gradient_mp(UNIT_SQ, p, 1.0)              # correct continuous-K gradient
        eo = _edge_only_mp(UNIT_SQ, p, 1.0)                          # discontinuous-r edge-only
        d = _cmp((float(cont[0]), float(cont[1])), eo)
        if d > worst:
            worst, worst_at = d, p
    assert worst > 1e-3, "circle term (cont K - disc-r edge-only) never exceeded 1e-3 (worst %.3e at %s)" % (worst, worst_at)
    # sanity: the reviewer's smoke values at x = (1.3, 0.5)
    cont = cover.cover_gradient_mp(UNIT_SQ, (1.3, 0.5), 1.0)          # continuous K -> (0.5929, 0)
    eo = _edge_only_mp(UNIT_SQ, (1.3, 0.5), 1.0)                     # discontinuous edge-only -> (-0.4071, 0)
    assert abs(float(cont[0]) - 0.5929) < 1e-3 and abs(float(cont[1])) < 1e-3
    assert abs(float(eo[0]) - (-0.4071)) < 1e-3 and abs(float(eo[1])) < 1e-3
    # the gap is exactly the -H*block_grad(P,0,H) term (the -H*1[r<=H] part of K = r - H)
    with mp.workdps(60):
        P = G.prepare(UNIT_SQ, (1.3, 0.5), h=1)
        g0 = block_grad(P, 0, 1.0)
    term = (-1.0 * float(g0[0]), -1.0 * float(g0[1]))
    got = (float(cont[0]) - eo[0], float(cont[1]) - eo[1])
    assert _cmp(got, term) < 1e-9, "circle term != -H*block_grad(0,H): got %s term %s" % (got, term)
    # recorded finding: the literal central-difference reading gives ~0 (true grad of disc-r IS edge-only)
    cd = _true_grad_discontinuous(UNIT_SQ, (1.3, 0.5), 1.0)
    assert _cmp(eo, cd) < 1e-2, "central-diff true grad of disc-r should equal edge-only (finding)"


# --------------------------------------------------------------------------- (ii) flat floor sanity check
def _halfplane_analytic(H, d):
    """grad_x int_{y'<0} K dA for a particle at (0, d): (0, H^2 cos(phi0) - d^2 ln(cot(phi0/2))), phi0 = asin(d/H).
    Derived: grad = int_{u2>d, |u|<H} uhat dA; x-comp 0 by symmetry; y-comp = H^2 cos(phi0) - d^2 ln(cot(phi0/2))."""
    phi0 = math.asin(d / H)
    return (0.0, H * H * math.cos(phi0) - d * d * math.log(1.0 / math.tan(0.5 * phi0)))


@pytest.mark.parametrize("dfrac", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_flat_floor_points_up_and_matches_analytic(dfrac):
    H = 1.0
    d = dfrac * H
    solid = [(-5.0 * H, -5.0 * H), (5.0 * H, -5.0 * H), (5.0 * H, 0.0), (-5.0 * H, 0.0)]     # disk of radius H around (0,d) never reaches the other edges
    x = (0.0, d)
    npv = cover.cover_vector_np(np.asarray(x, dtype=np.float64), solid, H)
    mpv = cover.cover_gradient_mp(solid, x, H)
    ov = oracle_grad(solid, x, H)
    ana = _halfplane_analytic(H, d)
    # points UP (away from the wall) and has no x-component
    assert npv[1] > 0 and abs(npv[0]) < 1e-9 * H * H
    for v, ref, tag in ((npv, ana, "np vs analytic"), (mpv, ana, "mp vs analytic"), (ov, ana, "oracle vs analytic"), (npv, ov, "np vs oracle")):
        assert _cmp(v, ref) <= 1e-9 * H * H, "%s: %.3e (d/H=%.2f)" % (tag, _cmp(v, ref) / (H * H), dfrac)
