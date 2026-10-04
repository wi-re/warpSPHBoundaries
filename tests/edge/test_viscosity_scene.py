"""Tests for viscosity.lap_lambda_scene (the exact wall Laplacian  Delta-lambda = int_solid lap W dA', Q1, Wendland C2 and C4).

Delta-lambda is evaluated through the unmodified Density and Covariance scene operations on the registered ordinary kernel
L = W'/r (the kernel `lw2` / `lw4`):  Delta-lambda = f (2 lambda[L] - tr Cov[L]),  f = P c / (C_l H^2).  Sign: positive for a
fluid-side particle (above a floor, solid below); 0 for a particle with the support fully inside the solid and beyond H.

Every reference below is own plain numpy / scipy code (nothing imported from the reviewer's probes): the normalised Wendland
kernel written out with analytic derivatives by the product rule (verified against 5-point finite differences, rel <= 1e-9),
scipy quad for the flat wall, and a polar midpoint brute force with an even-odd point-in-polygon test (no edge formulas, no
winding numbers).

Tolerances (stated with their reason, BEFORE looking at the results):
  * (a)  <= 1e-10 * max B vs the own scipy quad of B(z) = int_z^H (W'' + W'/r) 2 Theta(r) r dr (float64 round-off of the
    degree-<= 6 monomial plan, measured <= 4e-14 by the reviewer; a wrong factor or sign is >= 1e-2); the smoke B(z) of the
    work document at H = 1 (rtol 1e-6); all values > 0.
  * (b)  <= 1e-3 * max|brute| vs the own 500 x 1000 polar midpoint brute force (the grid's own error, measured 2.2e-4 / 2.1e-4
    by the reviewer; a formula error is >= 1e-2 * max, see the controls (f)); max|brute| > 10 and >= 50 points with
    |brute| > 1e-6 (not two arrays of zeros).
  * (c)  <= 1e-9 absolute for the degenerate exact zeros (tangent, fully inside, farther than H, on the edge, at the vertex);
    the outside point (1.2, 0.5), H = 0.3, is non-zero and <= 1e-3 * value vs the brute force.
  * (d)  scaling H^-2 (rtol 1e-9); moved body vs a fresh scene at the new pose (<= 1e-11 * max); two-body rows and column sum
    (<= 1e-11 * max).
  * (f)  negative controls, each must FAIL the comparison tolerances (numbers printed): 2 f lambda alone (trace dropped) > 1e-1
    * max|brute| (reviewer 5.76 / 5.19); -Delta-lambda > 1.0 * max|brute| (reviewer 2.00); the pairwise coefficient A(z) in
    place of B(z) at z/H = 0.1 differs from B by > 0.3 * B (reviewer |A| = 3.053 vs B = 1.960 for C2).
  * (g)  the moment identity int r W' dA = -2 (atol 1e-10) and the pairwise bulk term for v = (y^2, 0) = fac/4 (rtol 1e-10,
    fac = 1) -- pins nu_eff = fac/8.
"""
import math

import numpy as np
import pytest
import torch
import warp as wp
from scipy import integrate

from edgebound.scene import Body, ImplicitRep, Scene, SurfaceRep, VolumeRep
from edgebound.viscosity import lap_factor, lap_lambda_scene

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64
LS = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
CENTER, ANGLE, H = (0.3, -0.2), 0.7, 0.6
TRI30 = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]], dtype=float)
LV = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2]], dtype=float)
LE = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6]])
FLOOR = [(-5.0, -2.0), (5.0, -2.0), (5.0, 0.0), (-5.0, 0.0)]     # counter-clockwise, solid below y = 0
UNIT = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def world_polygon(local, center, angle):
    """the Body's local polygon in world coordinates, plain numpy (R(angle) p + center, as Pose.toWorld)."""
    return local @ rot(angle).T + np.asarray(center, dtype=float)


def kern(fam):
    """the normalised Wendland kernel W = c/(pi H^2) s(r/H) (c = 7 / 9) and the analytic derivatives by the product rule
    (verified against 5-point finite differences, rel <= 1e-9); returns W, dW/dr, lap W = W'' + W'/r as functions of (r, H)."""
    if fam == "w2":
        c = 7.0
        s = lambda q: (1 - q) ** 4 * (1 + 4 * q)
        sp = lambda q: -4 * (1 - q) ** 3 * (1 + 4 * q) + 4 * (1 - q) ** 4
        spp = lambda q: 12 * (1 - q) ** 2 * (1 + 4 * q) - 32 * (1 - q) ** 3
    else:
        c = 9.0
        s = lambda q: (1 - q) ** 6 * (1 + 6 * q + 35.0 / 3.0 * q * q)
        def sp(q):
            f = 35.0 / 3.0 * q * q + 6 * q + 1
            return -6 * (1 - q) ** 5 * f + (1 - q) ** 6 * (70.0 / 3.0 * q + 6)
        def spp(q):
            f = 35.0 / 3.0 * q * q + 6 * q + 1
            return 30 * (1 - q) ** 4 * f - 12 * (1 - q) ** 5 * (70.0 / 3.0 * q + 6) + (1 - q) ** 6 * 70.0 / 3.0
    W = lambda r, h: np.where(r <= h, c / (math.pi * h * h) * s(np.clip(r / h, 0.0, 1.0)), 0.0)
    dW = lambda r, h: np.where(r <= h, c / (math.pi * h ** 3) * sp(np.clip(r / h, 0.0, 1.0)), 0.0)
    lap = lambda r, h: np.where(r <= h, c / (math.pi * h ** 4) * (spp(np.clip(r / h, 0.0, 1.0)) + sp(np.clip(r / h, 0.0, 1.0)) / np.clip(r / h, 1e-300, 1.0)), 0.0)
    return W, dW, lap


def B_quad(fam, z, H):
    """B(z; H) = int_solid lap W dA' for the flat wall (solid below y = 0), particle at (0, z):  int_z^H (W'' + W'/r) 2 Theta(r) r dr,
    Theta = arccos(z/r) (own scipy quad, analytic derivatives)."""
    _, _, lap = kern(fam)
    def integrand(r):
        th = math.acos(min(z / r, 1.0))
        return float(lap(r, H)) * 2.0 * th * r
    return integrate.quad(integrand, z, H, epsabs=1e-13, epsrel=1e-13, limit=200)[0]


def A_quad(fam, z, H):
    """the pairwise (Monaghan, free-slip mirror) wall coefficient  A(z; H) = int_z^H W'(r) (Theta + sin Theta cos Theta) dr,
    Theta = arccos(z/r)  (own scipy quad; A < 0)."""
    _, dW, _ = kern(fam)
    def integrand(r):
        th = math.acos(min(z / r, 1.0))
        return float(dW(r, H)) * (th + math.sin(th) * math.cos(th))
    return integrate.quad(integrand, z, H, epsabs=1e-13, epsrel=1e-13, limit=200)[0]


def inside_eo(P, poly):
    """even-odd point-in-polygon, plain numpy (P [M,2], poly [V,2]); True inside."""
    x, y = P[:, 0], P[:, 1]
    ins = np.zeros(len(P), dtype=bool)
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        cr = ((a[1] > y) != (b[1] > y)) & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1] + 1e-300) + a[0])
        ins ^= cr
    return ins


def lap_brute(fam, p, H, poly, solid_inside=True, nr=500, nt=1000):
    """brute-force  int_solid lap W dA'  for each particle of p [N,2]: a polar midpoint grid nr x nt around each particle
    (no edge formulas, no winding numbers); the solid mask is the even-odd point-in-polygon test (solid = inside the polygon,
    outside for a cavity)."""
    _, _, lap = kern(fam)
    r = (np.arange(nr) + 0.5) / nr * H
    th = (np.arange(nt) + 0.5) / nt * 2.0 * math.pi
    R, T = np.meshgrid(r, th, indexing="ij")
    Rr, cT, sT = R.ravel(), np.cos(T).ravel(), np.sin(T).ravel()
    lapr = lap(R, H).ravel() * Rr
    dA = (H / nr) * (2.0 * math.pi / nt)
    out = np.zeros(len(p))
    for i, (px, py) in enumerate(p):
        P = np.stack([px + Rr * cT, py + Rr * sT], 1)
        mask = inside_eo(P, poly) if solid_inside else ~inside_eo(P, poly)
        out[i] = lapr[mask].sum() * dA
    return out


def _lam_l(scene, pts, H, fam, device):
    """the Density of the registered kernel L = W'/r (kernel `l` + family), per body [B, N] (for the negative control 2 f lambda)."""
    from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
    from edgebound.scene import BodyField, sceneOperation
    lap_factor(H, fam)                                              # idempotent registration
    pos = torch.as_tensor(np.asarray(pts), dtype=F64, device=device)
    n = len(pos)
    ps = ParticleState(positions=pos, supports=torch.full((n,), float(H), dtype=F64, device=device),
                       masses=torch.ones(n, dtype=F64, device=device), kinds=torch.zeros(n, dtype=torch.int32, device=device),
                       densities=torch.ones(n, dtype=F64, device=device))
    pr = OperationProperties(kernel="l" + fam, operation=WarpOperation.Density, gradientMode=GradientScheme.Naive,
                             operationMode=OperationDirection.BoundaryToFluid)
    return sceneOperation(ps, pr, scene, None, None, [BodyField(rho=1.0)] * len(scene.bodies), perBody=True).reshape(len(scene.bodies), n)


@pytest.mark.parametrize("device", DEVICES)
def test_flat_wall_matches_quad_and_smoke(device):
    """(a) the flat floor (solid below y = 0): the scene result vs the own scipy quad of B(z) (max|diff| <= 1e-10 max B, both
    families, H in {1, 0.7}, z/H in {0.02, ..., 0.9}); the smoke B(z) of the work document at H = 1 (rtol 1e-6); all > 0."""
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])
    sc = Scene([body], device)
    zs = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9]
    smoke = {
        "w2": {0.02: 0.4417774, 0.1: 1.959941, 0.3: 3.190942, 0.5: 1.943131, 0.7: 0.5164654, 0.9: 0.01570062},
        "w4": {0.02: 0.6086062, 0.1: 2.765873, 0.3: 4.083071, 0.5: 1.708461, 0.7: 0.2060951, 0.9: 8.409243e-4},
    }
    worst = 0.0
    for fam in ("w2", "w4"):
        for Hh in (1.0, 0.7):
            pts = np.array([[0.0, z * Hh] for z in zs])
            got = lap_lambda_scene(sc, pts, Hh, fam).cpu().numpy().ravel()
            ref = np.array([B_quad(fam, z * Hh, Hh) for z in zs])
            scale = float(ref.max())
            err = float(np.abs(got - ref).max())
            worst = max(worst, err / scale)
            assert err <= 1e-10 * scale, (fam, Hh, err, scale)
            assert (got > 0).all(), (fam, Hh, got)
            if Hh == 1.0:
                for i, z in enumerate(zs):
                    assert abs(got[i] - smoke[fam][z]) <= 1e-6 * smoke[fam][z], (fam, z, got[i], smoke[fam][z])
    print("(a) flat wall: worst |scene - quad| / max B = %.3e over both families and H in {1, 0.7} (tol 1e-10)" % worst)


@pytest.mark.parametrize("device", DEVICES)
def test_l_body_and_cavity_match_brute(device):
    """(b) the rotated/translated L body (center (0.3, -0.2), angle 0.7, H = 0.6, 200 pts seed 3 in [-2, 3]^2) and the cavity
    (solid='outside', H = 0.5, 40 pts seed 5 in [-0.1, 2.1] x [-0.1, 1.1]): the scene result vs the own 500 x 1000 polar midpoint
    brute force (max|scene - brute| <= 1e-3 max|brute|, the grid's own error); max|brute| > 10 and >= 50 non-zero points on the L body."""
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
    sc = Scene([body], device)
    w = world_polygon(LS, CENTER, ANGLE)
    pts = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    worst = {}
    for fam in ("w2", "w4"):
        got = lap_lambda_scene(sc, pts, H, fam).cpu().numpy().ravel()
        ref = lap_brute(fam, pts, H, w)
        s = float(np.abs(ref).max())
        assert s > 10.0, (fam, s)
        assert int((np.abs(ref) > 1e-6).sum()) >= 50, (fam, int((np.abs(ref) > 1e-6).sum()))
        err = float(np.abs(got - ref).max())
        worst["L " + fam] = err / s
        assert err <= 1e-3 * s, (fam, err, s)
    BOX = [(0, 0), (2, 0), (2, 1), (0, 1)]
    sc2 = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(BOX, solid="outside")])], device)
    pts2 = np.random.default_rng(5).uniform([-0.1, -0.1], [2.1, 1.1], (40, 2))
    for fam in ("w2", "w4"):
        got = lap_lambda_scene(sc2, pts2, 0.5, fam).cpu().numpy().ravel()
        ref = lap_brute(fam, pts2, 0.5, np.array(BOX, dtype=float), solid_inside=False)
        s = float(np.abs(ref).max())
        assert s > 0.1, (fam, s)                    # sanity: not two arrays of zeros
        err = float(np.abs(got - ref).max())
        worst["cavity " + fam] = err / s
        assert err <= 1e-3 * s, (fam, err, s)
    print("(b) worst |scene - brute| / max|brute| (tol 1e-3): " + "  ".join("%s %.2e" % (k, v) for k, v in worst.items()))


@pytest.mark.parametrize("device", DEVICES)
def test_degenerate_cases(device):
    """(c) the unit square: exact zeros (<= 1e-9 absolute) at the centre with H = 0.5 (tangent z == H: the support disk is the
    inscribed circle, a full disk -> int lap W = 0), the same particle with H = 0.4 (support fully inside the body: the indicator
    trap case, a directly registered lap W would give c/H^2 there), farther than H from every edge and outside, on the edge
    (0.5, 0) and at the vertex (0, 0) with H = 0.3; the outside particle (1.2, 0.5), H = 0.3, is non-zero (<= 1e-3 value vs the
    brute force; reviewer 7.754195 C2 / 3.688068 C4, context only)."""
    sc = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(UNIT)])], device)
    unit = np.array(UNIT, dtype=float)
    zeros = [((0.5, 0.5), 0.5), ((0.5, 0.5), 0.4), ((2.5, 2.5), 0.5), ((0.5, 0.0), 0.3), ((0.0, 0.0), 0.3)]
    for fam in ("w2", "w4"):
        for (px, py), Hh in zeros:
            val = float(lap_lambda_scene(sc, np.array([[px, py]]), Hh, fam).cpu().numpy().ravel()[0])
            assert abs(val) <= 1e-9, (fam, px, py, Hh, val)
        got = float(lap_lambda_scene(sc, np.array([[1.2, 0.5]]), 0.3, fam).cpu().numpy().ravel()[0])
        ref = float(lap_brute(fam, np.array([[1.2, 0.5]]), 0.3, unit)[0])
        assert abs(got) > 0.0
        assert abs(got - ref) <= 1e-3 * abs(ref), (fam, got, ref)
        print("(c) %s: zeros <= 1e-9 (5 cases); outside (1.2, 0.5) H = 0.3: scene %.6f  brute %.6f (|diff|/value %.1e)" %
              (fam, got, ref, abs(got - ref) / abs(ref)))


@pytest.mark.parametrize("device", DEVICES)
def test_scaling_and_moving_body(device):
    """(d) Delta-lambda(z; H) = Delta-lambda(z/H; 1)/H^2 (H = 0.7 vs H = 1, rtol 1e-9, flat floor, z/H = 0.3); after
    body.center = ... without rebuilding the scene the result equals a scene built fresh at the new pose (<= 1e-11 max, L body,
    moved by (0.4, -0.3)); two bodies (box at the origin, triangle at (3, 0) as in test_cover_scene.py, H = 1): row b of the
    [B, N] result is the single-body result of body b and the column sum is the sum of the single-body results (<= 1e-11 max)."""
    for fam in ("w2", "w4"):
        # scaling
        scf = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])], device)
        dl1 = float(lap_lambda_scene(scf, np.array([[0.0, 0.3]]), 1.0, fam).cpu().numpy().ravel()[0])
        dl07 = float(lap_lambda_scene(scf, np.array([[0.0, 0.21]]), 0.7, fam).cpu().numpy().ravel()[0])
        assert abs(dl07 - dl1 / 0.7 ** 2) <= 1e-9 * max(abs(dl07), abs(dl1 / 0.7 ** 2)), (fam, dl07, dl1)
        # moving body
        body = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
        sc = Scene([body], device)
        pts = np.random.default_rng(3).uniform(-2, 3, (200, 2))
        body.center = body.center + torch.tensor([0.4, -0.3], dtype=F64, device=device)
        got = lap_lambda_scene(sc, pts, H, fam).cpu().numpy()
        fresh = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=(CENTER[0] + 0.4, CENTER[1] - 0.3), angle=ANGLE)], device)
        ref = lap_lambda_scene(fresh, pts, H, fam).cpu().numpy()
        s = float(np.abs(ref).max())
        err = float(np.abs(got - ref).max())
        assert err <= 1e-11 * s, (fam, err, s)
        # two bodies: box at the origin + triangle at (3, 0) (test_cover_scene.py), H = 1
        b1 = Body(bodyId=0, reps=[SurfaceRep.polygon(UNIT)])
        b2 = Body(bodyId=1, reps=[SurfaceRep.polygon(TRI30)], center=(3.0, 0.0), angle=math.pi / 6)
        sc2 = Scene([b1, b2], device)
        pos = np.random.default_rng(7).uniform(-1, 4.2, (200, 2))
        got2 = lap_lambda_scene(sc2, pos, 1.0, fam).cpu().numpy()          # [2, 200]
        sc1 = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(UNIT)])], device)
        sc3 = Scene([Body(bodyId=1, reps=[SurfaceRep.polygon(TRI30)], center=(3.0, 0.0), angle=math.pi / 6)], device)
        g1 = lap_lambda_scene(sc1, pos, 1.0, fam).cpu().numpy()[0]
        g3 = lap_lambda_scene(sc3, pos, 1.0, fam).cpu().numpy()[0]
        s2 = float(max(np.abs(g1).max(), np.abs(g3).max()))
        assert s2 > 0.0
        e_box = float(np.abs(got2[0] - g1).max())
        e_tri = float(np.abs(got2[1] - g3).max())
        e_sum = float(np.abs(got2.sum(0) - (g1 + g3)).max())
        assert e_box <= 1e-11 * s2, (fam, e_box, s2)
        assert e_tri <= 1e-11 * s2, (fam, e_tri, s2)
        assert e_sum <= 1e-11 * s2, (fam, e_sum, s2)
        print("(d) %s: scaling |dl07 - dl1/0.49| = %.1e; moved body %.1e; two-body rows/sum %.1e / %.1e / %.1e (scale %.3e, tol 1e-11)"
              % (fam, abs(dl07 - dl1 / 0.7 ** 2), err, e_box, e_tri, e_sum, s2))


@pytest.mark.parametrize("device", DEVICES)
def test_guards(device):
    """(e) family='w9', an ImplicitRep body (DiskBody) and a VolumeRep body raise NotImplementedError before computing anything
    (messages as in the spec)."""
    from edgebound.implicitBodies import DiskBody
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])
    sc = Scene([body], device)
    pts = np.array([[0.0, 0.3]])
    with pytest.raises(NotImplementedError, match="Wendland C2 and C4 only"):
        lap_lambda_scene(sc, pts, 1.0, family="w9")
    si = Scene([Body(bodyId=0, reps=[ImplicitRep(DiskBody(center=(0.5, 0.5), radius=0.5))])], device)
    with pytest.raises(NotImplementedError, match="SurfaceRep bodies only"):
        lap_lambda_scene(si, pts, 1.0)
    sv = Scene([Body(bodyId=0, reps=[VolumeRep(LV, LE)])], device)
    with pytest.raises(NotImplementedError, match="SurfaceRep bodies only"):
        lap_lambda_scene(sv, pts, 1.0)
    print("(e) guards: family='w9', ImplicitRep (DiskBody) and VolumeRep bodies all raise NotImplementedError")


@pytest.mark.parametrize("device", DEVICES)
def test_negative_controls(device):
    """(f) negative controls on the L body (each must FAIL the (b) tolerance; numbers printed, relative to max|brute|): 2 f lambda
    alone (trace dropped, reviewer 5.76 / 5.19) > 1e-1; -Delta-lambda (reviewer 2.00) > 1.0; on the flat floor, the pairwise
    coefficient A(z) in place of B(z) at z/H = 0.1 differs from B by > 0.3 B (reviewer |A| = 3.053 vs B = 1.960 for C2)."""
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
    sc = Scene([body], device)
    w = world_polygon(LS, CENTER, ANGLE)
    pts = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    scf = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])], device)
    out = []
    for fam in ("w2", "w4"):
        got = lap_lambda_scene(sc, pts, H, fam).cpu().numpy().ravel()
        ref = lap_brute(fam, pts, H, w)
        s = float(np.abs(ref).max())
        assert float(np.abs(got - ref).max()) <= 1e-3 * s        # the real result agrees (the controls below are only meaningful if it does)
        lam = _lam_l(sc, pts, H, fam, device).cpu().numpy()[0]
        f = lap_factor(H, fam)
        d_trace = float(np.abs(f * (2.0 * lam) - ref).max()) / s
        d_neg = float(np.abs(-got - ref).max()) / s
        assert d_trace > 1e-1, (fam, d_trace)
        assert d_neg > 1.0, (fam, d_neg)
        A = A_quad(fam, 0.1, 1.0)
        B = B_quad(fam, 0.1, 1.0)
        d_ab = abs(A - B) / abs(B)
        assert d_ab > 0.3, (fam, A, B)
        out.append((fam, d_trace, d_neg, A, B, d_ab))
    for fam, d_trace, d_neg, A, B, d_ab in out:
        print("(f) %s: 2f*lambda only %.3f (tol 1e-1)   -Delta-lambda %.3f (tol 1.0)   A(0.1) = %.4f vs B(0.1) = %.4f, |A-B|/B = %.3f (tol 0.3)"
              % (fam, d_trace, d_neg, A, B, d_ab))


def test_nu_eff_moment_identity():
    """(g) the moment identity that pins nu_eff = fac/8 (plain quadrature, no scene): for both families  int r W' dA =
    2 pi int_0^H r^2 W'(r) dr = -2 (atol 1e-10) -- the only surviving second-order moment of the pairwise expansion -- and for
    v = (y^2, 0) (lap v = (2, 0), div v = 0) the pairwise bulk term  a_x = fac (-int W'/r^3 x^2 y^2 dA) = fac (-(pi/4) int_0^H r^2 W' dr)
    = fac/4 = (fac/8) |lap v| (rtol 1e-10, fac = 1)."""
    for fam in ("w2", "w4"):
        _, dW, _ = kern(fam)
        Hh = 1.0
        I = 2.0 * math.pi * integrate.quad(lambda r: r * r * float(dW(r, Hh)), 0.0, Hh, epsabs=1e-14, epsrel=1e-14)[0]
        assert abs(I - (-2.0)) <= 1e-10, (fam, I)
        a_x = -(math.pi / 4.0) * (I / (2.0 * math.pi))
        assert abs(a_x - 0.25) <= 1e-10 * 0.25, (fam, a_x)
        print("(g) %s: int r W' dA = %.12f (expect -2, atol 1e-10)   pairwise bulk term for v = (y^2, 0) = %.12f fac = fac/4 = (fac/8) |lap v| (rtol 1e-10)" % (fam, I, a_x))


@pytest.mark.parametrize("device", DEVICES)
def test_one_adjacency_per_call(device):
    """(h) lap_lambda_scene builds EXACTLY ONE adjacency per call (one kernel, one adjacency, passed to both the Density and the
    Covariance operation) and the result equals the two-adjacency route written out with sceneOperation directly (each operation
    without an adjacency argument, f (2 lambda - tr Cov) with lap_factor): max|diff| <= 1e-12 max|result| (same pairs, same
    arithmetic, summation noise 1e-16; a wrongly shared adjacency is >= 1e-1)."""
    from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
    from edgebound.scene import BodyField, sceneOperation
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
    sc = Scene([body], device)
    pts = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    pos = torch.as_tensor(pts, dtype=F64, device=device)
    n = len(pos)
    B = len(sc.bodies)
    ps = ParticleState(positions=pos, supports=torch.full((n,), float(H), dtype=F64, device=device),
                       masses=torch.ones(n, dtype=F64, device=device), kinds=torch.zeros(n, dtype=torch.int32, device=device),
                       densities=torch.ones(n, dtype=F64, device=device))
    orig = Scene.buildAdjacency
    counts = {"n": 0}
    def counting(self, queryParticles, operationProperties):
        counts["n"] += 1
        return orig(self, queryParticles, operationProperties)
    Scene.buildAdjacency = counting
    try:
        for fam in ("w2", "w4"):
            counts["n"] = 0
            got = lap_lambda_scene(sc, pts, H, fam).cpu().numpy()
            n_adj = counts["n"]
            assert n_adj == 1, (fam, n_adj)
            # the two-adjacency route: Density and Covariance each build their own adjacency (sceneOperation without an adjacency argument)
            lam = sceneOperation(ps, OperationProperties(kernel="l" + fam, operation=WarpOperation.Density,
                                                         gradientMode=GradientScheme.Naive,
                                                         operationMode=OperationDirection.BoundaryToFluid),
                                 sc, None, None, [BodyField(rho=1.0)] * B, perBody=True).reshape(B, n)
            cov = sceneOperation(ps, OperationProperties(kernel="l" + fam, operation=WarpOperation.Covariance,
                                                         gradientMode=GradientScheme.Naive,
                                                         operationMode=OperationDirection.BoundaryToFluid),
                                 sc, None, None, [BodyField(rho=1.0)] * B, perBody=True).reshape(B, n, 2, 2)
            ref = (lap_factor(H, fam) * (2.0 * lam - cov[:, :, 0, 0] - cov[:, :, 1, 1])).cpu().numpy()
            s = float(np.abs(got).max())
            err = float(np.abs(got - ref).max())
            assert err <= 1e-12 * s, (fam, err, s)
            print("(h) %s: one lap_lambda_scene call builds %d adjacency (expect 1); two-adjacency route max|diff| = %.2e (tol 1e-12 * %.3e)" % (fam, n_adj, err, s))
    finally:
        Scene.buildAdjacency = orig
