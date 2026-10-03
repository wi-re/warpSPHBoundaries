"""Tests for tensile.tensile_vector_scene with the Wendland C4 family (WORK-004 T4.2).

The exact wall part  T = int_solid W^4 grad W dA' = (1/5) grad int_solid W^5 dA'  for the Wendland C4 kernel
(the degree-40 kernel `w4p5`, routed through the Chebyshev-quadrature plan).  Mirrors test_tensile_scene.py
(Wendland C2); the helpers are written out again for W = Wendland C4:
    W = 9/(pi H^2) (1-q)^6 (1 + 6q + 35/3 q^2),   q = r/H,   9 = KERNELS["w4"].c2_pi.

tolerances (fixed in the WORK-004 T4.2 spec, stated BEFORE looking at the results):
  * (a)  <= 1e-10 * max|T|:  the scene route vs  tensile_factor(H, family) * np2d.gradient(..., family+"p5",
    stable=(16, 8))  (the SAME algorithm in two independent implementations; the series/quadrature vertex limit is
    2.7e-12, so 1e-10 excludes the monomial route -- see (c)).  Both families.
  * (b)  <= 5e-5 relative vs the plain-numpy 2000 x 2000 midpoint grid of W^4 grad W (the integral's own quadrature,
    the dominant term; measured 6.13e-6 for both points); the smoke T_y (rtol 1e-6) are the reviewer's references and
    T_x = 0 (atol 1e-6 |T_y|) by the floor's symmetry; sign T_y < 0.
  * (c)  negative controls compared with the GRID value of (b): 1.01*T and -T differ by > 1e-3 relative; the plain
    monomial route for the L-shape of (a) (np2d.gradient without stable, same factor) differs from the stable result
    by > 1e-4 * max|T| (measured 2.2e-3).
"""
import math

import numpy as np
import pytest
import torch
import warp as wp

from edgebound.scene import Body, ImplicitRep, Scene, SurfaceRep, VolumeRep
from edgebound.tensile import tensile_factor, tensile_vector_scene

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
LS = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
CENTER, ANGLE, H = (0.3, -0.2), 0.7, 0.6
LV = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2]], dtype=float)
LE = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6]])
FLOOR = [(-2.0, -2.0), (2.0, -2.0), (2.0, 0.0), (-2.0, 0.0)]
C4 = 9.0  # KERNELS["w4"].c2_pi


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def world_polygon(local, center, angle):
    """the Body's local polygon in world coordinates, plain numpy (R(angle) p + center, as Pose.toWorld)."""
    return local @ rot(angle).T + np.asarray(center, dtype=float)


def dense_T(px, py, H, n=2000):
    """the integral's own quadrature for Wendland C4: W^4 grad W over the solid (x' in [-H,H], y' in [-H,0], r < H).
    W = 9/(pi H^2) (1-q)^6 (1 + 6q + 35/3 q^2),  q = r/H,
    dW/dq = 9/(pi H^2)(-6(1-q)^5 (1 + 6q + 35/3 q^2) + (1-q)^6 (6 + 70/3 q)),   grad W = (dW/dq)/H * (x-x')/r."""
    xp = -H + (np.arange(n) + 0.5) / n * (2 * H)
    yp = -H + (np.arange(n) + 0.5) / n * (H)
    X, Y = np.meshgrid(xp, yp, indexing="ij")
    rx = px - X
    ry = py - Y
    r = np.sqrt(rx * rx + ry * ry)
    m = r < H
    q = r[m] / H
    c = C4 / (math.pi * H * H)
    P = 1.0 + 6.0 * q + (35.0 / 3.0) * q * q
    W = c * (1.0 - q) ** 6 * P
    dWdq = c * (-6.0 * (1.0 - q) ** 5 * P + (1.0 - q) ** 6 * (6.0 + (70.0 / 3.0) * q))
    gWx = (dWdq / H) * (rx[m] / r[m])
    gWy = (dWdq / H) * (ry[m] / r[m])
    dA = (2 * H / n) * (H / n)
    return np.array([np.sum(W ** 4 * gWx), np.sum(W ** 4 * gWy)]) * dA


@pytest.mark.parametrize("device", DEVICES)
def test_l_shape_matches_numpy_w4p5_gradient(device):
    """(a) the rotated, translated L-shape (center (0.3,-0.2), angle 0.7, H = 0.6, 200 pts seed 3): the scene route vs
    factor * np2d.gradient(family+"p5", stable=(16,8)) (<= 1e-10 max|T|), both families."""
    from edgebound import np2d
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
    sc = Scene([body], device)
    pos = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    w = world_polygon(LS, CENTER, ANGLE)
    verts_rep = np.repeat(w[None], 200, axis=0)
    for family, kname in (("w4", "w4p5"), ("w2", "w2p5")):
        T = tensile_vector_scene(sc, pos, H, family=family).cpu().numpy()
        ref = tensile_factor(H, family) * np2d.gradient(verts_rep, pos, kname, h=H, dtype=np.float64, stable=(16, 8))
        scale = float(np.abs(T).max())
        assert scale > 0.0
        err = float(np.abs(T - ref).max())
        assert err <= 1e-10 * scale, (family, err, scale)
        print(f"(a) {family}: scene vs factor*np2d.gradient({kname}, stable) max|diff| {err / scale:.2e} (<= 1e-10)  scale {scale:.3e}")


@pytest.mark.parametrize("device", DEVICES)
def test_flat_floor_matches_dense_grid_and_smoke(device):
    """(b) the half-plane (solid below y = 0): the scene route vs the 2000 x 2000 midpoint grid (rel <= 5e-5); the smoke
    T_y (rtol 1e-6), T_x = 0 (atol 1e-6 |T_y|) and sign T_y < 0 on the scene result by the floor's symmetry."""
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])
    sc = Scene([body], device)
    for (px, py, Hh, smokeTy) in [(0.0, 0.3, 1.0, -0.1978237590), (0.0, 0.15, 0.5, -101.2857646)]:
        T = tensile_vector_scene(sc, np.array([[px, py]]), Hh, family="w4").cpu().numpy()[0]
        grid = dense_T(px, py, Hh)
        # scene vs the grid (the grid's quadrature error is the dominant term, ~6.13e-6)
        assert float(np.abs(T - grid).max()) <= 5e-5 * float(np.abs(grid).max()), (T, grid)
        # smoke: T_y (rtol 1e-6); T_x = 0 (atol 1e-6 |T_y|); sign T_y < 0
        assert abs(T[1] - smokeTy) <= 1e-6 * abs(smokeTy), (T[1], smokeTy)
        assert abs(T[0]) <= 1e-6 * abs(T[1])
        assert T[1] < 0.0
        print(f"(b) floor ({px},{py},H={Hh}): T_y {T[1]:.10f} (smoke {smokeTy}), grid {grid[1]:.10f}, |T-grid|/max|grid| {np.abs(T - grid).max() / np.abs(grid).max():.2e}")


@pytest.mark.parametrize("device", DEVICES)
def test_negative_controls(device):
    """(c) negative controls: 1.01*T and -T differ from the grid of (b) by > 1e-3 relative; the plain monomial route for
    the L-shape of (a) (np2d.gradient without stable, same factor) differs from the stable result by > 1e-4 * max|T|."""
    from edgebound import np2d
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])
    sc = Scene([body], device)
    T = tensile_vector_scene(sc, np.array([[0.0, 0.3]]), 1.0, family="w4").cpu().numpy()[0]
    grid = dense_T(0.0, 0.3, 1.0)
    s = float(np.abs(grid).max())
    d_scale = float(np.abs(1.01 * T - grid).max())
    d_neg = float(np.abs(-T - grid).max())
    assert d_scale > 1e-3 * s, d_scale
    assert d_neg > 1e-3 * s, d_neg
    print(f"(c) 1.01*T - grid: {d_scale / s:.2e} (> 1e-3)   -T - grid: {d_neg / s:.2e} (> 1e-3)   (scale {s:.3e})")
    bodyL = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
    scL = Scene([bodyL], device)
    pos = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    w = world_polygon(LS, CENTER, ANGLE)
    verts_rep = np.repeat(w[None], 200, axis=0)
    T_stable = tensile_vector_scene(scL, pos, H, family="w4").cpu().numpy()
    T_mono = tensile_factor(H, "w4") * np2d.gradient(verts_rep, pos, "w4p5", h=H, dtype=np.float64)   # plain monomial route
    scale = float(np.abs(T_stable).max())
    dmono = float(np.abs(T_mono - T_stable).max())
    assert dmono > 1e-4 * scale, (dmono, scale)
    print(f"(c) monomial route (w4p5, no stable) vs stable: {dmono / scale:.2e} (> 1e-4)   scale {scale:.3e}")


@pytest.mark.parametrize("device", DEVICES)
def test_guards(device):
    """(d) guards: family != "w2"/"w4" and non-SurfaceRep bodies raise NotImplementedError (family first, then the
    SurfaceRep guard)."""
    from edgebound.implicitBodies import DiskBody
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])
    sc = Scene([body], device)
    pts = np.array([[0.0, 0.3]])
    with pytest.raises(NotImplementedError, match="Wendland C2 and C4 only"):
        tensile_vector_scene(sc, pts, 1.0, family="w9")
    si = Scene([Body(bodyId=0, reps=[ImplicitRep(DiskBody(center=(0.5, 0.5), radius=0.5))])], device)
    with pytest.raises(NotImplementedError, match="SurfaceRep bodies only"):
        tensile_vector_scene(si, pts, 1.0, family="w4")
    sv = Scene([Body(bodyId=0, reps=[VolumeRep(LV, LE)])], device)
    with pytest.raises(NotImplementedError, match="SurfaceRep bodies only"):
        tensile_vector_scene(sv, pts, 1.0, family="w4")


@pytest.mark.parametrize("device", DEVICES)
def test_idempotence(device):
    """(e) idempotence: two successive calls (each family) give identical results; warpbc.STABLE_KERNELS has exactly the
    keys it had after the first call (len unchanged after the second)."""
    from edgebound import warpbc
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])
    sc = Scene([body], device)
    pts = np.array([[0.0, 0.3]])
    for family in ("w2", "w4"):
        T1 = tensile_vector_scene(sc, pts, 1.0, family=family).cpu().numpy()
        n1 = len(warpbc.STABLE_KERNELS)
        T2 = tensile_vector_scene(sc, pts, 1.0, family=family).cpu().numpy()
        n2 = len(warpbc.STABLE_KERNELS)
        assert np.array_equal(T1, T2), (family, T1, T2)
        assert n1 == n2, (family, n1, n2)
    print("(e) idempotence OK: two calls identical (w2, w4); STABLE_KERNELS len unchanged on the second call")
