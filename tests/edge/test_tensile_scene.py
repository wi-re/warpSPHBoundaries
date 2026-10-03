"""Tests for tensile.tensile_vector_scene (the exact wall part of the shifting tensile control, Q2, Wendland C2).

T = int_solid W^4 grad W dA'  =  (1/5) grad int_solid W^5 dA'   (the identity, no boundary term: W^5 vanishes at the
support edge).  The scene route is the exact edge kernel W^5 (the kernel `w2p5`) in the Gradient operation, times the
factor (1/5) c2^5 / (pi^4 c25 H^8).

Tolerances (stated with their reason):
  * (a)  <= 1e-6 * max|T|:  both sides are the SAME integral through independent code (the warp edge kernel w2p5 in
    the scene Gradient operation, vs the float64 plain-numpy stable=(16,8) edge reduction of the same kernel); the
    difference is at the level of the numpy route's own conditioning (~1e-14 for w2p5), so 1e-6 is generous.
  * (b)  <= 5e-5 relative vs the plain-numpy 2000 x 2000 midpoint grid of W^4 grad W (the integral's own quadrature,
    the dominant term; measured ~3.3e-6 on the two points); the smoke T_y values (rtol 1e-6) are the reviewer's
    reference and T_x = 0 (atol 1e-6 |T_y|) by the floor's symmetry (the particle is on the symmetry axis).
"""
import math

import numpy as np
import pytest
import torch
import warp as wp

from edgebound.kernels import KERNELS
from edgebound.scene import Body, ImplicitRep, Scene, SurfaceRep, VolumeRep
from edgebound.tensile import tensile_factor, tensile_vector_scene

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
TD = torch.float64
LS = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
CENTER, ANGLE, H = (0.3, -0.2), 0.7, 0.6
LV = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2]], dtype=float)
LE = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6]])
FLOOR = [(-2.0, -2.0), (2.0, -2.0), (2.0, 0.0), (-2.0, 0.0)]


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def world_polygon(local, center, angle):
    """the Body's local polygon in world coordinates, plain numpy (R(angle) p + center, as Pose.toWorld)."""
    return local @ rot(angle).T + np.asarray(center, dtype=float)


def dense_T(px, py, H, n=2000):
    """the integral's own quadrature: W^4 grad W over the solid (x' in [-H,H], y' in [-H,0], r < H), Wendland C2.
    W = 7/(pi H^2) (1-q)^4 (1+4q),  q = r/H,  dW/dq = 7/(pi H^2)(-4(1-q)^3(1+4q) + 4(1-q)^4),  grad W = (dW/dq)/H * (x-x')/r."""
    xp = -H + (np.arange(n) + 0.5) / n * (2 * H)
    yp = -H + (np.arange(n) + 0.5) / n * (H)
    X, Y = np.meshgrid(xp, yp, indexing="ij")
    rx = px - X
    ry = py - Y
    r = np.sqrt(rx * rx + ry * ry)
    m = r < H
    q = r[m] / H
    c = 7.0 / (math.pi * H * H)
    W = c * (1 - q) ** 4 * (1 + 4 * q)
    dWdq = c * (-4 * (1 - q) ** 3 * (1 + 4 * q) + 4 * (1 - q) ** 4)
    gWx = (dWdq / H) * (rx[m] / r[m])
    gWy = (dWdq / H) * (ry[m] / r[m])
    dA = (2 * H / n) * (H / n)
    return np.array([np.sum(W ** 4 * gWx), np.sum(W ** 4 * gWy)]) * dA


@pytest.mark.parametrize("device", DEVICES)
def test_l_shape_matches_numpy_w2p5_gradient(device):
    """(a) the rotated, translated L-shape (center (0.3,-0.2), angle 0.7, H = 0.6, 200 pts seed 3): the scene route
    vs factor * np2d.gradient(w2p5, stable=(16,8)) (<= 1e-6 max|T|)."""
    from edgebound import np2d
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
    sc = Scene([body], device)
    pos = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    T = tensile_vector_scene(sc, pos, H).cpu().numpy()
    w = world_polygon(LS, CENTER, ANGLE)
    verts_rep = np.repeat(w[None], 200, axis=0)
    ref = tensile_factor(H) * np2d.gradient(verts_rep, pos, "w2p5", h=H, dtype=np.float64, stable=(16, 8))
    scale = float(np.abs(T).max())
    assert scale > 0.0
    err = float(np.abs(T - ref).max())
    assert err <= 1e-6 * scale, (err, scale)


@pytest.mark.parametrize("device", DEVICES)
def test_flat_floor_matches_dense_grid_and_smoke(device):
    """(b) the half-plane (solid below y = 0): the scene route vs the 2000 x 2000 midpoint grid (rel <= 5e-5); the
    smoke T_y (rtol 1e-6) and T_x = 0 (atol 1e-6 |T_y|) on the scene result by the floor's symmetry."""
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])
    sc = Scene([body], device)
    for (px, py, Hh, smoke) in [(0.0, 0.3, 1.0, -0.141011574), (0.0, 0.15, 0.5, -72.1979258)]:
        T = tensile_vector_scene(sc, np.array([[px, py]]), Hh).cpu().numpy()[0]
        grid = dense_T(px, py, Hh)
        # scene vs the grid (the grid's quadrature error is the dominant term)
        assert float(np.abs(T - grid).max()) <= 5e-5 * float(np.abs(grid).max()), (T, grid)
        # scene vs the reviewer's smoke reference (the exact route; the grid is ~3.3e-6 from the smoke)
        assert abs(T[1] - smoke) <= 1e-6 * abs(smoke), (T[1], smoke)
        # symmetry: the particle is on the floor's axis, so T_x = 0
        assert abs(T[0]) <= 1e-6 * abs(T[1])
        assert abs(grid[0]) <= 1e-6 * abs(grid[1])


@pytest.mark.parametrize("device", DEVICES)
def test_magnitude_and_sign_are_real(device):
    """(c) negative controls: 1.01*T and -T differ from T by > 1e-3 relative (a correct, non-zero magnitude and a
    definite sign); both differences are reported."""
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])
    sc = Scene([body], device)
    T = tensile_vector_scene(sc, np.array([[0.0, 0.3]]), 1.0).cpu().numpy()[0]
    grid = dense_T(0.0, 0.3, 1.0)                    # the independent value of (b); the controls are compared with IT (REVIEW-003: comparing 1.01*T with T is arithmetic, not a check)
    s = float(np.abs(grid).max())
    assert float(np.abs(T - grid).max()) <= 5e-5 * s       # the real result agrees (the control below is only meaningful if it does)
    d_scale = float(np.abs(1.01 * T - grid).max())
    d_neg = float(np.abs(-T - grid).max())
    assert d_scale > 1e-3 * s, d_scale
    assert d_neg > 1e-3 * s, d_neg
    print("(c) 1.01*T - grid: max|.| = %.4e   -T - grid: max|.| = %.4e   (scale %.4e)" % (d_scale, d_neg, s))


@pytest.mark.parametrize("device", DEVICES)
def test_non_w2_and_non_surface_raise(device):
    """(d) guards: family != "w2" and non-SurfaceRep bodies raise NotImplementedError before computing anything."""
    from edgebound.implicitBodies import DiskBody
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])
    sc = Scene([body], device)
    pts = np.array([[0.0, 0.3]])
    with pytest.raises(NotImplementedError, match="Wendland C2 only"):
        tensile_vector_scene(sc, pts, 1.0, family="w4")
    si = Scene([Body(bodyId=0, reps=[ImplicitRep(DiskBody(center=(0.5, 0.5), radius=0.5))])], device)
    with pytest.raises(NotImplementedError, match="SurfaceRep bodies only"):
        tensile_vector_scene(si, pts, 1.0)
    sv = Scene([Body(bodyId=0, reps=[VolumeRep(LV, LE)])], device)
    with pytest.raises(NotImplementedError, match="SurfaceRep bodies only"):
        tensile_vector_scene(sv, pts, 1.0)


@pytest.mark.parametrize("device", DEVICES)
def test_w2p5_registration_is_idempotent(device):
    """(e) the w2p5 kernel is registered idempotently: a second call does not change len(KERNELS), and the key is present.
    (The parametrised tests use a hardcoded kernel list, not this dict, so registering w2p5 here cannot add cases.)"""
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])
    sc = Scene([body], device)
    pts = np.array([[0.0, 0.3]])
    tensile_vector_scene(sc, pts, 1.0)
    n1 = len(KERNELS)
    assert "w2p5" in KERNELS
    tensile_vector_scene(sc, pts, 1.0)
    n2 = len(KERNELS)
    assert n1 == n2, (n1, n2)
