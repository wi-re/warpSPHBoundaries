"""cone_area_scene (Q3b through the Scene layer) vs the T3.1 scalar / brute-force reference.

Tolerances (stated with their reason):
  * (a), (c), (e)  <= 1e-11 * H^2 (absolute 1e-11 for (c), H = 1):  both sides are float64 closed forms of the SAME area through independent code
    (the scene route cone_area_scene -> cone_area, world vertices, vs the plain-numpy world polygon fed to cone_area_scalar); the difference
    is a few ulps of an O(H^2)-magnitude number; (b) is the independent check.
  * (b)  <= 1e-3 * H^2 vs the own brute-force midpoint polar grid (1000 x 2000, even-odd); the brute grid error is the dominant term
    (measured ~3e-4 H^2 on the T3.1 recipe), while a formula / pose error is >= O(H^2).
  * (f)  negative control: the local (un-posed) polygon in the scalar reference must differ from the scene result by > 1e-2 * H^2 at some
    point of (a) (a wrong pose is a geometry error, >= O(H^2)).
"""
import math

import numpy as np
import pytest
import torch
import warp as wp

from edgebound.scene.cone_area import cone_area_scene, cone_area_scalar
from edgebound.scene.implicitBodies import DiskBody
from edgebound.scene.scene import Body, ImplicitRep, Scene, SurfaceRep, VolumeRep

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
TD = torch.float64
LS = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
CENTER, ANGLE, H = (0.3, -0.2), 0.7, 0.6
TRI30 = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]], dtype=float)
CAV = np.array([(0, 0), (2, 0), (2, 1), (0, 1)], dtype=float)                        # the cavity square (a hole in an infinite solid)
LV = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2]], dtype=float)
LE = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6]])


def edges_of(pts):
    n = len(pts)
    return np.stack([np.arange(n), (np.arange(n) + 1) % n], 1)


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def world_polygon(local, center, angle):
    """the Body's local polygon in world coordinates, plain numpy (R(angle) p + center, as Pose.toWorld)."""
    return local @ rot(angle).T + np.asarray(center, dtype=float)


def brute(p, V, E, H, th, al, background=0, nr=1000, nphi=2000):
    """independent oracle (own copy of the T3.1 brute force): midpoint polar grid over disk ∩ wedge (full circle if al >= pi), even-odd point-in-polygon (solid = inside, or outside if background = 1); no winding numbers, no edge formula."""
    p = np.asarray(p, float)
    V = np.asarray(V, float)
    E = np.asarray(E, int)
    a0, a1 = (th - al, th + al) if al < math.pi else (0.0, 2.0 * math.pi)
    ph = a0 + (np.arange(nphi) + 0.5) / nphi * (a1 - a0)
    rr = (np.arange(nr) + 0.5) / nr * H
    X = p[0] + rr[:, None] * np.cos(ph)[None]
    Y = p[1] + rr[:, None] * np.sin(ph)[None]
    inside = np.zeros(X.shape, bool)
    for e0, e1 in E:
        (x1, y1), (x2, y2) = V[int(e0)], V[int(e1)]
        cond = (y1 > Y) != (y2 > Y)
        xi = x1 + (Y - y1) * (x2 - x1) / (y2 - y1 + 1e-300)
        inside ^= cond & (X < xi)
    solid = inside if background == 0 else ~inside
    w = rr[:, None] * (H / nr) * ((a1 - a0) / nphi)
    return float((w * solid).sum())


def _axis_angles(ax):
    return np.array([math.atan2(a[1], a[0]) for a in ax])


@pytest.mark.parametrize("device", DEVICES)
def test_l_shape_scene_matches_scalar_world_polygon_and_brute(device):
    """(a) the rotated, translated L-shape: scene vs scalar on the world polygon (all 200 pts, <= 1e-11 H^2), vs brute (first 20, <= 1e-3 H^2)."""
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
    sc = Scene([body], device)
    pos = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    ax = np.random.default_rng(4).uniform(-1, 1, (200, 2))
    th = _axis_angles(ax)
    got = cone_area_scene(sc, pos, ax, math.pi / 6, H).cpu().numpy()
    w = world_polygon(LS, CENTER, ANGLE)
    wE = edges_of(w)
    ref = np.array([cone_area_scalar(tuple(pos[i]), th[i], math.pi / 6, H, w, wE, 0) for i in range(200)])
    assert np.abs(got - ref).max() <= 1e-11 * H * H
    worst = 0.0
    for i in range(20):
        ref_i = brute(tuple(pos[i]), w, wE, H, th[i], math.pi / 6, 0)
        worst = max(worst, abs(got[i] - ref_i) / (H * H))
    assert worst <= 1e-3, worst


@pytest.mark.parametrize("device", DEVICES)
def test_cavity_scene_matches_brute(device):
    """(b) a cavity body (solid = the unbounded outside, background 1): vs brute for the narrow and the full wedge (<= 1e-3 H^2)."""
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(CAV, solid="outside")])
    sc = Scene([body], device)
    rng = np.random.default_rng(5)
    pos = rng.uniform([-0.1, -0.1], [2.1, 1.1], (40, 2))
    ax = rng.uniform(-1, 1, (40, 2))
    th = _axis_angles(ax)
    for al in (math.pi / 6, math.pi):
        got = cone_area_scene(sc, pos, ax, al, 0.5).cpu().numpy()
        worst = 0.0
        for i in range(40):
            ref_i = brute(tuple(pos[i]), CAV, edges_of(CAV), 0.5, th[i], al, 1)
            worst = max(worst, abs(got[i] - ref_i) / (0.5 * 0.5))
        assert worst <= 1e-3, (al, worst)


@pytest.mark.parametrize("device", DEVICES)
def test_two_bodies_is_the_sum_of_the_single_body_results(device):
    """(c) two bodies (a unit box at the origin and the 30-deg triangle at (3,0), H = 1): the scene result equals the sum of the two single-body results (<= 1e-11)."""
    b1 = Body(bodyId=0, reps=[SurfaceRep.polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
    b2 = Body(bodyId=1, reps=[SurfaceRep.polygon(TRI30)], center=(3.0, 0.0), angle=math.pi / 6)
    sc = Scene([b1, b2], device)
    pos = np.random.default_rng(7).uniform(-1, 4.2, (200, 2))
    ax = np.random.default_rng(8).uniform(-1, 1, (200, 2))
    got = cone_area_scene(sc, pos, ax, math.pi / 6, 1.0).cpu().numpy()
    s1 = cone_area_scene(Scene([b1], device), pos, ax, math.pi / 6, 1.0).cpu().numpy()
    s2 = cone_area_scene(Scene([b2], device), pos, ax, math.pi / 6, 1.0).cpu().numpy()
    assert np.abs(got - (s1 + s2)).max() <= 1e-11


@pytest.mark.parametrize("device", DEVICES)
def test_non_surface_bodies_raise(device):
    """(d) guard: ImplicitRep (DiskBody) and VolumeRep bodies raise NotImplementedError before computing anything."""
    pos = np.random.default_rng(3).uniform(-2, 3, (8, 2))
    ax = np.random.default_rng(4).uniform(-1, 1, (8, 2))
    disk = DiskBody(center=(0.5, 0.5), radius=0.5)
    si = Scene([Body(bodyId=0, reps=[ImplicitRep(disk)])], device)
    with pytest.raises(NotImplementedError, match="SurfaceRep bodies only"):
        cone_area_scene(si, pos, ax, math.pi / 6, 1.0)
    sv = Scene([Body(bodyId=0, reps=[VolumeRep(LV, LE)])], device)
    with pytest.raises(NotImplementedError, match="SurfaceRep bodies only"):
        cone_area_scene(sv, pos, ax, math.pi / 6, 1.0)


@pytest.mark.parametrize("device", DEVICES)
def test_moved_body_without_rebuilding_the_scene(device):
    """(e) moved body: after body.center = ... without rebuilding the scene, the result equals the scalar reference for the moved polygon (<= 1e-11 H^2)."""
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
    sc = Scene([body], device)
    pos = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    ax = np.random.default_rng(4).uniform(-1, 1, (200, 2))
    th = _axis_angles(ax)
    newc = (1.7, 0.9)
    body.center = torch.tensor(newc, dtype=TD, device=device)
    got = cone_area_scene(sc, pos, ax, math.pi / 6, H).cpu().numpy()
    w = world_polygon(LS, newc, ANGLE)
    wE = edges_of(w)
    ref = np.array([cone_area_scalar(tuple(pos[i]), th[i], math.pi / 6, H, w, wE, 0) for i in range(200)])
    assert np.abs(got - ref).max() <= 1e-11 * H * H


@pytest.mark.parametrize("device", DEVICES)
def test_negative_control_local_polygon_differs(device):
    """(f) negative control: using the local (un-posed) polygon in the scalar reference instead of the world polygon differs from the scene result by > 1e-2 H^2 at some point of (a)."""
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
    sc = Scene([body], device)
    pos = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    ax = np.random.default_rng(4).uniform(-1, 1, (200, 2))
    th = _axis_angles(ax)
    got = cone_area_scene(sc, pos, ax, math.pi / 6, H).cpu().numpy()
    wE = edges_of(LS)
    ref_local = np.array([cone_area_scalar(tuple(pos[i]), th[i], math.pi / 6, H, LS, wE, 0) for i in range(200)])
    assert np.abs(got - ref_local).max() > 1e-2 * H * H
