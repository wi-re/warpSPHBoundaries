"""cover_vector_scene (Q3a through the unmodified scene layer, kernel cone) vs the independent numpy closed form cover_vector_np.

Tolerances (stated with their reason):
  * (a), (b), (f)  <= 1e-12 * H^2:  both sides are float64 closed forms of the SAME integral through independent code
    (the Warp edge channels of the scene layer vs the own numpy edge formula of cover.py); the difference is a few
    ulps of an O(H^2)-magnitude number (reviewer probe: 3.9e-16). 1e-12 is ~6 orders above float64 epsilon.
  * (c)  absolute 1e-6 on O(1) values: smoke values from the work document, 5 orders of float64 slack.
  * (e)  negative controls: a 1 % error in H (case a against cover_vector_np(..., H * 1.01)) must be visible at
    > 1e-4 (the values are O(H^2) = O(0.36), so 1 % of scale is ~3e-3); a sign flip must be visible at > 1e-2.
"""
import math

import numpy as np
import pytest
import torch
import warp as wp

from warpSPHBoundaries.scene import cover
from warpSPHBoundaries.scene.implicitBodies import DiskBody
from warpSPHBoundaries.scene.scene import Body, ImplicitRep, Scene, SurfaceRep, VolumeRep

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
TD = torch.float64
LS = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
CENTER, ANGLE, H = (0.3, -0.2), 0.7, 0.6
TRI30 = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]], dtype=float)             # local triangle of case (b)
LV = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2]], dtype=float)
LE = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6]])


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def world_polygon(local, center, angle):
    """the Body's local polygon in world coordinates, plain numpy (R(angle) p + center, as Pose.toWorld)."""
    return local @ rot(angle).T + np.asarray(center, dtype=float)


@pytest.mark.parametrize("device", DEVICES)
def test_l_shape_scene_matches_numpy_world_polygon(device):
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
    sc = Scene([body], device)
    pos = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    got = cover.cover_vector_scene(sc, pos, H).cpu().numpy()
    ref = cover.cover_vector_np(pos, world_polygon(LS, CENTER, ANGLE), H)
    assert np.abs(got - ref).max() <= 1e-12 * H * H


@pytest.mark.parametrize("device", DEVICES)
def test_two_bodies_is_the_sum_of_the_single_body_results(device):
    b1 = Body(bodyId=0, reps=[SurfaceRep.polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
    b2 = Body(bodyId=1, reps=[SurfaceRep.polygon(TRI30)], center=(3.0, 0.0), angle=math.pi / 6)
    sc = Scene([b1, b2], device)
    pos = np.random.default_rng(7).uniform(-1, 4.2, (200, 2))
    got = cover.cover_vector_scene(sc, pos, 1.0).cpu().numpy()
    ref = cover.cover_vector_np(pos, [(0, 0), (1, 0), (1, 1), (0, 1)], 1.0) + cover.cover_vector_np(pos, world_polygon(TRI30, (3.0, 0.0), math.pi / 6), 1.0)
    assert np.abs(got - ref).max() <= 1e-12 * 1.0 ** 2


@pytest.mark.parametrize("device", DEVICES)
def test_smoke_values_unit_square(device):
    sc = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])], device)
    x = torch.tensor([[0.3, 0.4], [1.3, 0.5]], dtype=TD, device=device)
    got = cover.cover_vector_scene(sc, x, 1.0).cpu().numpy()
    np.testing.assert_allclose(got, [[-0.3457904, -0.1702250], [0.5929108, 0.0]], atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_non_surface_bodies_raise(device):
    pos = np.random.default_rng(3).uniform(-2, 3, (8, 2))
    disk = DiskBody(center=(0.5, 0.5), radius=0.5)
    si = Scene([Body(bodyId=0, reps=[ImplicitRep(disk)])], device)
    with pytest.raises(NotImplementedError, match="SurfaceRep bodies only"):
        cover.cover_vector_scene(si, pos, 1.0)
    sv = Scene([Body(bodyId=0, reps=[VolumeRep(LV, LE)])], device)
    with pytest.raises(NotImplementedError, match="SurfaceRep bodies only"):
        cover.cover_vector_scene(sv, pos, 1.0)


@pytest.mark.parametrize("device", DEVICES)
def test_negative_controls_wrong_H_and_wrong_sign(device):
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
    sc = Scene([body], device)
    pos = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    got = cover.cover_vector_scene(sc, pos, H).cpu().numpy()
    w = world_polygon(LS, CENTER, ANGLE)
    assert np.abs(got - cover.cover_vector_np(pos, w, H * 1.01)).max() > 1e-4
    assert np.abs(got - (-got)).max() > 1e-2


@pytest.mark.parametrize("device", DEVICES)
def test_moved_body_without_rebuilding_the_scene(device):
    body = Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)
    sc = Scene([body], device)
    pos = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    newc = (1.7, 0.9)
    body.center = torch.tensor(newc, dtype=TD, device=device)
    got = cover.cover_vector_scene(sc, pos, H).cpu().numpy()
    ref = cover.cover_vector_np(pos, world_polygon(LS, newc, ANGLE), H)
    assert np.abs(got - ref).max() <= 1e-12 * H * H
