"""`Body.massProperties` / scene/massprops.py: area, centre of mass and polar inertia of the analytic bodies.

Reference values: closed forms (disk pi R^2, J = m R^2 / 2; rectangle J = m (w^2 + h^2) / 12; regular n-gon) and a brute-force midpoint grid over the SOLID built from `Scene.inside` (independent of the
Green's-theorem sums), with the body displaced from the origin (parallel-axis shifts).  Unbounded solids (a tank, a hole, a half plane) are refused.
"""
import math

import numpy as np
import pytest
import torch

from warpSPHBoundaries.scene.implicitBodies import DiskBody, HalfPlaneBody
from warpSPHBoundaries.scene.scene import Body, BoxRep, DiskArrayRep, ImplicitRep, Scene, SurfaceRep


def grid_moments(body, lo, hi, n=900):
    sc = Scene([body], "cpu")
    xs = np.linspace(lo[0], hi[0], n, endpoint=False) + 0.5 * (hi[0] - lo[0]) / n
    ys = np.linspace(lo[1], hi[1], n, endpoint=False) + 0.5 * (hi[1] - lo[1]) / n
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float64)
    m = sc.inside(P).numpy().astype(float)
    dA = (hi[0] - lo[0]) * (hi[1] - lo[1]) / n ** 2
    # the body frame has centre 0 and angle 0, so world = body frame
    A = m.sum() * dA
    cx, cy = (m * X.ravel()).sum() * dA / A, (m * Y.ravel()).sum() * dA / A
    J = (m * (X.ravel() ** 2 + Y.ravel() ** 2)).sum() * dA
    return A, (cx, cy), J


def check(body, lo, hi, tol=3e-3):
    mp = body.massProperties(1.0)
    A, c, J = grid_moments(body, lo, hi)
    assert abs(mp["area"] - A) < tol * A
    assert np.allclose(mp["com"], c, atol=tol * math.sqrt(A))
    assert abs(mp["inertiaOrigin"] - J) < tol * J
    assert abs(mp["inertia"] - (J - A * (c[0] ** 2 + c[1] ** 2))) < 2 * tol * J
    return mp


def test_disk_and_disk_bundle_closed_forms():
    R = 0.3
    mp = Body(reps=[DiskArrayRep([(0.0, 0.0)], [R])]).massProperties(2.0)
    assert abs(mp["mass"] - 2.0 * math.pi * R * R) < 1e-14 and abs(mp["inertia"] - 0.5 * mp["mass"] * R * R) < 1e-14
    mp = Body(reps=[ImplicitRep(DiskBody(center=(0.2, -0.1), radius=R))]).massProperties(1.0)
    assert np.allclose(mp["com"], (0.2, -0.1)) and abs(mp["inertia"] - 0.5 * mp["mass"] * R * R) < 1e-14
    assert abs(mp["inertiaOrigin"] - (0.5 * mp["mass"] * R * R + mp["mass"] * (0.2 ** 2 + 0.1 ** 2))) < 1e-14
    b = Body(reps=[DiskArrayRep([(-0.5, 0.0), (0.6, 0.3)], [0.1, 0.2])])
    check(b, (-0.8, -0.4), (0.9, 0.6))


def test_box_and_polygons():
    mp = Body(reps=[BoxRep((0.1, 0.2), (0.5, 0.3))]).massProperties(1.0)
    w, h = 0.4, 0.1
    assert abs(mp["area"] - w * h) < 1e-15 and np.allclose(mp["com"], (0.3, 0.25))
    assert abs(mp["inertia"] - mp["mass"] * (w * w + h * h) / 12.0) < 1e-15
    n, R = 6, 0.4
    hexa = SurfaceRep.polygon([(0.2 + R * math.cos(2 * math.pi * k / n), -0.1 + R * math.sin(2 * math.pi * k / n)) for k in range(n)], "inside")
    mp = Body(reps=[hexa]).massProperties(1.0)
    A = 1.5 * math.sqrt(3) * R * R
    assert abs(mp["area"] - A) < 1e-14 and np.allclose(mp["com"], (0.2, -0.1))
    assert abs(mp["inertia"] - A * (5.0 / 12.0) * R * R) < 1e-13                                # J of a regular hexagon: (5 / 12) m R^2 (circumradius R)
    tri = SurfaceRep.polygon([(0.0, 0.0), (0.8, 0.1), (0.3, 0.6)], "inside")
    check(Body(reps=[tri]), (-0.1, -0.1), (0.9, 0.7))
    check(Body(reps=[tri, BoxRep((-0.6, -0.5), (-0.3, -0.2))]), (-0.7, -0.6), (0.9, 0.7))     # two parts in one body


def test_unbounded_solids_are_refused():
    with pytest.raises(ValueError):
        Body(reps=[SurfaceRep.polygon([(0, 0), (1, 0), (1, 1), (0, 1)], "outside")]).massProperties()
    with pytest.raises(ValueError):
        Body(reps=[BoxRep((0, 0), (1, 1), solid="outside")]).massProperties()
    with pytest.raises(ValueError):
        Body(reps=[ImplicitRep(HalfPlaneBody(point=(0.0, 0.0), normal=(0.0, 1.0)))]).massProperties()
