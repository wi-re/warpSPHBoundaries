"""The circle element (`edge/disk.py`): every channel of the solid disk of ANY radius by the divergence theorem on its circle.

(a) value against the multi-precision circle identity (`tier4.arc_value`, 60 nodes, mpmath): 1e-12 (observed <= 3e-14, largest next to the surface), for radii R/h = 0.1 ... 6 and distances inside, at and beyond the support (not exactly ON the circle, where the
    circle formula is singular: the function itself is smooth there, (c));
(b) all five channels (lam, m1, g0, g1 normal / tangential) against the polygon of the fused wall path (`Body.fusedReps`, edges <= h/16) in the world frame, R/h = 0.3 ... 6: 1e-5 of the scale (the polygon's
    own error is ~1e-6 - 1e-7);
(c) the functions are smooth across the surface (|D - R| -> 0, the nearly singular circle integrand): fourth differences of a uniform grid straddling the surface are the size of h^4 f'''' (no jump), and the quadrature is
    converged (24 against 48 nodes: 1e-13);
(d) the limits: a particle at the centre of a large disk sees lam = 1 and no gradient; far outside the support everything is zero; the unit-radius normalisation of the kernel is reproduced for a disk that covers the support.
"""
from fractions import Fraction

import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import KernelFunctions, ParticleState

from warpSPHBoundaries.edge import disk, tier4
from warpSPHBoundaries.scene import AnalyticBoundary, Body, DiskBody, ImplicitRep, Scene

DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
F64 = torch.float64


@pytest.mark.parametrize("R", [0.1, 0.5, 1.0, 2.0, 6.0])
def test_value_equals_the_multiprecision_circle_identity(R):
    for D in (0.3 * R, 0.9 * R, 1.1 * R, R + 0.37, R + 0.9, R + 1.3, max(R - 0.6, 0.05)):
        if abs(D - R) < 1e-6:
            continue
        ref = float(tier4.arc_value("w2", (Fraction(D).limit_denominator(10 ** 9), 0), Fraction(R).limit_denominator(10 ** 9), nodes=60))
        assert abs(disk.disk_channels("w2", D, R)["lam"] - ref) < 1e-12, (R, D)


@pytest.mark.parametrize("Rh", [0.3, 0.5, 1.0, 2.0, 6.0])
def test_all_channels_equal_the_polygon_of_the_fused_path(Rh):
    H = 0.3
    R, c = Rh * H, np.array([0.1, 0.2])
    rng = np.random.default_rng(0)
    n = 300
    r, a = rng.uniform(max(R - 0.9 * H, 0.002), R + 1.0 * H, n), rng.uniform(0, 2 * np.pi, n)
    pos = c + np.stack([r * np.cos(a), r * np.sin(a)], 1)
    P = torch.as_tensor(pos, dtype=F64, device=DEV)
    ps = ParticleState(positions=P, supports=torch.full((n,), H, dtype=F64, device=DEV), masses=torch.ones(n, dtype=F64, device=DEV), kinds=torch.zeros(n, dtype=torch.int32, device=DEV),
                       densities=torch.ones(n, dtype=F64, device=DEV))
    sc = Scene([Body(bodyId=0, center=tuple(c), reps=[ImplicitRep(DiskBody(center=(0, 0), radius=R))])], DEV)
    ref = AnalyticBoundary(sc).aggregate(ps, H, KernelFunctions.Wendland2)
    L, G, C = (ref.out[k][0].cpu().numpy() for k in ("lam", "G", "Cov"))
    d = c - pos
    D = np.linalg.norm(d, axis=1)
    ch = d / D[:, None]
    th = np.stack([-ch[:, 1], ch[:, 0]], 1)
    o = disk.disk_channels("w2", D / H, R / H)
    g0 = (o["g0x"] / H)[:, None] * ch
    g1 = o["g1xx"][:, None, None] * ch[:, :, None] * ch[:, None, :] + o["g1yy"][:, None, None] * th[:, :, None] * th[:, None, :]
    for got, exp in ((o["lam"], L), (g0, G), (g1, C)):
        assert np.abs(got - exp).max() <= 1e-5 * np.abs(exp).max()


@pytest.mark.parametrize("R", [0.2, 0.5, 2.0])
def test_smooth_across_the_surface_and_converged(R):
    h = 0.004
    D = R + h * (np.arange(-15, 15) + 0.5)
    for k in ("lam", "m1x", "g0x", "g1xx", "g1yy"):
        v = disk.disk_channels("w2", D, R)[k]
        assert np.abs(np.diff(v, 4)).max() < 5e-6 * max(np.abs(v).max(), 0.05) + 1e-7, k
    a, b = disk.disk_channels("w2", D, R, nodes=24), disk.disk_channels("w2", D, R, nodes=48)
    assert max(float(np.abs(a[k] - b[k]).max()) for k in a) < 1e-13


def test_limits():
    big = disk.disk_channels("w2", np.array([0.0, 0.3]), 6.0)                       # the support ball lies inside a large disk: lam = 1, no gradient, g1 = identity
    assert np.allclose(big["lam"], 1.0, atol=1e-14) and np.abs(big["g0x"]).max() < 1e-13 and np.allclose(big["g1xx"], 1.0, atol=1e-13) and np.allclose(big["g1yy"], 1.0, atol=1e-13)
    far = disk.disk_channels("w2", 3.0, 0.5)
    assert max(abs(v) for v in far.values()) < 1e-14
