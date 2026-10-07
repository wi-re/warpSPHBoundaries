"""Discrete-complement wall moments (cfg.complementMoments, prototype; Morris + noslipMoment): the wall moments are the full-plane value minus the particle's own discrete fluid moments, so fluid sum + wall term is EXACT for a
velocity field that is quadratic in the wall distance, on the actual particle neighbourhood.

Plane Poiseuille field u_x = A y (W - y) between two plates (quadratic in the distance to either wall, u'' = -2 A), uniform density (no pressure force): every particle that sees a wall gets a_x = nu (-2 A) to 1e-9
(nu = alpha c0 H / (8 xi), the calibrated viscosity); the curved-table closure (complementMoments False) misses it by more than 1 % on the first row (negative control).  Float64 contracts.
"""
import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, BoxRep, Scene
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
pytestmark = pytest.mark.skipif(real != wp.float64, reason="float64 contracts")


def channel(device, complement):
    n, W = 24, 0.5
    dx = 1.0 / n
    ny = int(round(W / dx))
    X, Y = np.meshgrid(dx * (np.arange(n) + .5), dx * (np.arange(ny) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    A = 0.4
    vel = np.stack([A * pos[:, 1] * (W - pos[:, 1]), np.zeros(len(pos))], 1)
    plate = lambda yc, k: Body(bodyId=k, center=(0.5, yc), reps=[BoxRep((-0.8, -0.15), (0.8, 0.15))])
    scene = Scene([plate(-0.15, 0), plate(W + 0.15, 1)], device)
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=10.0, alpha=0.5, periodic=Periodic((0, -9), (1, 9), (True, False)), graphStep=False, shifting=False, wallViscosityForm="noslipMoment",
                         fluidViscosity="morris", morrisCalibration=0.985, complementMoments=complement)
    sim = DeltaSPH2D(pos, vel, np.ones(len(pos)), dx, scene, cfg, device, support=4 * dx)
    acc, _, _ = sim.rhs(sim.x, sim.v, sim.rho)
    nu = cfg.alpha * cfg.c0 * sim.H / (8.0 * sim.xi)
    y = sim.x[:, 1]
    near = (y < sim.H) | (y > W - sim.H)
    first = (y < dx) | (y > W - dx)
    return acc[:, 0], nu * (-2 * A), near, first


@pytest.mark.parametrize("device", DEVICES)
def test_complement_is_exact_for_a_quadratic_wall_profile(device):
    ax, target, near, first = channel(device, True)
    assert int(near.sum()) > 100
    assert float((ax[near] - target).abs().max()) <= 1e-9 * abs(target)
    ax0, _, _, _ = channel(device, False)
    assert float((ax0[first] - target).abs().max()) > 1e-2 * abs(target)                      # the continuum tables are not exact on the discrete neighbourhood


@pytest.mark.parametrize("device", DEVICES)
def test_complement_is_automatic_with_morris(device):
    """complementMoments = None (the default) uses the complement with the Morris viscosity: the same acceleration as forcing it on."""
    assert DeltaSPHConfig().complementMoments is None
    a_auto, target, near, _ = channel(device, None)
    a_on, _, _, _ = channel(device, True)
    assert float((a_auto - a_on).abs().max()) == 0.0
    assert float((a_auto[near] - target).abs().max()) <= 1e-9 * abs(target)
