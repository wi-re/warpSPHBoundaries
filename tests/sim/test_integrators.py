"""`DeltaSPHConfig.integrator`: the warpSPHIntegrators scheme that advances particles AND bodies (sim/system.py).

(a) exact poses: a fibre with a constant angular rate and a constant acceleration ends every step of every scheme at the exact pose (angle = omega t to 1e-12, centre = c0 + v0 t + a t^2 / 2 to 1e-12; the
    bodies are integrated fields like the particles, not a host Euler update);
(b) the schemes agree on the physics: hydrostatic tank with a submerged fibre bundle, 0.3 s: stable (rho in [0.99, 1.02]), the buoyancy on the bundle within 8 % of rho g A (resolution) and within 2 % between the schemes
    (the stage loads are weighted with the scheme's weights: [0, 1] for the default, [1/2, 1/2] for the Verlet pair, the tableau b for the Runge-Kutta schemes);
(c) the default is the old symplectic Euler: `integrator='Symplectic Euler'` is the unset configuration.
Float64 contracts.
"""
import math

import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.sim.deltasph2d import DeltaSPHConfig
from warpSPHBoundaries.sim.loads import LoadHistory
from test_fibre_bundle_solver import FIBRES, G, make

pytestmark = pytest.mark.skipif(real != wp.float64 or not wp.is_cuda_available(), reason="float64 contracts, CUDA")
DEV = "cuda:0"
SCHEMES = ["Symplectic Euler", "Velocity Verlet", "Leap Frog", "Midpoint", "RK4"]


def cfg_for(name, dp=0.04, Hwater=0.5):
    return DeltaSPHConfig(gravity=(0.0, -G), c0=20.0 * math.sqrt(G * Hwater), noPen="impulse", shifting=True, integrator=name)


@pytest.mark.parametrize("name", SCHEMES)
def test_bodies_follow_the_exact_pose(name):
    sim = make(True, dp=0.04, cfg=cfg_for(name), motion=((0.1, 0.0), (0.0, 0.5)))
    b = sim.scene.bodies[1]
    b.angularVelocity = 2.0
    c0 = b.center.clone()
    for _ in range(40):
        sim.step()
    t = sim.time
    assert abs(float(b.angle) - 2.0 * t) < 1e-12
    exact = c0 + torch.tensor([0.1, 0.0], dtype=torch.float64, device=DEV) * t + 0.5 * torch.tensor([0.0, 0.5], dtype=torch.float64, device=DEV) * t * t
    assert float((b.center - exact).abs().max()) < 1e-12, (name, b.center, exact)


def test_schemes_agree_on_a_submerged_bundle():
    buoy = math.pi * sum(R * R for _, R in FIBRES) * G
    mean = {}
    for name in SCHEMES:
        sim = make(True, dp=0.04, cfg=cfg_for(name))
        hist = LoadHistory()
        while sim.time < 0.3:
            sim.step()
            hist.record(sim)
        assert 0.99 < float(sim.rho.min()) and float(sim.rho.max()) < 1.02, (name, float(sim.rho.min()), float(sim.rho.max()))
        mean[name] = float(hist.total(1)[-60:, 1].mean())
        assert abs(mean[name] / buoy - 1.0) < 0.08, (name, mean[name] / buoy)                        # dp = 0.04: the default itself is 5.2 % low (3 % at dp = 0.03, test_fibre_bundle_solver)
    vals = np.array(list(mean.values()))
    assert (vals.max() - vals.min()) / buoy < 0.02, mean


def test_default_is_the_symplectic_euler():
    assert DeltaSPHConfig().integrator == "Symplectic Euler"
    with pytest.raises(ValueError):
        sim = make(True, dp=0.04, cfg=cfg_for("no such scheme"))
        sim.step()
