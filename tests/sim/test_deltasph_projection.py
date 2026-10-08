"""delta+ with the converged compact projection (cfg.pressureSolver = 'projection', docs/plan-next-steps.md "DFSPH in closed periodic domains"): the shared sim/projection.py CG on this scheme's own continuity rate.

(a) the CG converges every step (relative residual below projectionTol) and the density stays rho0 (incompressible: V = m / rho0, no EOS feedback);
(b) a Taylor-Green vortex (periodic, no walls) decays at the analytic rate within 5 % (the EOS path needs c0 and carries pressure noise);
(c) plane Poiseuille flow between periodic-spanning plates (Morris viscosity + the complement no-slip closure): the plate loads balance the body force to 1 % and the amplitude is within 3 % of the parabola.
Float64 contracts.
"""
import math

import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, BoxRep, Scene
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.loads import LoadHistory

pytestmark = pytest.mark.skipif(real != wp.float64 or not wp.is_cuda_available(), reason="float64 contracts, CUDA")
DEV = "cuda:0"
NU = 0.0185
TOL = 1e-8
XI = 2.821384729                                                                # nominal nu = alpha c0 H / (8 xi); the Morris form is calibrated to it


def _alpha(dx, support=2.5):
    return NU * 8 * XI / (10.0 * support * dx)


def test_taylor_green_decays_at_the_analytic_rate_with_a_converged_projection():
    n, U = 32, 0.1
    dx = 1.0 / n
    X, Y = np.meshgrid(dx * (np.arange(n) + 0.5), dx * (np.arange(n) + 0.5), indexing="ij")
    x = np.stack([X.ravel(), Y.ravel()], 1)
    k = 2 * math.pi
    v = U * np.stack([-np.cos(k * x[:, 0]) * np.sin(k * x[:, 1]), np.sin(k * x[:, 0]) * np.cos(k * x[:, 1])], 1)
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=10.0, alpha=_alpha(dx), periodic=Periodic((0, 0), (1, 1)), shifting=True, fluidViscosity="morris", ddt=False,
                         pressureSolver="projection", projectionTol=TOL, graphStep=False)
    sim = DeltaSPH2D(x, v, np.ones(len(x)), dx, None, cfg, DEV, support=2.5 * dx)
    ke = lambda: float((sim.v ** 2).sum())
    ts, es = [0.0], [ke()]
    while sim.time < 1.0:
        sim.step()
        assert sim.projectionResidual < TOL
        assert abs(float(sim.rho.mean()) - cfg.rho0) < 1e-12 * cfg.rho0
        ts.append(sim.time)
        es.append(ke())
    rate = -np.polyfit(ts[5:], np.log(es[5:]), 1)[0]
    assert abs(rate / (4 * NU * k * k) - 1.0) < 0.05, rate / (4 * NU * k * k)


def test_poiseuille_with_the_projection_balances_the_body_force():
    n, W, f = 32, 0.5, 0.05
    dx = 1.0 / n
    ny = int(round(W / dx))
    W = ny * dx
    X, Y = np.meshgrid(dx * (np.arange(n) + .5), dx * (np.arange(ny) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    plate = lambda i, yc: Body(bodyId=i, center=(0.5, yc), reps=[BoxRep((-0.8, -0.15), (0.8, 0.15))])
    scene = Scene([plate(0, -0.15), plate(1, W + 0.15)], DEV)
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=10.0, alpha=_alpha(dx), periodic=Periodic((0.0, -9.0), (1.0, 9.0), (True, False)), bodyForce=(f, 0.0), shifting=True, wallViscosityForm="noslipMoment", fluidViscosity="morris",
                         ddt=False, pressureSolver="projection", projectionTol=TOL, graphStep=False)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, scene, cfg, DEV, support=2.5 * dx)
    hist = LoadHistory()
    while sim.time < 8.0:
        sim.step()
        hist.record(sim)
        assert sim.projectionResidual < TOL
    F = sum(hist.total(b)[-200:, 0].mean() for b in range(2))
    assert abs(F / (f * len(pos) * dx * dx) - 1.0) < 0.01
    y, u = sim.x[:, 1].cpu().numpy(), sim.v[:, 0].cpu().numpy()
    exact = f * y * (W - y) / (2 * NU)
    amp = float((u * exact).sum() / (exact * exact).sum())
    assert abs(amp - 1.0) < 0.03, amp
