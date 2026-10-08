"""DFSPH viscosity (docs/plan-next-steps.md "DFSPH track", D3): the Morris operator calibrated on the DFSPH lattice, the shared no-slip wall closure (complement moments), periodic domains and a body force.

(a) a shear wave in a periodic box decays with the calibrated viscosity (nu_eff / nu within 2 %);
(b) the shared closure is exact in DFSPH as in delta+: on the channel lattice (rest density 1) the viscous acceleration of a field quadratic in the wall distance is nu u'' to 1e-9 in the rows that see the wall;
(c) the momentum bookkeeping stays exact with viscosity and a rotating body (the viscous wall force is booked): residual < 1e-14.
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
from warpSPHBoundaries.sim.dfsph2d import DFSPH2D, DFSPHConfig, PACKING, lattice_calibration
from warpSPHBoundaries.sim.dfsph_cases import tank_with_obstacle

pytestmark = pytest.mark.skipif(real != wp.float64 or not wp.is_cuda_available(), reason="float64 contracts, GPU")
DEV = "cuda:0"
NU = 0.0185


def _lattice(nx, ny, dx, y0=0.0):
    X, Y = np.meshgrid(dx * (np.arange(nx) + 0.5), y0 + dx * (np.arange(ny) + 0.5), indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


def test_shear_wave_decays_with_the_calibrated_viscosity():
    n = 32
    dx = 1.0 / n
    h = dx / PACKING
    cal = lattice_calibration(dx, dx, h)
    pos = _lattice(n, n, dx)
    vel = np.zeros_like(pos)
    vel[:, 0] = 0.05 * np.sin(2 * math.pi * pos[:, 1])
    cfg = DFSPHConfig(gravity=(0.0, 0.0), viscosity=NU, boundaryFriction=0.0, wallMass=cal["mu"], maxDt=0.01, periodic=Periodic((0, 0), (1, 1)))
    sim = DFSPH2D(pos, vel, cal["V"], h, None, cfg, DEV)
    ts, A = [], []
    while sim.time < 0.6:
        sim.step()
        ts.append(sim.time)
        A.append(float((sim.v[:, 0] * torch.sin(2 * math.pi * sim.x[:, 1])).mean() * 2))
    nu = -np.polyfit(ts, np.log(A), 1)[0] / (2 * math.pi) ** 2
    assert abs(nu / NU - 1.0) < 0.02


def test_closure_is_exact_for_a_wall_quadratic_field():
    n = 32
    dx = 1.0 / n
    h = dx / PACKING
    cal = lattice_calibration(dx, dx, h)
    ny = 16
    dw = cal["dwallY"]
    W = 2 * dw + (ny - 1) * dx
    pos = _lattice(n, ny, dx, y0=dw - 0.5 * dx)
    plate = lambda yc, k: Body(bodyId=k, center=(0.5, yc), reps=[BoxRep((-0.8, -0.15), (0.8, 0.15))])
    cfg = DFSPHConfig(gravity=(0.0, 0.0), viscosity=NU, boundaryFriction=0.0, wallMass=cal["mu"], periodic=Periodic((0, -9), (1, 9), (True, False)), graphIterations=False)
    sim = DFSPH2D(pos, np.zeros_like(pos), cal["V"], h, Scene([plate(-0.15, 0), plate(W + 0.15, 1)], DEV), cfg, DEV)
    sim._prepare()
    sim.rho = torch.ones_like(sim.V)
    A = 0.7
    y = sim.x[:, 1]
    sim.v = torch.stack([A * y * (W - y), torch.zeros_like(y)], 1)
    sim.forceViscous = torch.zeros((2, 2), dtype=torch.float64, device=DEV)
    sim.torque = torch.zeros((3, 2), dtype=torch.float64, device=DEV)
    acc = sim._viscous_accel()
    near = (torch.minimum(y, W - y) < h)
    target = NU * (-2.0 * A)
    assert int(near.sum()) > 50
    assert float((acc[near, 0] - target).abs().max()) <= 1e-9 * abs(target)


def test_momentum_bookkeeping_with_viscosity_and_a_rotating_body():
    sim, _ = tank_with_obstacle(omega=3.0, r=0.008)
    sim.cfg.viscosity = 1e-3
    sim.cfg.boundaryFriction = 0.0
    for _ in range(15):
        sim.step()
        assert sim.balance < 1e-14
    assert float(sim.forceViscous.abs().sum()) > 0.0
