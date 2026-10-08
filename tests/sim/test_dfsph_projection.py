"""The compact approximate projection of DFSPH2D (cfg.projection = 'compact', docs/plan-next-steps.md "DFSPH in closed periodic domains"): a CONVERGED projection that does not damp the resolved flow.

(a) the operator is exactly symmetric (rows weighted by V/rho), negative semi-definite with the constant as its null vector, and a Laplacian: on the rest lattice it returns -(2 pi)^2 sin to 2 %;
(b) the CG converges every step (relative residual below the tolerance) and a Taylor-Green vortex decays at the analytic rate within 2 % (the composed DFSPH operator converged to 100 iterations decays it 1.38x).
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
from warpSPHBoundaries.sim.dfsph2d import DFSPH2D, DFSPHConfig, PACKING, lattice_calibration
from warpSPHBoundaries.sim.wallmoments import MORRIS_ETA2

pytestmark = pytest.mark.skipif(real != wp.float64 or not wp.is_cuda_available(), reason="float64 contracts, GPU")
DEV = "cuda:0"
NU = 0.0185


def _tgv(n, U=0.1):
    dx = 1.0 / n
    X, Y = np.meshgrid(dx * (np.arange(n) + 0.5), dx * (np.arange(n) + 0.5), indexing="ij")
    x = np.stack([X.ravel(), Y.ravel()], 1)
    k = 2 * math.pi
    v = U * np.stack([-np.cos(k * x[:, 0]) * np.sin(k * x[:, 1]), np.sin(k * x[:, 0]) * np.cos(k * x[:, 1])], 1)
    return x, v, dx


def _sim(n, **kw):
    x, v, dx = _tgv(n)
    h = dx / PACKING
    cal = lattice_calibration(dx, dx, h)
    cfg = DFSPHConfig(gravity=(0.0, 0.0), viscosity=NU, boundaryFriction=0.0, wallMass=cal["mu"], maxDt=0.01, periodic=Periodic((0, 0), (1, 1)), densitySolve=False,
                      projection="compact", projectionTol=1e-8, shifting="fixed", shiftA=0.5, **kw)
    return DFSPH2D(x, v, cal["V"], h, None, cfg, DEV)


def test_compact_operator_is_a_symmetric_semidefinite_laplacian():
    sim = _sim(16)
    sim._prepare()
    i, j = sim.pi, sim.pj
    sim.rho = sim._sum(sim.V[j] * sim.W) + sim.lam
    Vt = sim.V / sim.rho
    d = sim._delta(sim.x, i, j)
    w = 2.0 * Vt[i] * Vt[j] * (d * sim.gW).sum(1) / ((d * d).sum(1) + MORRIS_ETA2 * sim.h[i] ** 2) / sim._morris_cal()
    N = len(sim.x)
    A = torch.zeros((N, N), dtype=torch.float64, device=DEV)
    A.index_put_((i, i), w, accumulate=True)
    A.index_put_((i, j), -w, accumulate=True)
    assert float((A - A.T).abs().max()) < 1e-10 * float(A.abs().max())
    ev = torch.linalg.eigvalsh(A)
    assert float(ev.max()) < 1e-9 * float(ev.abs().max())
    assert int((ev.abs() < 1e-9 * float(ev.abs().max())).sum()) == 1                      # the constant only
    f = torch.sin(2 * math.pi * sim.x[:, 0])
    ratio = float((A @ f).dot(f) / (Vt * f).dot(f)) / (-(2 * math.pi) ** 2)
    assert abs(ratio - 1.0) < 0.02


def test_converged_compact_projection_keeps_the_taylor_green_decay():
    sim = _sim(32)
    ke = lambda: float((sim.V * (sim.v ** 2).sum(1)).sum())
    ts, es = [0.0], [ke()]
    while sim.time < 1.0:
        sim.step()
        assert sim.err < 1e-8
        ts.append(sim.time)
        es.append(ke())
    rate = -np.polyfit(ts[5:], np.log(es[5:]), 1)[0]
    assert abs(rate / (4 * NU * (2 * math.pi) ** 2) - 1.0) < 0.02
    assert all(b <= a * (1 + 1e-12) for a, b in zip(es, es[1:]))                            # monotone
