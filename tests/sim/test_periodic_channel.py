"""Rung 3 of the validation ladder (docs/plan-next-steps.md): plane Poiseuille flow between periodic-spanning plates, driven by a body force (`scripts/studies/periodic_channel.py` is the study).

The plates are `BoxRep` bodies longer than the box (a periodic wall: the nearest image of the centre covers every row; `Body._checkCompact` accepts a SPANNING body, refuses one that is neither compact nor spanning).
(a) momentum balance: the plate loads sum to the body force on the fluid, 0.5 %;
(b) the profile is symmetric (the two plates act alike, 3 %), monotone to the centre, and its amplitude is within 15 % of the parabola of the bulk viscosity (the first-order wall model error measured in the study:
    0.90 at 32 particles per unit length, 0.93 at 64; a regression gate, not an accuracy claim);
(c) the exact antisymmetric-mirror wall Laplacian (`wallViscosityForm="noslipMirror"`) acts on the whole relative velocity: a wall that moves drags the fluid (Couette, the sheared plate), `laplacian` (free slip) does not.
Float64 contracts.
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
from warpSPHBoundaries.sim.loads import LoadHistory

pytestmark = pytest.mark.skipif(real != wp.float64 or not wp.is_cuda_available(), reason="float64 contracts, CUDA")
DEV = "cuda:0"
PER = Periodic((0.0, -9.0), (1.0, 9.0), (True, False))


def channel(n=32, W=0.5, wall="noslipMirror", f=0.05, upper_velocity=0.0, body_force=True, alpha=0.5):
    dx = 1.0 / n
    ny = int(round(W / dx))
    W = ny * dx
    X, Y = np.meshgrid(dx * (np.arange(n) + .5), dx * (np.arange(ny) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    plate = lambda i, yc, v=0.0: Body(bodyId=i, center=(0.5, yc), linearVelocity=(v, 0.0), reps=[BoxRep((-0.8, -0.15), (0.8, 0.15))])
    scene = Scene([plate(0, -0.15), plate(1, W + 0.15, upper_velocity)], DEV)
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=10.0, alpha=alpha, periodic=PER, bodyForce=(f, 0.0) if body_force else (0.0, 0.0), graphStep=True, shifting=True, wallViscosityForm=wall)
    return DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, scene, cfg, DEV, support=4 * dx), W, dx


def test_poiseuille_balance_symmetry_and_amplitude():
    sim, W, dx = channel()
    hist = LoadHistory()
    while sim.time < 6.0:
        sim.step()
        hist.record(sim)
    F = sum(hist.total(b)[-200:, 0].mean() for b in range(2))
    assert abs(F / (0.05 * len(sim.x) * dx * dx) - 1.0) < 5e-3
    y, u = sim.x[:, 1].cpu().numpy(), sim.v[:, 0].cpu().numpy()
    bins = np.linspace(0, W, 11)
    prof = np.array([u[(y >= b0) & (y < b1)].mean() for b0, b1 in zip(bins[:-1], bins[1:])])
    assert np.abs(prof - prof[::-1]).max() < 0.03 * prof.max()
    assert (np.diff(prof[:5]) > 0).all()
    nu = 0.5 * 10.0 * 4 * dx / (8 * 2.821384729) * 0.955                      # the shear-wave law of the discretisation (0.955 +- 0.01 of alpha c0 H / (8 xi))
    umax = 0.05 * W * W / (8 * nu)
    assert 0.80 < prof.max() / umax < 1.05, prof.max() / umax


def test_a_moving_plate_drags_the_fluid_only_with_the_noslip_mirror():
    out = {}
    for wall in ("noslipMirror", "laplacian"):
        sim, W, dx = channel(wall=wall, upper_velocity=0.5, body_force=False)
        while sim.time < 1.5:
            sim.step()
        y, u = sim.x[:, 1].cpu().numpy(), sim.v[:, 0].cpu().numpy()
        out[wall] = float(u[y > W - 3 * dx].mean())
    assert out["noslipMirror"] > 0.1 and abs(out["laplacian"]) < 1e-2, out


def test_a_periodic_body_must_fit_or_span_its_cell():
    big = Scene([Body(bodyId=0, center=(0.5, 0.5), reps=[BoxRep((-0.45, -0.1), (0.45, 0.1))])], DEV).setPeriodic(Periodic((0, 0), (1, 1)), 0.1)
    with pytest.raises(ValueError, match="fit its cell"):
        big.bodies[0].fusedReps(0.1, DEV)
    ok = Scene([Body(bodyId=0, center=(0.5, 0.0), reps=[BoxRep((-0.8, -0.1), (0.8, 0.1))])], DEV).setPeriodic(PER, 0.1)
    assert ok.bodies[0].fusedReps(0.1, DEV) is not None
