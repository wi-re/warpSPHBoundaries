"""A bundle of fibres (disks of different radii, ONE body, `DiskArrayRep`) in the solver: a hydrostatic tank with three fibres.

(a) the bundle is the same physics as the fibres as separate bodies: right-hand side (a, drho, the loads summed over the bodies) equal to 1e-9 (the same disk element, only the summation grouping differs); the
    free-slip wall viscosity normalises the wall direction per body, so it differs only for the particles that see two fibres at once (checked separately);
(b) absolute: the pressure load of the water on the bundle is the total buoyancy rho g pi sum R_i^2 upwards to 3 % (the sampling around every fibre conforms to it, see `test_implicit_disk_solver.make`), the
    horizontal load vanishes to 1 %;
(c) a translating bundle: the graph replay equals the eager step bit for bit (the pose of the body is a device input, the slot list and the table lookups are inside the graph), and the step does not fall back;
(d) the bundle needs ONE wall evaluation, not one per fibre: the number of `FusedWall` items is that of one body.
"""
import math

import numpy as np
import pytest
import torch
import warp as wp


import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, ImplicitRep, Scene
from warpSPHBoundaries.scene.implicitBodies import DiskBody
from warpSPHBoundaries.sim import dfsph2d
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA")
DEV = "cuda:0"
F64 = torch.float64
G = 9.81
FIBRES = [((-0.4, -0.15), 0.08), ((0.0, -0.15), 0.10), ((0.4, -0.15), 0.12)]


def make(bundle, dp=0.03, cfg=None, motion=None, layers=4, Hwater=0.5, L=1.2, Htank=0.8):
    c0 = 20.0 * math.sqrt(G * Hwater)
    nx, ny = int(round(L / dp)), int(round(Hwater / dp))
    X, Y = np.meshgrid(-L / 2 + dp * (np.arange(nx) + 0.5), -Htank / 2 + dp * (np.arange(ny) + 0.5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    keep = np.ones(len(pos), dtype=bool)
    for c, R in FIBRES:
        keep &= np.linalg.norm(pos - np.array(c), axis=1) > R + layers * dp
    rings = []
    for c, R in FIBRES:                                                                     # conforming layers around every fibre
        for k in range(layers):
            r = R + (k + 0.5) * dp
            n = int(round(2 * math.pi * r / dp))
            a = 2 * math.pi * (np.arange(n) + 0.5 * (k % 2)) / n
            rings.append(np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1))
    pos = np.vstack([pos[keep]] + rings)
    rho = 1.0 + G * np.clip(-Htank / 2 + Hwater - pos[:, 1], 0.0, None) / c0 ** 2
    tank = dfsph2d.domain_scene("surface", (-L / 2, -Htank / 2), (L / 2, Htank / 2), 4 * dp, DEV).bodies[0]
    if bundle:
        bodies = [tank, Body(bodyId=1, reps=[DiskArrayRep([c for c, _ in FIBRES], [R for _, R in FIBRES])])]
    else:
        bodies = [tank] + [Body(bodyId=1 + i, center=c, reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=R))]) for i, (c, R) in enumerate(FIBRES)]
    cfg = cfg or DeltaSPHConfig(gravity=(0.0, -G), c0=c0, noPen="impulse", shifting=True)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), rho, dp, Scene(bodies, DEV), cfg, DEV)
    if motion:
        b = sim.scene.bodies[1]
        b.linearVelocity, b.linearAcceleration = torch.tensor(motion[0], dtype=F64, device=DEV), torch.tensor(motion[1], dtype=F64, device=DEV)
    return sim


def test_bundle_equals_separate_fibres():
    a, b = make(True), make(False)
    v = torch.as_tensor(np.random.default_rng(0).normal(0, 0.2, a.x.shape), dtype=F64, device=DEV)
    reach = torch.stack([(a.x - torch.tensor(c, dtype=F64, device=DEV)).norm(dim=1) < R + a.H for c, R in FIBRES])          # which fibres every particle can see
    single = reach.sum(0) <= 1
    assert int((~single).sum()) > 0
    ra, rb = a.rhs(a.x, v, a.rho, want_forces=True), b.rhs(b.x, v, b.rho, want_forces=True)
    # the integrals are additive over the disks, so everything linear in them is identical: continuity, pressure force, loads; the wall VISCOSITY of the free-slip mirror normalises the wall direction per body, so it differs
    # (by design) only for particles that see two fibres at once
    assert float((ra[1] - rb[1]).abs().max()) <= 1e-9 * float(rb[1].abs().max())
    assert float((ra[0][single] - rb[0][single]).abs().max()) <= 1e-9 * float(rb[0].abs().max())
    for s_ in (a, b):
        s_.cfg.wallViscosity = False
    ra, rb = a.rhs(a.x, v, a.rho, want_forces=True), b.rhs(b.x, v, b.rho, want_forces=True)
    for k in (0, 1):
        assert float((ra[k] - rb[k]).abs().max()) <= 1e-9 * float(rb[k].abs().max()), k
    assert float((ra[2][:, 1, :2] - rb[2][:, 1:, :2].sum(1)).abs().max()) <= 1e-9 * float(rb[2][..., :2].abs().max())          # forces (the torque is about the body centre: the origin for the bundle, each fibre's own centre otherwise)


@pytest.mark.parametrize("dp", [0.03, 0.02])
def test_total_buoyancy_of_the_bundle(dp):
    sim = make(True, dp=dp)
    _, _, loads = sim.rhs(sim.x, sim.v, sim.rho, want_forces=True)
    buoy = G * math.pi * sum(R * R for _, R in FIBRES)
    Fx, Fy = float(loads[0, 1, 0]), float(loads[0, 1, 1])
    assert abs(Fy - buoy) / buoy < 0.03, (Fy, buoy)
    assert abs(Fx) < 0.01 * buoy, Fx


def test_moving_bundle_graph_equals_eager():
    def go(graph):
        cfg = DeltaSPHConfig(gravity=(0.0, -G), c0=20.0 * math.sqrt(G * 0.5), noPen="impulse", shifting=True, graphStep=graph)
        sim = make(True, cfg=cfg, motion=((0.2, 0.0), (0.0, 0.05)))
        for _ in range(20):
            sim.step()
        return sim
    a, b = go(False), go(True)
    assert b._graphed and b._graphed.stats["replays"] >= 15 and b._graphed.stats["eager"] == 0
    for name in ("x", "v", "rho", "wallLoads"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert torch.equal(a.scene.bodies[1].center, b.scene.bodies[1].center) and float(a.scene.bodies[1].center[0]) > 0.0


def test_one_wall_evaluation_for_the_whole_bundle():
    sim = make(True)
    ws = sim._wall_state(sim.x, sim.rho)[3]
    kinds = [type(it["rep"]).__name__ for it in ws.items]
    assert kinds.count("DiskArrayRep") == 1 and len(kinds) <= 2, kinds                      # the tank (one polygon item) and ONE item for all fibres
