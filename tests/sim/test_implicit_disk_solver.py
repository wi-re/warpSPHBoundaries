"""Implicit and SDF bodies in the solver on the fused / graph path (`Body.fusedReps`: the exact tier-2 polygon of a disk resp. the contour of a sampled distance): a disk in a hydrostatic tank.

(a) parity: the right-hand side of the disk as `ImplicitRep` equals the one of the same disk as the polygon `SurfaceRep` (a, drho, the loads) to round-off (the plumbing is exact), the `SdfRep` of the disk to the
    contour accuracy;
(b) absolute: the pressure load of the fluid on the submerged disk is the buoyancy rho g pi R^2 upwards to 3 % (measured 1 %; the horizontal load vanishes by symmetry), and it is booked on the disk body (the sampling of the
    fluid around the disk conforms to it, see `make`);
(c) a translating, accelerating disk body: the graph replay equals the eager step bit for bit (eager and graph both go through the lowered polygon; the body state is integrated on the device);
(d) the disk sweeps through the water at 6 m/s (Ma 0.14): the run is stable and no particle gets inside the (analytic) disk (the wall pressure alone also keeps them out in this setup: not a test of the impulse);
(e) the no-penetration impulse for an implicit moving disk: particles 0.1 dx from the disk surface closing on it get the RELATIVE normal velocity (1 - f) v_n,rel with f = 3 - 4 clip(1/2 + d/dp, 1/4, 1), the
    wall velocity taken at the contact point of the translating disk; receding particles are untouched; with the disk at rest the same absolute velocities give a different result (negative control).
"""
import math

import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.scene.implicitBodies import DiskBody
from warpSPHBoundaries.scene.scene import Body, ImplicitRep, Scene, SdfRep, SurfaceRep
from warpSPHBoundaries.sim import dfsph2d
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA")
DEV = "cuda:0"
F64 = torch.float64
G = 9.81


def disk_rep(kind, c, R, dx):
    if kind == "implicit":
        return ImplicitRep(DiskBody(center=(0.0, 0.0), radius=R))
    if kind == "polygon":
        return SurfaceRep.regularPolygon((0.0, 0.0), R, max(24, int(math.ceil(2 * math.pi * R / (4 * dx / 16)))))
    lo, hi = np.array([-R - 0.3, -R - 0.3]), np.array([R + 0.3, R + 0.3])
    return SdfRep.fromFunction(lambda p: torch.as_tensor(np.linalg.norm(p.numpy(), axis=1) - R, dtype=F64), lo, hi, dx / 4)      # body frame: the disk is at the origin


def make(kind, dp=0.03, R=0.123, center=(0.0, -0.15), cfg=None, motion=None, Hwater=0.5, L=1.2, Htank=0.8, layers=4):
    """a water tank on a lattice with a disk of radius R: the lattice is cut at R + `layers` dp and `layers` conforming particle rings (radius R + (k + 1/2) dp, arc spacing dp, mass as the lattice) fill
    the support of the wall: a lattice cut by a circle has first-layer distances between 0.5 and 1.5 dp, which is a packing error of the particle sampling, not of the wall integrals (sampling quality is outside this test)."""
    c0 = 20.0 * math.sqrt(G * Hwater)
    nx, ny = int(round(L / dp)), int(round(Hwater / dp))
    X, Y = np.meshgrid(-L / 2 + dp * (np.arange(nx) + 0.5), -Htank / 2 + dp * (np.arange(ny) + 0.5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    pos = pos[np.linalg.norm(pos - np.array(center), axis=1) > R + layers * dp]
    rings = []
    for k in range(layers):
        r = R + (k + 0.5) * dp
        n = int(round(2 * math.pi * r / dp))
        a = 2 * math.pi * (np.arange(n) + 0.5 * (k % 2)) / n
        rings.append(np.stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a)], 1))
    pos = np.vstack([pos] + rings)
    rho = 1.0 + G * np.clip(-Htank / 2 + Hwater - pos[:, 1], 0.0, None) / c0 ** 2
    tank = dfsph2d.domain_scene("surface", (-L / 2, -Htank / 2), (L / 2, Htank / 2), 4 * dp, DEV).bodies[0]
    disk = Body(bodyId=1, center=center, reps=[disk_rep(kind, center, R, dp)])
    cfg = cfg or DeltaSPHConfig(gravity=(0.0, -G), c0=c0, noPen="impulse", shifting=True)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), rho, dp, Scene([tank, disk], DEV), cfg, DEV)
    if motion:
        sim.scene.bodies[1].linearVelocity = torch.tensor(motion[0], dtype=F64, device=DEV)
        sim.scene.bodies[1].linearAcceleration = torch.tensor(motion[1], dtype=F64, device=DEV)
    return sim


def rel(a, b):
    return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-300)


def test_implicit_equals_the_polygon_and_sdf_is_close():
    out = {}
    for kind in ("polygon", "implicit", "sdf"):
        sim = make(kind)
        v = torch.as_tensor(np.random.default_rng(0).normal(0, 0.2, sim.x.shape), dtype=F64, device=DEV)
        out[kind] = sim.rhs(sim.x, v, sim.rho, want_forces=True)
        assert sim.scene.bodies[1].fusedReps(sim.H, DEV) is not None and sim._wall_state(sim.x, sim.rho)[3].__class__.__name__ == "FusedWall"
    for k in range(3):
        assert rel(out["implicit"][k], out["polygon"][k]) < 1e-9, k
        assert rel(out["sdf"][k], out["polygon"][k]) < 5e-3, k


@pytest.mark.parametrize("dp", [0.03, 0.02])
def test_buoyancy_of_the_submerged_disk(dp):
    R = 0.123
    sim = make("implicit", dp=dp, R=R)
    _, _, loads = sim.rhs(sim.x, sim.v, sim.rho, want_forces=True)
    Fy, Fx = float(loads[0, 1, 1]), float(loads[0, 1, 0])
    buoy = G * math.pi * R * R
    assert abs(Fy - buoy) / buoy < 0.03, (Fy, buoy)                                       # measured 1.001 - 1.011 at dp = 0.03 / 0.02 / 0.015 (lattice cut by the circle instead of the rings: 0.6 - 0.78)
    assert abs(Fx) < 0.01 * buoy, Fx
    assert float(loads[0, 0, 1]) < 0                                                       # the tank: the weight of the water (and the buoyancy reaction) pushes the floor down


@pytest.mark.parametrize("kind", ["implicit", "sdf"])
def test_moving_disk_graph_equals_eager(kind):
    def go(graph):
        cfg = DeltaSPHConfig(gravity=(0.0, -G), c0=20.0 * math.sqrt(G * 0.5), noPen="impulse", shifting=True, graphStep=graph)
        sim = make(kind, cfg=cfg, motion=((0.15, 0.0), (0.0, 0.05)))
        for _ in range(25):
            sim.step()
        return sim
    a, b = go(False), go(True)
    assert b._graphed and b._graphed.stats["replays"] >= 20
    for name in ("x", "v", "rho", "wallLoads"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    ba, bb = a.scene.bodies[1], b.scene.bodies[1]
    assert torch.equal(ba.center, bb.center) and float(ba.center[0]) > 0.0


def test_fluid_stays_out_of_the_swept_disk():
    cfg = DeltaSPHConfig(gravity=(0.0, -G), c0=20.0 * math.sqrt(G * 0.5), noPen="impulse", shifting=True)
    sim = make("implicit", cfg=cfg, motion=((6.0, 0.0), (0.0, 0.0)))
    dmin = 1e9
    for _ in range(100):
        sim.step()
        d, _, _ = sim.scene.signed_distance(sim.x, body=1)
        dmin = min(dmin, float(d.min()))
    assert float(sim.scene.bodies[1].center[0]) > 0.2 and bool(torch.isfinite(sim.x).all()) and float(sim.v.abs().max()) < 20.0
    assert dmin > -0.3 * sim.dx, dmin / sim.dx


def test_noPen_impulse_of_an_implicit_moving_disk():
    sim = make("implicit")
    b = sim.scene.bodies[1]
    U = torch.tensor([1.0, 0.5], dtype=F64, device=DEV)
    b.linearVelocity = U.clone()
    c = b.center
    th = torch.linspace(0.3, 6.0, 12, dtype=F64, device=DEV)
    nrm = torch.stack([th.cos(), th.sin()], 1)
    R = 0.123
    x0 = c + (R + 0.1 * sim.dx) * nrm
    rows = torch.arange(len(th), device=DEV) * 7                                           # twelve existing particles are moved next to the disk
    sim.x = sim.x.clone()
    sim.x[rows] = x0
    vn = -3.0
    closing = torch.zeros(len(th), dtype=torch.bool, device=DEV)
    closing[::2] = True
    v = torch.zeros_like(sim.v)
    v[rows] = U + torch.where(closing, vn, -vn)[:, None] * nrm * torch.ones((len(th), 1), dtype=F64, device=DEV)      # relative normal velocity vn (closing) / -vn (receding)
    sim.v = v.clone()
    f = 3.0 - 4.0 * min(max(0.5 + 0.1 * sim.dx / sim.dx, 0.25), 1.0)
    cnt = sim.no_penetration()
    assert int(cnt) == int(closing.sum())
    rel = ((sim.v[rows] - U) * nrm).sum(1)
    assert float((rel[closing] - (1 - f) * vn).abs().max()) < 1e-9, (rel[closing], (1 - f) * vn)
    assert float((rel[~closing] + vn).abs().max()) < 1e-12                                # receding: untouched
    sim.v = v.clone()                                                                      # negative control: the same velocities against a disk at rest are all closing or not by the absolute law
    b.linearVelocity = torch.zeros(2, dtype=F64, device=DEV)
    sim.no_penetration()
    assert float(((sim.v[rows] - v[rows]).abs().max())) > 0.5 and float((((sim.v[rows] - 0.0) * nrm).sum(1)[closing] - (1 - f) * ((v[rows] * nrm).sum(1)[closing])).abs().max()) < 1e-9
