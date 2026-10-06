"""Step 1c of docs/plan-next-steps.md: the prescribed-velocity band of fluid particles (`sim/pinned.py`) and the momentum-only body force (`cfg.bodyForce`).

(a) a uniform stream in an empty periodic box with a band: stationary (no acceleration, no density change), the raw coordinate advances by u t through the seam (no wrapping), count and rows unchanged;
(b) fluid at rest with the band at the free stream: the band keeps the stream exactly, the free region is accelerated towards it, nothing becomes non-finite, no particle is created or removed;
(c) a body force accelerates a uniform periodic fluid as f t without a pressure response (it is not gravity: no hydrostatic term);
(d) fibre + band + body force: graph replay equals the eager step, and the periodic offsets of the particles do not matter.
Float64 contracts.
"""
import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, Scene
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.pinned import Pinned

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64
pytestmark = pytest.mark.skipif(real != wp.float64, reason="float64 contracts")
BOX = Periodic((0.0, 0.0), (1.0, 1.0))
BAND = Pinned(slabs=((0, 0.0, 0.12),), velocity=(1.0, 0.0))                 # the slab at the seam x = 0 = 1


def lattice(n, jitter=0.0, seed=0):
    dx = 1.0 / n
    X, Y = np.meshgrid(dx * (np.arange(n) + 0.5), dx * (np.arange(n) + 0.5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    return pos + jitter * dx * np.random.default_rng(seed).normal(size=pos.shape), dx


def sim_of(device, pos, vel, dx, scene=None, **cfgkw):
    cfg = DeltaSPHConfig(gravity=(0.0, 0.0), c0=20.0, periodic=BOX, graphStep=False, **cfgkw)
    return DeltaSPH2D(pos, vel, np.ones(len(pos)), dx, scene, cfg, device, support=3.0 * dx)


@pytest.mark.parametrize("device", DEVICES)
def test_a_uniform_stream_with_a_band_is_stationary(device):
    pos, dx = lattice(30, jitter=0.05)
    sim = sim_of(device, pos, np.tile([1.0, 0.0], (len(pos), 1)), dx, pinned=BAND)
    x0, t = sim.x.clone(), 0.0
    for _ in range(40):
        sim.step()
        t += sim.dt
    assert float((sim.v - torch.tensor([1.0, 0.0], dtype=F64, device=device)).abs().max()) < 1e-9
    assert float((sim.rho - 1.0).abs().max()) < 1e-9
    assert float((sim.x - x0 - torch.tensor([1.0, 0.0], dtype=F64, device=device) * t).abs().max()) < 1e-6      # raw, unwrapped: the band crosses the seam region freely
    assert len(sim.x) == len(pos)


@pytest.mark.parametrize("device", DEVICES)
def test_the_band_holds_the_stream_and_drives_the_rest(device):
    pos, dx = lattice(30, jitter=0.05)
    ramp = Pinned(slabs=BAND.slabs, velocity=BAND.velocity, ramp=0.0)
    sim = sim_of(device, pos, np.zeros_like(pos), dx, pinned=ramp, alpha=0.1)
    for _ in range(150):
        sim.step()
    w = ramp.weight(sim.x, BOX) > 0.5
    assert int(w.sum()) > 50
    assert float((sim.v[w] - torch.tensor([1.0, 0.0], dtype=F64, device=device)).abs().max()) < 1e-12        # the band IS the stream
    free = ~w
    assert bool(torch.isfinite(sim.v).all()) and bool(torch.isfinite(sim.rho).all())
    assert float(sim.v[free, 0].mean()) > 0.02                                                                 # the viscosity carries the stream into the rest (the band moved on with the flow)
    assert len(sim.x) == len(pos)


@pytest.mark.parametrize("device", DEVICES)
def test_the_body_force_is_not_gravity(device):
    pos, dx = lattice(30, jitter=0.05)
    sim = sim_of(device, pos, np.zeros_like(pos), dx, bodyForce=(2.0, 0.0))
    t = 0.0
    for _ in range(30):
        sim.step()
        t += sim.dt
    assert float((sim.v[:, 0] - 2.0 * t).abs().max()) < 1e-8 and float(sim.v[:, 1].abs().max()) < 1e-8
    assert float((sim.rho - 1.0).abs().max()) < 1e-8                                                          # no pressure response, no hydrostatic density diffusion


def fibre_sim(device, offsets=None, graph=False):
    pos, dx = lattice(36, jitter=0.03)
    keep = np.linalg.norm(pos - 0.5, axis=1) >= 0.14 + 0.5 * dx
    pos = pos[keep]
    scene = Scene([Body(bodyId=0, center=(0.5, 0.5), reps=[DiskArrayRep([(0.0, 0.0)], [0.14])])], device)
    off = 0.0 if offsets is None else offsets(pos)
    sim = sim_of(device, pos + off, np.tile([0.5, 0.0], (len(pos), 1)), dx, scene, pinned=BAND, bodyForce=(0.3, 0.0))
    sim.cfg.graphStep = graph
    return sim, pos


@pytest.mark.parametrize("device", DEVICES)
def test_fibre_band_and_body_force_graph_equals_eager_and_offsets_do_not_matter(device):
    a, pos = fibre_sim(device)
    offs = np.random.default_rng(1).integers(-2, 3, pos.shape).astype(float)
    b, _ = fibre_sim(device, offsets=lambda p: offs, graph=True)
    for _ in range(8):
        a.step()
        b.step()
    assert float((b.x - torch.as_tensor(offs, dtype=F64, device=device) - a.x).abs().max()) <= 1e-9
    assert float((a.v - b.v).abs().max()) <= 1e-8
    assert float((a.wallLoads - b.wallLoads).abs().max()) <= 1e-8 * max(float(a.wallLoads.abs().max()), 1e-9)
    assert b._graphed not in (None, False)                                                                     # the graph step was used
