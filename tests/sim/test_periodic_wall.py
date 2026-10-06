"""Step 1b of docs/plan-next-steps.md in the solver: a fibre (or a bundle) in a periodic box, raw positions.

(a) the wall terms and the loads of `rhs` are invariant under an integer number of box lengths per particle (the raw positions are any trajectory), the positions are not written;
(b) a fluid at rest, symmetric about a single fibre, exerts no net force on it (periodic pair sums + periodic wall integrals are consistent: rung 1 of the validation ladder);
(c) a few steps (graph replay and eager) keep the offsets: the raw trajectory minus the offsets equals the run without offsets;
(d) the loads are those of the one body: the sum over the images of the load of the fibre is the load the solver books; a fibre at the seam is the same problem as the fibre in the centre (translation of the
    whole problem by a vector is a symmetry of the periodic setup).
Float64 contracts.
"""
import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.scene.implicitBodies import DiskBody
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, ImplicitRep, Scene
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64
pytestmark = pytest.mark.skipif(real != wp.float64, reason="float64 contracts")
BOX = Periodic((0.0, 0.0), (1.0, 1.0))


def setup(device, bundle=False, n=40, offsets=None, centre=(0.5, 0.5), jitter=0.0, vel=0.0, seed=0, **cfgkw):
    dx = 1.0 / n
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(dx * (np.arange(n) + 0.5), dx * (np.arange(n) + 0.5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    if bundle:
        centres, radii = [(0.0, 0.0), (0.33, 0.2), (-0.3, 0.3), (0.1, -0.35)], [0.1, 0.07, 0.08, 0.06]
    else:
        centres, radii = [(0.0, 0.0)], [0.14]
    c = np.array(centre)
    cut = np.array((0.5, 0.5))                                                                  # the fibre is cut out of the lattice around (0.5, 0.5); `centre` only places the body
    dist = np.min([np.linalg.norm(pos - (cut + np.array(m) + [i, j]), axis=1) - r for m, r in zip(centres, radii) for i in (-1, 0, 1) for j in (-1, 0, 1)], axis=0)
    pos = pos[dist >= 0.5 * dx]
    pos = pos + jitter * dx * rng.normal(size=pos.shape)
    v = vel * rng.normal(size=pos.shape)
    rho = np.ones(len(pos))
    rep = DiskArrayRep(centres, radii) if (bundle or True) else None
    scene = Scene([Body(bodyId=0, center=tuple(c), reps=[rep])], device)
    cfg = DeltaSPHConfig(gravity=(0.0, 0.0), c0=20.0, periodic=BOX, graphStep=False, **cfgkw)
    off = 0.0 if offsets is None else offsets(pos)
    sim = DeltaSPH2D(pos + off, v, rho, dx, scene, cfg, device, support=3.0 * dx)
    return sim, pos


def loads(sim):
    out = sim.rhs(sim.x, sim.v, sim.rho, want_forces=True)
    return out[0], out[2]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bundle", [False, True])
def test_rhs_and_loads_are_invariant_under_box_offsets(device, bundle):
    base, pos = setup(device, bundle=bundle, jitter=0.08, vel=0.2)
    acc0, ld0 = loads(base)
    rng = np.random.default_rng(3)
    offs = rng.integers(-3, 4, pos.shape).astype(float)
    sim, _ = setup(device, bundle=bundle, jitter=0.08, vel=0.2, offsets=lambda p: offs)
    x0 = sim.x.clone()
    acc, ld = loads(sim)
    assert torch.equal(sim.x, x0)
    assert float((acc - acc0).abs().max()) <= 1e-9 * float(acc0.abs().max())
    assert float((ld - ld0).abs().max()) <= 1e-9 * max(float(ld0.abs().max()), 1e-9)
    assert float(acc0.abs().max()) > 1e-3 and float(ld0.abs().max()) > 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_fluid_at_rest_around_a_fibre_exerts_no_net_force(device):
    sim, _ = setup(device, jitter=0.0)
    sim.rho = sim.rho.clone()
    acc, ld = loads(sim)
    assert float(ld[:, 0, :2].abs().max()) < 1e-9, ld                                            # (pressure, viscous) x (Fx, Fy): the lattice is symmetric about the fibre
    assert float(ld[0, 0, 2].abs()) < 1e-9                                                       # and the torque


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("graph", [False, True])
def test_steps_keep_the_offsets_and_the_loads(device, graph):
    kw = dict(jitter=0.05, vel=0.1)
    a, pos = setup(device, **kw)
    rng = np.random.default_rng(8)
    offs = rng.integers(-2, 3, pos.shape).astype(float)
    b, _ = setup(device, offsets=lambda p: offs, **kw)
    a.cfg.graphStep = b.cfg.graphStep = graph
    for _ in range(6):
        a.step()
        b.step()
    assert float((b.x - torch.as_tensor(offs, dtype=F64, device=device) - a.x).abs().max()) <= 1e-9
    assert float((a.v - b.v).abs().max()) <= 1e-8
    assert float((a.wallLoads - b.wallLoads).abs().max()) <= 1e-8 * max(float(a.wallLoads.abs().max()), 1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_translating_the_problem_is_a_symmetry(device):
    """the fibre centre at the seam: the same problem shifted by a fraction of the box (particles and body together, no wrapping): the same loads."""
    a, pos = setup(device, jitter=0.05, vel=0.1, centre=(0.5, 0.5))
    shift = np.array([0.5, 0.5])
    b, _ = setup(device, jitter=0.05, vel=0.1, centre=(1.0, 1.0), offsets=lambda p: np.broadcast_to(shift, p.shape))
    acc_a, ld_a = loads(a)
    acc_b, ld_b = loads(b)
    assert float((acc_a - acc_b).abs().max()) <= 1e-9 * float(acc_a.abs().max())
    assert float((ld_a - ld_b).abs().max()) <= 1e-9 * max(float(ld_a.abs().max()), 1e-9)
