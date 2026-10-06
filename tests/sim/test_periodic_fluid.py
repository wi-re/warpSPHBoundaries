"""Step 1a of docs/plan-next-steps.md: periodic fluid pairs on raw (never wrapped) positions.

(a) `neighbor_pairs` with a `Periodic` box equals the brute-force minimum-image pair list, in the dense branch and in the cell-list branch (also in a box of < 3 cells), for positions that are any
    integer number of box lengths out of the box (a different offset per particle);
(b) the fluid terms of the solver (`rhs`, and a few steps with shifting): invariant under a translation by any vector and under an integer box offset per particle, the stored positions are bit-identical
    after `rhs`, and a step moves every particle by the physical displacement only (no jump of a box length, the offset survives the step);
(c) the warpSPH-module path (cfg.fluidWarp) equals the torch oracle on the periodic box;
(d) a particle that leaves the box keeps its row and its sums continuous: a uniform stream through the seam leaves the state unchanged.
Skipped in float32 mode (tolerances are float64 contracts).
"""
import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.pairs import Periodic, min_image, neighbor_pairs, pair_delta

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64
pytestmark = pytest.mark.skipif(real != wp.float64, reason="float64 contracts")


def brute(pos, h, per):
    d = min_image(pos[:, None, :] - pos[None, :, :], per)
    r = d.norm(dim=2)
    i, j = torch.nonzero(r <= 0.5 * (h[:, None] + h[None, :]), as_tuple=True)
    return set(zip(i.tolist(), j.tolist())), r


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n,box,flags", [(900, (1.0, 1.0), (True, True)), (3600, (1.0, 0.7), (True, True)), (3600, (1.0, 0.9), (True, False)), (2500, (0.4, 0.4), (True, True))])
def test_neighbor_pairs_are_the_minimum_image_pairs(device, n, box, flags):
    rng = np.random.default_rng(1)
    per = Periodic((0.0, 0.0), box, flags)
    pos = torch.as_tensor(rng.uniform(0, 1, (n, 2)) * np.array(box), dtype=F64, device=device)
    h = torch.full((n,), 0.9 * min(box) / 2 if n == 2500 else 0.05, dtype=F64, device=device)
    ref, _ = brute(pos, h, per) if n <= 3600 else (None, None)
    for k in (0, 1, 3):                                                                    # raw positions any number of box lengths out, a different count per particle
        off = torch.as_tensor(rng.integers(-k, k + 1, (n, 2)) * np.array(box), dtype=F64, device=device) * torch.tensor([float(f) for f in flags], dtype=F64, device=device)
        i, j, r = neighbor_pairs(pos + off, h, per)
        got = set(zip(i.tolist(), j.tolist()))
        assert got == ref and len(got) == len(i), (k, len(got), len(ref))                  # also no pair twice
        assert float((r - pair_delta(pos, i, j, per).norm(dim=1)).abs().max()) < 1e-12


def fluid(device, n=36, fluidWarp=True, offsets=None, shift=(0.0, 0.0), seed=0, **cfgkw):
    dx = 1.0 / n
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(dx * (np.arange(n) + 0.5), dx * (np.arange(n) + 0.5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1) + rng.normal(0, 0.1 * dx, (n * n, 2))
    vel = rng.normal(0, 0.3, pos.shape)
    rho = 1.0 + rng.normal(0, 0.003, len(pos))
    off = np.zeros_like(pos) if offsets is None else offsets * 1.0
    cfg = DeltaSPHConfig(gravity=(0.0, 0.0), c0=20.0, fluidWarp=fluidWarp, graphStep=False, periodic=Periodic((0.0, 0.0), (1.0, 1.0)), **cfgkw)
    sim = DeltaSPH2D(pos + off + np.array(shift), vel, rho, dx, None, cfg, device)
    sim.rho = torch.as_tensor(rho, dtype=F64, device=device)
    return sim, pos


def sums(sim):
    out = sim.rhs(sim.x, sim.v, sim.rho)
    return [o for o in (out if isinstance(out, (tuple, list)) else (out,)) if isinstance(o, torch.Tensor)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fluidWarp", [True, False])
def test_rhs_is_invariant_under_translations_and_box_offsets(device, fluidWarp):
    base, pos = fluid(device, fluidWarp=fluidWarp)
    ref = sums(base)
    rng = np.random.default_rng(5)
    offs = rng.integers(-3, 4, pos.shape).astype(float)
    for kw in (dict(shift=(0.31, -0.17)), dict(shift=(2.0, -5.0)), dict(offsets=offs)):
        sim, _ = fluid(device, fluidWarp=fluidWarp, **kw)
        x0 = sim.x.clone()
        got = sums(sim)
        assert torch.equal(sim.x, x0)                                                      # the positions are not written
        shift = torch.tensor(kw.get("shift", (0.0, 0.0)), dtype=F64, device=device)
        for a, b in zip(got, ref):
            if a.shape == b.shape and a.dtype == b.dtype:
                tol = 1e-9 * max(float(b.abs().max()), 1e-12) if kw.get("shift") != (0.31, -0.17) else 1e-6 * max(float(b.abs().max()), 1e-12)
                assert float((a - b).abs().max()) <= tol, (kw.keys(), float((a - b).abs().max()), float(b.abs().max()))


@pytest.mark.parametrize("device", DEVICES)
def test_the_warp_modules_equal_the_torch_oracle_on_the_periodic_box(device):
    a, _ = fluid(device, fluidWarp=True)
    b, _ = fluid(device, fluidWarp=False)
    for x, y in zip(sums(a), sums(b)):
        if x.shape == y.shape:
            assert float((x - y).abs().max()) <= 1e-9 * max(float(y.abs().max()), 1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fluidWarp", [True, False])
def test_steps_keep_the_raw_trajectory(device, fluidWarp):
    rng = np.random.default_rng(7)
    _, pos = fluid(device, fluidWarp=fluidWarp)
    offs = rng.integers(-2, 3, pos.shape).astype(float)
    a, _ = fluid(device, fluidWarp=fluidWarp)
    b, _ = fluid(device, fluidWarp=fluidWarp, offsets=offs)
    x0 = b.x.clone()
    for _ in range(5):
        a.step()
        b.step()
    assert float((b.x - x0).abs().max()) < 0.2                                             # no jump by a box length: the displacement is the physical one
    tolx = 1e-9
    assert float((b.x - torch.as_tensor(offs, dtype=F64, device=device) - a.x).abs().max()) <= tolx
    assert float((b.v - a.v).abs().max()) <= 1e-8


@pytest.mark.parametrize("device", DEVICES)
def test_a_uniform_stream_through_the_seam_is_stationary(device):
    sim, pos = fluid(device, n=30)
    sim.v = torch.zeros_like(sim.v) + torch.tensor([1.0, 0.5], dtype=F64, device=device)
    sim.rho = torch.ones_like(sim.rho)
    x0 = sim.x.clone()
    t = 0.0
    for _ in range(60):
        sim.step()
        t += sim.dt
    assert float((sim.v - torch.tensor([1.0, 0.5], dtype=F64, device=device)).abs().max()) < 1e-6
    d = (sim.x - x0)
    assert float((d - torch.tensor([1.0, 0.5], dtype=F64, device=device) * t).abs().max()) < 1e-6         # the raw coordinate is continuous: x(t) = x0 + u t, no wrapping, even past the seam
