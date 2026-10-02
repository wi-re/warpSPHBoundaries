"""DeltaSPH2D (delta+-SPH on the scene boundary layer): the DDT is exactly zero on a hydrostatic field, the wall continuity term has the right sign, a wall never pulls (ceiling), the Barecasco switch
flags the free surface and not the walls, hydrostatic tank at rest, representation independence."""
import math

import numpy as np
import pytest
import torch
import warp as wp

from edgebound import deltasph2d as D
from edgebound.dfsph2d import domain_scene

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64


def small_tank(device, dp=0.04, L=0.8, Ht=0.6, Hw=0.3, domain="surface", **cfgkw):
    sim, info = D.hydrostatic_tank(dp=dp, L=L, Htank=Ht, Hwater=Hw, domain=domain, device=device)
    sim.cfg.__dict__.update(cfgkw)
    return sim, info


@pytest.mark.parametrize("device", DEVICES)
def test_density_diffusion_is_zero_on_a_hydrostatic_field(device):
    """fourtakas2019: psi_ij = 0 pair by pair for the exact hydrostatic density, truncated stencils at the surface included (no wall term: fluid-to-fluid only)."""
    sim, info = small_tank(device)
    sim.scene, sim.nb = None, 0
    c0, g = sim.cfg.c0, 9.81
    depth = (info["bed"] + info["Hwater"] - sim.x[:, 1]).clamp(min=0)
    rho = 1.0 * (1.0 + g * depth / c0 ** 2)
    sim.cfg.viscosity = False
    acc, drho, _ = sim.rhs(sim.x, torch.zeros_like(sim.x), rho)
    assert float(drho.abs().max()) < 1e-10
    sim.cfg.ddt = False
    sim.cfg.delta = 0.0
    off = sim.rhs(sim.x, torch.zeros_like(sim.x), rho)[1]
    assert float((drho - off).abs().max()) < 1e-10


@pytest.mark.parametrize("device", DEVICES)
def test_wall_continuity_compresses_toward_and_expands_away_from_the_wall(device):
    """free-slip mirror: a particle moving toward the bed has drho/dt > 0, away < 0, along it (tangential) 0."""
    sim, info = small_tank(device)
    dp, bed = sim.dx, info["bed"]
    x = torch.tensor([[0.0, bed + 0.5 * dp]], dtype=F64, device=device)
    rho = torch.ones(1, dtype=F64, device=device)
    sim.cfg.ddt = False
    sim.cfg.viscosity = False
    sim.Hvec = torch.full((1,), sim.H, dtype=F64, device=device)
    sim.kinds = torch.zeros(1, dtype=torch.int32, device=device)
    out = {}
    for name, v in (("toward", (0.0, -1.0)), ("away", (0.0, 1.0)), ("along", (1.0, 0.0))):
        out[name] = float(sim.rhs(x, torch.tensor([v], dtype=F64, device=device), rho)[1][0])
    assert out["toward"] > 1.0 and out["away"] < -1.0
    assert abs(out["toward"] + out["away"]) < 1e-9 and abs(out["along"]) < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_a_wall_above_a_particle_does_not_pull_it(device):
    """isolated particle at the ceiling at rest and at rest density: only gravity acts with the p_b >= 0 clamp; without it the hydrostatic wall pressure p_i - rho g d < 0 is a suction that holds the particle up."""
    res = {}
    for clamp in (True, False):
        sim, info = small_tank(device, clampWallPressure=clamp)
        top = info["bed"] + 0.6
        x = torch.tensor([[0.0, top - 0.5 * sim.dx]], dtype=F64, device=device)
        rho = torch.ones(1, dtype=F64, device=device)
        sim.cfg.ddt = False
        sim.cfg.viscosity = False
        sim.Hvec = torch.full((1,), sim.H, dtype=F64, device=device)
        sim.kinds = torch.zeros(1, dtype=torch.int32, device=device)
        res[clamp] = float(sim.rhs(x, torch.zeros(1, 2, dtype=F64, device=device), rho)[0][0, 1])
    assert abs(res[True] + 9.81) < 1e-9
    assert res[False] > -9.81 + 1.0                 # the unclamped wall holds up a good part of the weight


@pytest.mark.parametrize("device", DEVICES)
def test_free_surface_detector_flags_the_top_row_and_not_the_walls(device):
    sim, info = small_tank(device)
    sim.rhs(sim.x, torch.zeros_like(sim.x), sim.rho)
    top = sim.x[:, 1] > info["bed"] + info["Hwater"] - 0.9 * sim.dx
    assert bool(sim.surface[top].all())
    assert int(sim.surface.sum()) == int(top.sum())


@pytest.mark.parametrize("device", DEVICES)
def test_hydrostatic_tank_stays_at_rest_and_the_walls_carry_the_weight(device):
    sim, info = small_tank(device)
    forces = []
    for _ in range(150):
        sim.step()
        forces.append(sim.wallForce.sum(0).cpu().numpy())
    weight = sim.m * len(sim.x) * 9.81
    assert float(sim.v.norm(dim=1).max()) < 0.05
    assert abs(np.mean(forces[-50:], 0)[1] / weight + 1.0) < 0.03           # force of the fluid on the walls points down and equals its weight
    assert abs(np.mean(forces[-50:], 0)[0]) < 0.02 * weight
    assert 0.99 < float(sim.rho.min()) and float(sim.rho.max()) < 1.02


@pytest.mark.parametrize("device", DEVICES)
def test_dynamics_independent_of_the_wall_representation(device):
    xs = {}
    for kind in ("surface", "volume"):
        sim, _ = small_tank(device, domain=kind)
        for _ in range(25):
            sim.step()
        xs[kind] = sim.x.cpu().numpy()
    assert np.abs(xs["surface"] - xs["volume"]).max() < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_pair_list_is_complete_for_a_lattice_whose_spacing_divides_the_support(device):
    """regression: with support = 4 dx the lattice puts particles exactly on the cell-list borders; the cell index was rounded differently at insertion and at query and ~2 % of the reverse pairs went missing
    (the pairwise antisymmetric forces then no longer conserved momentum).  The pair list must equal the brute-force list of r < H, in both directions."""
    from edgebound.dfsph2d import neighbor_pairs
    sim, _ = D.hydrostatic_tank(dp=0.02, device=device)                       # the English 2.4 x 1.2 m tank: the bug needs this lattice / cell alignment
    x = sim.x
    i, j, r = neighbor_pairs(x, sim.Hvec)
    n = len(x)
    inside = r < sim.H - 1e-9
    assert int(inside.sum()) == int((torch.cdist(x, x) < sim.H - 1e-9).sum())
    keys = set((i[inside] * n + j[inside]).tolist())
    assert all((b * n + a) in keys for a, b in zip(i[inside][:3000].tolist(), j[inside][:3000].tolist()))
    sim.scene, sim.nb = None, 0
    a_ff = sim.rhs(x, torch.zeros_like(x), sim.rho)[0] - sim.g[None]       # fluid-fluid pressure force without gravity
    assert float((sim.m * a_ff.sum(0)).abs().max()) < 1e-12                # momentum conservation
