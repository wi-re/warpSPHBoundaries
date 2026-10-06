"""Phase 3 of docs/plan-wall-evaluation.md: the fluid-fluid terms of `DeltaSPH2D` from the warpSPH modules and the pair kernels of `sim/fluidkernels.py` (cfg.fluidWarp) against the torch pair sums.

(a) the four module terms (continuity, fourtakas2019 density diffusion, Antuono pressure force, alpha viscosity) on a jittered lattice with random velocities / densities / surface flags: 1e-10 of the largest value
    (float64: different summation order only; the viscosity needs the xi compensation of `fluidwarp.py`, without it the difference is 2e-8);
(b) the detector / shifting sums (cover vector, neighbour count, Mf, shift raw sum, cone count, dilation, grad lambda, min dot): 1e-10 of the scale, on a jittered lattice (no pair sits exactly at r = H, where the
    torch distance matrix and the kernel's own r may round differently);
(c) the solver, dam break and sloshing, 100 steps with fluidWarp True / False: positions 1e-10, velocities 1e-9 (the harness `check --cfg fluidWarp=true` is the bit-level gate: OVERALL PASS).
Skipped in float32 mode (tolerances are float64 contracts).
"""
import math

import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.sim import cases
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.fluidwarp import FluidWarp
from warpSPHBoundaries.sim.pairs import neighbor_pairs

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64
pytestmark = pytest.mark.skipif(real != wp.float64, reason="float64 contracts")


def jittered(device, n=30, seed=0):
    dx = 0.01
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(dx * (np.arange(n) + 0.5), dx * (np.arange(n) + 0.5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1) + rng.normal(0, 0.1 * dx, (n * n, 2))
    cfg = DeltaSPHConfig(gravity=(0.0, -9.81), c0=20.0, fluidWarp=True)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, None, cfg, device)
    sim.v = torch.as_tensor(rng.normal(0, 0.3, pos.shape), dtype=F64, device=device)
    sim.rho = torch.as_tensor(1.0 + rng.normal(0, 0.003, len(pos)), dtype=F64, device=device)
    return sim


def close(a, b, tol=1e-10):
    scale = max(float(b.abs().max()), 1e-300)
    assert float((a - b).abs().max()) <= tol * scale, (float((a - b).abs().max()), scale)


def torch_pairs(sim, x, rho):
    i, j, r = neighbor_pairs(x, sim.Hvec)
    nz = i != j
    d = x[i] - x[j]
    gW = torch.where(nz[:, None], sim.dW(r, sim.H)[:, None] * d / r.clamp(min=1e-300)[:, None], torch.zeros_like(d))
    return i, j, r, nz, d, gW, sim.m / rho


@pytest.mark.parametrize("device", DEVICES)
def test_module_terms_equal_the_torch_pair_sums(device):
    sim = jittered(device)
    x, v, rho, cfg, H = sim.x, sim.v, sim.rho, sim.cfg, sim.H
    i, j, r, nz, d, gW, V = torch_pairs(sim, x, rho)
    fw = FluidWarp(sim)
    ps, adj = fw.state(x, v, rho)
    close(fw.continuity(ps, adj, v), -rho * sim._sum(V[j] * ((v[j] - v[i]) * gW).sum(1), i))
    tot = (rho[j] - rho[i]) + cfg.rho0 * (d @ sim.g) / cfg.c0 ** 2
    psi = -2.0 * tot[:, None] * d / (r * (r + 1e-14 * H)).clamp(min=1e-300)[:, None]
    close(fw.density_diffusion(ps, adj), cfg.delta * H * cfg.c0 / sim.xi * sim._sum(torch.where(nz, V[j] * (psi * gW).sum(1), torch.zeros_like(r)), i))
    P = cfg.c0 ** 2 * (rho - cfg.rho0)
    surf = torch.as_tensor(np.random.default_rng(1).random(len(x)) < 0.25, device=device)
    s = torch.where((P >= 0) | surf, torch.ones_like(P), -torch.ones_like(P))
    close(fw.pressure(ps, adj, P, surf), -sim._sum((V[j] * (P[j] + s[i] * P[i]))[:, None] * gW, i) / rho[:, None])
    mu = ((v[i] - v[j]) * d).sum(1) / (r * r + 1e-14 * H * H)
    fac = cfg.alpha * cfg.c0 * H / sim.xi
    close(fw.viscosity(ps, adj, v), fac * sim._sum(torch.where(nz, V[j] / (0.5 * (rho[i] + rho[j])) * mu, torch.zeros_like(r))[:, None] * gW, i))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("skin", [1.0, 1.25])
def test_pair_kernels_equal_the_torch_pair_sums(device, skin):
    sim = jittered(device)
    x, rho, cfg, H = sim.x, sim.rho, sim.cfg, sim.H
    i, j, r, nz, d, gW, V = torch_pairs(sim, x, rho)
    fw = FluidWarp(sim, verletScale=skin)
    ps, adj = fw.state(x, sim.v, rho)
    fk = fw.kernels(adj, x, rho)
    p1 = fk.pass1()
    ii, jj, rr = i[nz], j[nz], r[nz]
    unit = (x[ii] - x[jj]) / rr.clamp(min=1e-300)[:, None]
    close(p1["C"], sim._sum(unit, ii))
    close(p1["nAll"], sim._sum(torch.ones_like(rr), ii))
    close(p1["Mf"], torch.zeros((len(x), 2, 2), dtype=F64, device=device).index_add_(0, i, V[j][:, None, None] * (-d)[:, :, None] * gW[:, None, :]))
    w0 = sim._w0
    coef = 0.5 * sim.m / (rho[i] + rho[j]) * (1.0 + cfg.shiftR * (sim.W(r, H) / w0) ** 4)
    close(p1["raw"], sim._sum(coef[:, None] * gW, i))
    c = p1["C"] / p1["C"].norm(dim=1, keepdim=True).clamp(min=1e-300)
    half = cfg.barecascoThreshold / 2
    inCone = torch.acos((-(unit * c[ii]).sum(1)).clamp(-1.0, 1.0)) <= half
    close(fk.cone_count(c, half), sim._sum(inCone.to(F64), ii))
    surf = torch.as_tensor(np.random.default_rng(2).random(len(x)) < 0.3, device=device)
    assert torch.equal(fk.dilate(surf), sim._sum(surf[j].to(F64), i) > 0.5)
    lam = torch.as_tensor(np.random.default_rng(3).random(len(x)), dtype=F64, device=device)
    close(fk.lam_gradient(lam), sim._sum((V[j] * (lam[j] - lam[i]))[:, None] * gW, i))
    F = torch.as_tensor(np.random.default_rng(4).random(len(x)) < 0.6, device=device)
    nrm = torch.as_tensor(np.random.default_rng(5).normal(size=(len(x), 2)), dtype=F64, device=device)
    nrm = nrm / nrm.norm(dim=1, keepdim=True)
    both = F[i] & F[j] & nz
    dots = torch.where(both, (nrm[i] * nrm[j]).sum(1), torch.full_like(r, float("inf")))
    ref = torch.full((len(x),), float("inf"), dtype=F64, device=device).scatter_reduce(0, i, dots, reduce="amin", include_self=False)
    got = fk.min_dot(F, nrm)
    fin = torch.isfinite(ref)
    assert torch.equal(fin, got < 1e299) and float((got[fin] - ref[fin]).abs().max()) <= 1e-13


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["dambreak", "sloshing"])
def test_solver_state_does_not_depend_on_the_fluid_path(device, case):
    res = {}
    for fw in (False, True):
        if case == "dambreak":
            sim = cases.marrone_dambreak(nx=30, shifting=True, noPen="impulse", device=device, fluidWarp=fw)[0]
        else:
            sim = cases.sloshing_tank(nx=40, device=device, fluidWarp=fw)[0]
        for _ in range(100):
            sim.step()
        res[fw] = (sim.x.clone(), sim.v.clone(), sim.rho.clone())
    assert float((res[True][0] - res[False][0]).abs().max()) <= 1e-10
    assert float((res[True][1] - res[False][1]).abs().max()) <= 1e-9
    assert float((res[True][2] - res[False][2]).abs().max()) <= 1e-10
