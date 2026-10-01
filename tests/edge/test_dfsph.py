"""DFSPH2D on the scene boundary layer: operators vs warpSPHCore, lattice calibration, representation independence of the dynamics, agreement with the
compiled omniSPH reference (skipped when `omnySPH` is not importable)."""
import os
import sys

import numpy as np
import pytest
import torch
import warp as wp

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation, DomainDescription
import warpSPHCore as core
from edgebound import dfsph2d as D

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
TD = torch.float64


def lattice(nx, ny, dx, dy, x0=0.0, y0=0.0):
    X, Y = np.meshgrid(x0 + dx * (np.arange(nx) + 0.5), y0 + dy * (np.arange(ny) + 0.5), indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


@pytest.mark.parametrize("device", DEVICES)
def test_pair_sums_match_warpoperation(device):
    """density and the symmetric pressure gradient of the torch pair sums equal `warpSPHCore.warpOperation` (same kernel, same support)."""
    rng = np.random.default_rng(0)
    dx = 0.01
    pos = lattice(14, 14, dx, dx) + rng.normal(0, 0.1 * dx, (196, 2))
    h = 2.5 * dx
    t = lambda a: torch.as_tensor(a, dtype=TD, device=device)
    n = len(pos)
    V = np.full(n, dx * dx)
    p = rng.uniform(0, 1, n)
    sim = D.DFSPH2D(pos, np.zeros_like(pos), V, np.full(n, h), None, D.DFSPHConfig(), device)
    sim._prepare()
    rho = sim._sum(sim.V[sim.pj] * sim.W)
    sim.rho = rho
    acc = sim._fluid_accel(t(p))
    # warpSPHCore reference
    t32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)            # warpOperation computes in float32
    ps = ParticleState(positions=t32(pos), supports=t32(np.full(n, h)), masses=t32(V), kinds=torch.zeros(n, dtype=torch.int32, device=device), densities=rho.float())
    lo, hi = pos.min(0) - 2 * h, pos.max(0) + 2 * h
    domain = DomainDescription(t32(lo), t32(hi), torch.zeros(2, dtype=torch.bool, device=device), 2)
    props_d = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Density, operationMode=OperationDirection.AllToAll)
    rho_ref = core.warpOperation(ps, props_d, domain)
    np.testing.assert_allclose(rho.cpu().numpy(), rho_ref.double().cpu().numpy(), rtol=2e-5)
    props_g = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Gradient, gradientMode=GradientScheme.Symmetric,
                                  operationMode=OperationDirection.AllToAll)
    g = core.warpOperation(ps, props_g, domain, queryValues=t32(p), referenceValues=t32(p))
    # warp Symmetric: m_j rho_i (p_i/rho_i^2 + p_j/rho_j^2) grad W; our acceleration is -(1/rho_i) of it
    np.testing.assert_allclose(acc.cpu().numpy(), (-g / rho[:, None]).double().cpu().numpy(), rtol=1e-3, atol=3e-4 * float(acc.abs().max()))


@pytest.mark.parametrize("device", DEVICES)
def test_lattice_calibration_is_a_rest_state(device):
    dx, dy, h = 0.0089, 0.0088, 0.02236
    cal = D.lattice_calibration(dx, dy, h)
    pos = lattice(40, 12, dx, dy, x0=0.1, y0=0.1)
    lo = pos.min(0) - np.array([cal["dwallX"], cal["dwallY"]])
    hi = np.array([pos[:, 0].max() + cal["dwallX"], 0.6])
    for kind in ["surface", "volume", "sdf"]:
        sc = D.domain_scene(kind, lo, hi, h, device)
        sim = D.DFSPH2D(pos, np.zeros_like(pos), cal["V"], np.full(len(pos), h), sc, D.DFSPHConfig(wallMass=cal["mu"]), device)
        sim._prepare()
        rho = sim._sum(sim.V[sim.pj] * sim.W) + sim.lam
        x = sim.x
        bulk = (x[:, 0] > 0.1 + 6 * dx) & (x[:, 0] < 0.1 + 34 * dx) & (x[:, 1] > 0.1 + 4 * dy) & (x[:, 1] < 0.1 + 8 * dy)
        assert abs(float(rho[bulk].mean()) - 1) < 1e-9
        row0 = (x[:, 1] < 0.1 + dy) & (x[:, 0] > 0.1 + 6 * dx) & (x[:, 0] < 0.1 + 34 * dx)
        assert abs(float(rho[row0].mean()) - 1) < 3e-4


@pytest.mark.parametrize("device", DEVICES)
def test_dynamics_independent_of_the_representation(device):
    """30 steps of a settling block: volume slabs, surface loop and tier-3 SDF (+ corner fallback) give the same trajectories (3e-13 / 3e-13 / 1e-5)."""
    dx, dy, h = 0.0089, 0.0088, 0.02236
    cal = D.lattice_calibration(dx, dy, h)
    pos = lattice(30, 10, dx, dy, x0=0.1, y0=0.1)
    lo = pos.min(0) - np.array([cal["dwallX"], cal["dwallY"]])
    hi = np.array([pos[:, 0].max() + cal["dwallX"], 0.5])
    xs = {}
    for kind in ["surface", "volume", "sdf"]:
        sc = D.domain_scene(kind, lo, hi, h, device)
        sim = D.DFSPH2D(pos, np.zeros_like(pos), cal["V"], np.full(len(pos), h), sc, D.DFSPHConfig(wallMass=cal["mu"]), device)
        for _ in range(30):
            sim.step()
        xs[kind] = sim.x.cpu().numpy()
    assert np.abs(xs["volume"] - xs["surface"]).max() < 1e-10
    assert np.abs(xs["sdf"] - xs["surface"]).max() < 1e-4
    assert np.abs(xs["surface"] - pos).max() > 1e-6          # something actually happened


def test_agrees_with_omnisph_reference():
    sys.path.insert(0, os.path.expanduser("~/dev/omniSPH/omnySPH/src"))
    try:
        import omnySPH  # noqa: F401
    except Exception:
        pytest.skip("omnySPH not importable")
    scratch = os.path.join(os.path.dirname(__file__), "..", "..", ".tmp", "omni")
    os.makedirs(scratch, exist_ok=True)
    link = os.path.join(scratch, "cfg")
    if not os.path.exists(link):
        os.symlink(os.path.expanduser("~/dev/omniSPH/cfg"), link)
    old = os.getcwd()
    os.chdir(scratch)
    try:
        from edgebound import dfsph_ref as R
        r = 0.005
        fmin, fmax = (0.02, 0.02), (0.5, 0.15)
        c = R.omni_case(r, fmin, fmax, top=0.4)
        sp = np.array([c["dx"], c["dy"]])
        lo = np.array(fmin) - sp
        hi = np.array([fmax[0], 0.4]) + np.array([sp[0], 0])
        import omnySPH as O
        so = O.SPHSimulation(R._yaml(r, fmin, fmax, lo - c["epsAdj"], hi + c["epsAdj"], 0.06))
        dev = DEVICES[0]
        sc = D.domain_scene("volume", lo, hi, float(c["h"][0]), dev)
        sim = D.DFSPH2D(c["x"], np.zeros_like(c["x"]), c["V"], c["h"], sc, D.DFSPHConfig(), dev)
        for _ in range(120):
            so.timestep()
            sim.step()
        st = R.omni_state(so, c["n"])
        xo, xs = st["x"], sim.x.cpu().numpy()
        # a settling block: bulk statistics must agree closely (particle-by-particle agreement is not expected: different wall closures)
        assert abs(xo[:, 1].mean() - xs[:, 1].mean()) < 1e-3
        assert abs(xo[:, 0].mean() - xs[:, 0].mean()) < 1e-3
        rms = lambda v: float(np.sqrt((np.asarray(v) ** 2).sum(1).mean()))
        assert abs(rms(st["v"]) / rms(sim.v.cpu().numpy()) - 1) < 0.35
        assert np.sqrt(((xo - xs) ** 2).sum(1).mean()) < 3e-3
    finally:
        os.chdir(old)
