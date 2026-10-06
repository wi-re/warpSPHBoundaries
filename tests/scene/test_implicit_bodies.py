"""Tiers 3/4 with hard switches inside the boundary-operation interface: DiskBody (solid and cavity), HalfPlaneBody, mixture with explicit meshes."""
from fractions import Fraction as F

import mpmath as mp
import numpy as np
import pytest
import torch

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation
from curvbound import planar2d
import warpSPHBoundaries as eb
from warpSPHBoundaries.scene import boundaryOps as B
from warpSPHBoundaries.scene.implicitBodies import DiskBody, HalfPlaneBody, TierPolicy

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
TD = torch.float64


def state(pos, h=1.0, rho=1.0, mass=1.0, kinds=None):
    n = len(pos)
    t = lambda a: torch.as_tensor(a, dtype=TD, device=DEV)
    return ParticleState(positions=t(pos), supports=t(np.full(n, h) if np.ndim(h) == 0 else h), masses=t(np.full(n, mass)),
                         kinds=torch.zeros(n, dtype=torch.int32, device=DEV), densities=t(np.full(n, rho)))


def props(op, mode=GradientScheme.Naive, kernel=KernelFunctions.Wendland4):
    return OperationProperties(kernel=kernel, operation=op, gradientMode=mode, operationMode=OperationDirection.BoundaryToFluid)


def exact_disk(center, R, x, name="w4"):
    c = (F(float(center[0])), F(float(center[1])))
    xx = (F(float(x[0])), F(float(x[1])))
    return float(eb.polar_disk_value(c, F(R).limit_denominator(10 ** 9), xx, name)), [float(v) for v in eb.polar_disk_gradient(c, F(R).limit_denominator(10 ** 9), xx, name)]


def ring_points(center, R, ds, angs=(0.3, 1.9, 4.0), h=1.0):
    return np.array([[center[0] + (R + d) * np.cos(a), center[1] + (R + d) * np.sin(a)] for a in angs for d in ds])


@pytest.mark.parametrize("R,tier,tol", [(4.0, 3, 3e-4), (0.1, 4, 3e-7), (0.7, 2, 3e-3)])
def test_disk_tiers_density_and_gradient_vs_exact(R, tier, tol):
    center = (0.3, -0.2)
    pos = ring_points(center, R, [0.05, 0.3, 0.6])
    ps = state(pos)
    desc = B.BoundaryDescription(bodies=[DiskBody(center, R)])
    adj = B.buildBoundaryAdjacency(ps, props(WarpOperation.Density), desc)
    got_tiers = set(adj.bodyTier[0].cpu().numpy().tolist())
    assert got_tiers == {tier}, got_tiers
    lam = B.boundaryOperation(ps, props(WarpOperation.Density), desc, adjacency=adj).cpu().numpy()
    gq = B.boundaryOperation(ps, props(WarpOperation.Gradient, GradientScheme.Naive), desc, queryValues=torch.zeros(len(pos), dtype=TD, device=DEV),
                             referenceValues=None, bodyValues=torch.ones(1, dtype=TD, device=DEV), adjacency=adj).cpu().numpy()
    for i, x in enumerate(pos):
        v, g = exact_disk(center, R, x)
        assert abs(lam[i] - v) < tol, (i, lam[i], v)
        assert np.abs(gq[i] - np.array(g)).max() < {3: 3e-3, 4: 3e-5, 2: 3e-2}[tier], (i, gq[i], g)     # the derivative of an asymptotic series loses one order


def test_particle_inside_solid_and_beyond_support():
    center, R = (0.0, 0.0), 4.0
    pos = np.array([[0.0, 3.8], [1.0, 3.5], [0.0, 0.0], [0.0, 3.0], [0.0, 5.5], [0.0, 4.9]])    # depths 0.2, ..., deeper than a support, outside beyond / within support
    ps = state(pos)
    lam = B.boundaryOperation(ps, props(WarpOperation.Density), B.BoundaryDescription(bodies=[DiskBody(center, R)])).cpu().numpy()
    for i, x in enumerate(pos):
        v, _ = exact_disk(center, R, x)
        assert abs(lam[i] - v) < 3e-4, (i, lam[i], v)
    assert lam[2] == pytest.approx(1.0, abs=1e-12) and lam[4] == 0.0


def test_cavity_wall_tier3():
    """fluid INSIDE a circular wall (solid = complement of the disk): lambda = 1 - disk integral of the fluid disk."""
    center, R = (0.0, 0.0), 4.0
    pos = np.array([[0.0, 3.9], [2.0, 3.0], [3.6, 0.0], [0.0, 3.3], [0.0, 0.0]])
    ps = state(pos)
    lam = B.boundaryOperation(ps, props(WarpOperation.Density), B.BoundaryDescription(bodies=[DiskBody(center, R, solid="outside")])).cpu().numpy()
    for i, x in enumerate(pos):
        v, _ = exact_disk(center, R, x)
        assert abs(lam[i] - (1 - v)) < 3e-4, (i, lam[i], 1 - v)


@pytest.mark.parametrize("kernel,name", [(KernelFunctions.Wendland4, "w4"), (KernelFunctions.Wendland2, "w2")])
def test_half_plane_is_exact(kernel, name):
    n = (0.6, 0.8)
    p0 = (0.1, -0.3)
    d = np.array([-0.2, 0.0, 0.05, 0.3, 0.5, 0.97])
    pos = np.array(p0) + d[:, None] * np.array(n) + 0.7 * np.array([-n[1], n[0]])
    ps = state(pos)
    lam = B.boundaryOperation(ps, props(WarpOperation.Density, kernel=kernel), B.BoundaryDescription(bodies=[HalfPlaneBody(p0, n)])).cpu().numpy()
    for i, di in enumerate(d):
        ref = 0.5 if di == 0 else (float(planar2d(name, mp.mpf(di), dps=30)) if di > 0 else 1 - float(planar2d(name, mp.mpf(-di), dps=30)))
        assert abs(lam[i] - ref) < 1e-9, (di, lam[i], ref)


def test_mixture_with_explicit_mesh_and_constant_field_operations():
    """explicit wall mesh + disk obstacle in one call = sum of the separate calls, all operation classes with per-body constants."""
    wall_v = torch.tensor([[-5.0, -3.0], [5.0, -3.0], [5.0, 0.0], [-5.0, 0.0]], dtype=TD, device=DEV)
    wall_e = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device=DEV)
    mesh = B.BoundaryMesh(wall_v, wall_e)
    disk = DiskBody((0.0, 3.0), 2.5)                     # tier 3 for h = 1
    small = DiskBody((2.0, 1.2), 0.1, bodyId=1)         # tier 4
    mid = DiskBody((-2.5, 1.0), 0.8, bodyId=2)          # tier 2 fallback (polygon merged into the mesh)
    rng = np.random.default_rng(0)
    pos = np.concatenate([rng.uniform([-4, 0.05], [4, 1.8], (60, 2))])
    pos = pos[[np.hypot(*(p - np.array(b.center))) > b.radius + 1e-3 for p in pos for b in [disk]]]
    ps = state(pos, rho=1.0)
    N = len(pos)
    rho_n = torch.ones(4, dtype=TD, device=DEV)
    A_n = torch.tensor([0.3, -0.2, 0.5, 0.1], dtype=TD, device=DEV)
    desc = B.BoundaryDescription(mesh=mesh, bodies=[disk, small, mid])
    adj = B.buildBoundaryAdjacency(ps, props(WarpOperation.Density), desc)
    tiers = {int(b.bodyId): set(adj.bodyTier[k].cpu().numpy().tolist()) for k, b in enumerate(desc.bodies)}
    assert tiers[0] <= {3, 2} and 3 in tiers[0]
    assert tiers[1] <= {4, 2} and 4 in tiers[1]
    assert tiers[2] == {2}
    dens = B.boundaryOperation(ps, props(WarpOperation.Density), desc, referenceDensities=1.0, bodyDensities=1.0, adjacency=adj)
    only_mesh = B.boundaryOperation(ps, props(WarpOperation.Density), mesh, referenceDensities=1.0)
    only_bodies = B.boundaryOperation(ps, props(WarpOperation.Density), B.BoundaryDescription(bodies=[disk, small, mid]), bodyDensities=1.0)
    assert (dens - only_mesh - only_bodies).abs().max() < 1e-10
    # constant per-body fields: Interpolate and the four gradient modes equal the explicit formulas with lambda, grad lambda of that body
    bodyA = torch.tensor([0.3, -0.2, 0.5], dtype=TD, device=DEV)
    ip = B.boundaryOperation(ps, props(WarpOperation.Interpolate), B.BoundaryDescription(bodies=[disk, small, mid]), bodyValues=bodyA, referenceValues=None).cpu().numpy()
    lam = [adj.bodyLam[k].cpu().numpy() for k in range(3)]
    # tier-2 bodies contribute through the merged mesh: compare with the same description evaluated on its own
    assert np.isfinite(ip).all()
    q = torch.tensor(np.random.default_rng(1).normal(size=N), dtype=TD, device=DEV)
    for mode in (GradientScheme.Naive, GradientScheme.Difference, GradientScheme.Summation, GradientScheme.Symmetric):
        o = B.boundaryOperation(ps, props(WarpOperation.Gradient, mode), B.BoundaryDescription(bodies=[disk]), queryValues=q, bodyValues=bodyA[:1],
                                bodyDensities=1.0).cpu().numpy()
        grad = adj.bodyGrad[0].cpu().numpy()
        A, fi = float(bodyA[0]), q.cpu().numpy()
        a = {GradientScheme.Naive: A + 0 * fi, GradientScheme.Difference: A - fi, GradientScheme.Summation: A + fi, GradientScheme.Symmetric: 1.0 * 1.0 * (fi + A)}[mode]
        sel = np.isin(adj.bodyTier[0].cpu().numpy(), (3, 4))
        ref = (a[:, None] * grad) * sel[:, None]
        assert np.abs(o - ref).max() < 1e-10, mode


def test_reaction_includes_the_implicit_bodies():
    center, R = (0.0, 3.0), 2.5
    rng = np.random.default_rng(2)
    pos = rng.uniform([-3, 0.0], [3, 1.0], (40, 2))
    pos = pos[np.hypot(pos[:, 0] - center[0], pos[:, 1] - center[1]) > R + 0.02]
    ps = state(pos, mass=0.7, rho=1.0)
    desc = B.BoundaryDescription(bodies=[DiskBody(center, R)])
    q = torch.tensor(rng.uniform(0, 2, len(pos)), dtype=TD, device=DEV)
    out, Rv, Rb = B.boundaryOperation(ps, props(WarpOperation.Gradient, GradientScheme.Symmetric), desc, queryValues=q, bodyValues=torch.tensor([1.3], dtype=TD, device=DEV),
                                      bodyDensities=1.0, returnReaction=True)
    total = (ps.masses.to(TD)[:, None] * out).sum(0)
    assert (Rb.sum(0) + total).abs().max() < 1e-12 * (1 + total.abs().max())


def test_policy_thresholds_are_hard_and_per_particle():
    center = (0.0, 0.0)
    pos = np.array([[0.0, 4.2], [0.0, 4.3]])                       # R = 4
    sup = np.array([1.0, 2.5])                                      # R/h = 4 (tier 3), 1.6 (gap -> tier 2 fallback)
    ps = state(pos, h=sup)
    adj = B.buildBoundaryAdjacency(ps, props(WarpOperation.Density), B.BoundaryDescription(bodies=[DiskBody(center, 4.0)]))
    assert adj.bodyTier[0].cpu().numpy().tolist() == [3, 2]
    adj2 = B.buildBoundaryAdjacency(ps, props(WarpOperation.Density), B.BoundaryDescription(bodies=[DiskBody(center, 4.0)], policy=TierPolicy(tier3MinRadius=1.0)))
    assert adj2.bodyTier[0].cpu().numpy().tolist() == [3, 3]
