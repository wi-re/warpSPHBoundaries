"""Channel-restricted edge plans of warpbc and the DevicePlan cache (WORK-008 T8.1).

`edge_channels(..., channels=(...))` evaluates only the terms of the asked channels: the channels are independent sums and
a launch has one thread per (query, edge) pair with no reduction, so the asked channels are bit-identical (torch.equal) to
the full plan's result and the others are exactly 0.  The monomial DevicePlan arrays (host -> device uploads, 0.37 ms per
build) are now cached per (kernel, device, channels) like the Chebyshev plans already were, and `channels=None` keeps the
unchanged all-nine behaviour.

Tolerances (stated before looking): exact (torch.equal / == 0 / == 1 construction) by the no-reduction structure above;
the negative controls assert the pruned result is NOT the full one (max|full g0| > 0.5; measured 0.95 / 1.59 / 1.49 / 1.49
on the four (kernel, stable) cases).
"""
import numpy as np
import pytest
import torch
import warp as wp

from edgebound import warpbc

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
TD = torch.float64


@pytest.fixture
def geometry(device):
    """the square [-.25, .25]^2 (counter-clockwise), 200 random query points (seed 1) in [-.5, .5]^2, support 0.12,
    every one of the 200 x 4 (query, edge) pairs."""
    verts = torch.tensor([(-.25, -.25), (.25, -.25), (.25, .25), (-.25, .25)], dtype=TD, device=device)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int32, device=device)
    pos = torch.tensor(np.random.default_rng(1).uniform(-.5, .5, (200, 2)), dtype=TD, device=device)
    sup = torch.full((200,), 0.12, dtype=TD, device=device)
    pair_q = torch.arange(200).repeat_interleave(4).to(torch.int32).to(device)
    pair_e = torch.arange(4).tile(200).to(torch.int32).to(device)
    return verts, edges, pos, sup, pair_q, pair_e


@pytest.mark.parametrize("device", DEVICES)
def test_pruned_edge_channels_equal_the_full_channels(device, geometry):
    """(a) for cone, lw2, w2 (monomial) and w2 stable=(16, 8) (Chebyshev route), on all 800 pairs: the (3, 4)-pruned
    channels 3:5 are torch.equal to the full result, the other seven are exactly 0; the (5, 6)-pruned channels 5:7 are
    torch.equal to the full result, its 3:5 are 0, and the full g0 is not small (the pruned result is not the full one)."""
    verts, edges, pos, sup, pair_q, pair_e = geometry
    dev = str(device)
    from edgebound.viscosity import lap_factor
    lap_factor(1.0, "w2")                                                    # registers the 'lw2' kernel (lazy registration)
    for kernel, stable in (("cone", None), ("lw2", None), ("w2", None), ("w2", (16, 8))):
        full = warpbc.edge_channels(pair_q, pair_e, pos, sup, verts, edges, kernel, device=dev, stable=stable)
        p34 = warpbc.edge_channels(pair_q, pair_e, pos, sup, verts, edges, kernel, device=dev, stable=stable, channels=(3, 4))
        assert torch.equal(p34[:, 3:5], full[:, 3:5]), (kernel, stable)
        assert (p34[:, [0, 1, 2, 5, 6, 7, 8]] == 0).all(), (kernel, stable)
        p56 = warpbc.edge_channels(pair_q, pair_e, pos, sup, verts, edges, kernel, device=dev, stable=stable, channels=(5, 6))
        assert (p56[:, 3:5] == 0).all(), (kernel, stable)
        assert torch.equal(p56[:, 5:7], full[:, 5:7]), (kernel, stable)
        g0max = float(full[:, 3:5].abs().max())
        assert g0max > 0.5, (kernel, stable, g0max)
        print(f"(a) {kernel} stable={stable}: max|full g0| = {g0max:.3f} (> 0.5), pruned (3,4) torch.equal, (5,6) g0 = 0")
    # plan size: the pruned (3, 4) plan keeps only the four g0 terms (E 4, V 0), the full w2 plan has all 21 (E 16, V 5)
    for k in ("cone", "lw2", "w2"):
        p = warpbc._device_plan(k, dev, (3, 4))
        assert p.nE + p.nV == 4, (k, p.nE, p.nV)
    pf = warpbc._device_plan("w2", dev)
    assert pf.nE + pf.nV == 21, (pf.nE, pf.nV)
    print(f"(a) plan sizes: pruned (3,4) nE+nV = 4 (cone, lw2, w2); full w2 nE+nV = 21 (E {pf.nE}, V {pf.nV})")


@pytest.mark.parametrize("device", DEVICES)
def test_device_plan_cache(device, geometry):
    """(b) the monomial plan is cached per (kernel, device, channels): same key (any order of the same channels) -> the
    same object, pruned and full are different objects, and two edge_channels calls of the same kernel construct exactly
    one plan (with a counting wrapper around DevicePlan.__init__, restored in a finally)."""
    verts, edges, pos, sup, pair_q, pair_e = geometry
    dev = str(device)
    warpbc._DEVICE_PLANS.clear()
    assert warpbc._device_plan("w2", dev) is warpbc._device_plan("w2", dev)
    assert warpbc._device_plan("w2", dev, (3, 4)) is warpbc._device_plan("w2", dev, [4, 3])
    assert warpbc._device_plan("w2", dev, (3, 4)) is not warpbc._device_plan("w2", dev)
    warpbc._DEVICE_PLANS.clear()
    n = 0
    orig = warpbc.DevicePlan.__init__

    def counting(self, *a, **k):
        nonlocal n
        n += 1
        orig(self, *a, **k)

    warpbc.DevicePlan.__init__ = counting
    try:
        c1 = warpbc.edge_channels(pair_q, pair_e, pos, sup, verts, edges, "w2", device=dev)
        c2 = warpbc.edge_channels(pair_q, pair_e, pos, sup, verts, edges, "w2", device=dev)
    finally:
        warpbc.DevicePlan.__init__ = orig
    assert n == 1, n
    assert torch.equal(c1, c2)
    print(f"(b) cache: same key is the same object, pruned != full, two edge_channels calls construct {n} plan")


@pytest.mark.parametrize("device", DEVICES)
def test_pruned_adjacency_guard_and_equality(device):
    """(c) the (3,4)-pruned adjacency carries channels == {3,4} (the full one has channels None), and the Naive Gradient
    of a constant scalar field through it is torch.equal to the one through the full adjacency (same pairs, the pruned one
    holds the same g0 and 0 elsewhere, the constant gradient reads g0 only).  The guard raises ValueError for every other
    use of a pruned adjacency: a Density, a Covariance, a Gradient of a perQuery field with a1, a Gradient with
    returnReaction, and a Gradient through an adjacency pruned to a set that does not contain {3,4} (channels=(0,))."""
    from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
    from edgebound.deltasph2d import hydrostatic_tank
    from edgebound.scene import BodyField, sceneOperation
    dev = str(device)
    sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device)
    ps = ParticleState(positions=sim.x, supports=sim.Hvec, masses=torch.full_like(sim.rho, sim.m),
                       kinds=sim.kinds, densities=sim.rho)
    pr = OperationProperties(kernel="cone", operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive,
                             operationMode=OperationDirection.BoundaryToFluid)
    one = BodyField(torch.tensor(1.0, dtype=TD, device=dev))
    adj = sim.scene.buildAdjacency(ps, pr, channels=(3, 4))
    full = sim.scene.buildAdjacency(ps, pr)
    assert adj.channels == frozenset({3, 4}), adj.channels
    assert full.channels is None, full.channels
    got = sceneOperation(ps, pr, sim.scene, adj, None, [one], perBody=True)
    ref = sceneOperation(ps, pr, sim.scene, full, None, [one], perBody=True)
    assert torch.equal(got, ref)
    print(f"(c) pruned adjacency channels = {sorted(adj.channels)} (full is None); Naive Gradient of a constant torch.equal")
    density = OperationProperties(kernel="cone", operation=WarpOperation.Density, gradientMode=GradientScheme.Naive,
                                  operationMode=OperationDirection.BoundaryToFluid)
    covariance = OperationProperties(kernel="cone", operation=WarpOperation.Covariance, gradientMode=GradientScheme.Naive,
                                     operationMode=OperationDirection.BoundaryToFluid)
    with pytest.raises(ValueError, match="built with the channels"):
        sceneOperation(ps, density, sim.scene, adj, None, [BodyField(rho=1.0)], perBody=True)
    with pytest.raises(ValueError, match="built with the channels"):
        sceneOperation(ps, covariance, sim.scene, adj, None, [one], perBody=True)
    nq = len(ps.positions)
    pq = BodyField(torch.zeros(nq, dtype=TD, device=dev), torch.zeros(nq, 2, dtype=TD, device=dev), rho=1.0, perQuery=True)
    with pytest.raises(ValueError, match="built with the channels"):
        sceneOperation(ps, pr, sim.scene, adj, None, [pq], perBody=True)
    with pytest.raises(ValueError, match="built with the channels"):
        sceneOperation(ps, pr, sim.scene, adj, None, [one], returnReaction=True, perBody=True)
    adj0 = sim.scene.buildAdjacency(ps, pr, channels=(0,))
    assert adj0.channels == frozenset({0})
    with pytest.raises(ValueError, match="built with the channels"):
        sceneOperation(ps, pr, sim.scene, adj0, None, [one], perBody=True)
    print("(c) guard: Density, Covariance, perQuery+a1 Gradient, returnReaction Gradient, channels=(0,) all raise ValueError")


@pytest.mark.parametrize("device", DEVICES)
def test_cover_and_tensile_pruned_equal_full(device, monkeypatch):
    """(d) the cover vector and the tensile term built through the (3,4)-pruned adjacency (the default path) equal the
    full-adjacency results to 1e-13 * scale: the L body of test_cover_scene, 200 queries (34 inside the body, so the
    indicator pseudo-pairs of the full adjacency matter).  The tolerance allows for the index_add_ atomics of
    sceneOperation (not guaranteed bit-reproducible); the reviewer measured 0 on all three (scales 0.308 / 235 / 751)."""
    from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
    from edgebound import cover, tensile
    from edgebound.scene import Body, Scene, SurfaceRep
    dev = str(device)
    LS = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    CENTER, ANGLE, H = (0.3, -0.2), 0.7, 0.6
    sc = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)], device)
    pos = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    n = len(pos)
    ps = ParticleState(positions=torch.tensor(pos, dtype=TD, device=dev),
                       supports=torch.full((n,), H, dtype=TD, device=dev),
                       masses=torch.ones(n, dtype=TD, device=dev), kinds=torch.zeros(n, dtype=torch.int32, device=dev),
                       densities=torch.ones(n, dtype=TD, device=dev))
    pr = OperationProperties(kernel="cone", operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive,
                             operationMode=OperationDirection.BoundaryToFluid)
    full_adj = sc.buildAdjacency(ps, pr)
    assert full_adj.channels is None
    got_c = cover.cover_vector_scene(sc, pos, H)                       # default: the (3,4)-pruned adjacency
    ref_c = cover.cover_vector_scene(sc, pos, H, adjacency=full_adj)   # explicit full adjacency
    dc = float(np.abs(got_c.cpu().numpy() - ref_c.cpu().numpy()).max())
    sc_c = float(np.abs(ref_c.cpu().numpy()).max())
    assert dc <= 1e-13 * sc_c, (dc, sc_c)
    print(f"(d) cover: max|Δ| = {dc:.2e} (<= 1e-13 * scale {sc_c:.3f})")
    got_t = {fam: tensile.tensile_vector_scene(sc, pos, H, fam) for fam in ("w2", "w4")}     # default: pruned
    orig = Scene.buildAdjacency

    def _full_only(self, queryParticles, operationProperties, channels=None):
        return orig(self, queryParticles, operationProperties)         # drop `channels` -> the full adjacency

    monkeypatch.setattr(Scene, "buildAdjacency", _full_only)
    for fam in ("w2", "w4"):
        ref_t = tensile.tensile_vector_scene(sc, pos, H, fam)          # patched: full
        dt_ = float(np.abs(got_t[fam].cpu().numpy() - ref_t.cpu().numpy()).max())
        sc_t = float(np.abs(ref_t.cpu().numpy()).max())
        assert dt_ <= 1e-13 * sc_t, (fam, dt_, sc_t)
        print(f"(d) tensile {fam}: max|Δ| = {dt_:.2e} (<= 1e-13 * scale {sc_t:.3f})")
