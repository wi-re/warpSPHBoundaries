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
