"""Fused multi-kernel edge channels (`warpfused.fused_channels`, docs/plan-wall-evaluation.md step 3a): the groups of one launch equal the separate `warpbc.edge_channels` calls.

Groups (float64): `w2` all channels (monomial), `lw2` all (monomial), `cone` (3, 4) (monomial), `w2p5` (3, 4) (the registered Chebyshev route (16, 8)), `w2` (5, 6) on the Chebyshev (8, 6) route,
mixed in ONE launch (per-term route tag, Gauss table offset, panel count).  Tolerance stated before looking: 1e-14 of the largest channel value of the group (the two paths differ only by
the compiler's FMA contraction; measured in the term-parallel test: 3e-16); two launches are torch.equal (no atomics); an empty pair list gives an empty result.
"""
import numpy as np
import pytest
import torch
import warp as wp

from edgebound.edge import warpbc, warpfused
from edgebound.edge.warpfused import FusedGroup

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
TD = torch.float64


@pytest.fixture
def geometry(device):
    P = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    verts = torch.tensor(np.vstack([P, P[:1]]), dtype=TD, device=device)
    edges = torch.tensor([[i, (i + 1) % 6] for i in range(6)] + [[6, 0]], dtype=torch.int32, device=device)      # the last edge has zero length
    rng = np.random.default_rng(5)
    n = 300
    pos = torch.tensor(rng.uniform(-1, 3, (n, 2)), dtype=TD, device=device)
    pos[:6] = torch.tensor(P, dtype=TD, device=device) + 1e-3
    sup = torch.tensor(rng.uniform(0.35, 0.7, n), dtype=TD, device=device)
    ne = len(edges)
    return verts, edges, pos, sup, torch.arange(n).repeat_interleave(ne).to(torch.int32).to(device), torch.arange(ne).tile(n).to(torch.int32).to(device)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_groups_equal_separate_launches(device, geometry):
    from edgebound.scene.viscosity import lap_factor
    from edgebound.scene import tensile
    lap_factor(1.0, "w2"); tensile._register("w2")
    verts, edges, pos, sup, pq, pe = geometry
    groups = (FusedGroup("w2"), FusedGroup("lw2"), FusedGroup("cone", (3, 4)), FusedGroup("w2p5", (3, 4)), FusedGroup("w2", (5, 6), "cheb", 8, 6))
    c = warpfused.fused_channels(pq, pe, pos, sup, verts, edges, groups, device=str(device), as_float64=True)
    assert c.shape == (len(pq), 9 * len(groups))
    for g, grp in enumerate(groups):
        r = grp.resolved()
        ref = warpbc.edge_channels(pq, pe, pos, sup, verts, edges, r.kernel, device=str(device), channels=r.channels,
                                   stable=(r.nodes, r.panels) if r.route == "cheb" else False)
        got = c[:, 9 * g:9 * g + 9]
        sc = float(ref.abs().max())
        d = float((got - ref).abs().max())
        assert sc > 0.1 and d <= 1e-14 * sc, (r, d, sc)
        if r.channels is not None:
            assert (got[:, [k for k in range(9) if k not in r.channels]] == 0).all(), r
        print(f"{r.kernel} {r.channels} {r.route}: max|fused - separate| {d:.1e} (scale {sc:.2f})")
    c2 = warpfused.fused_channels(pq, pe, pos, sup, verts, edges, groups, device=str(device), as_float64=True)
    assert torch.equal(c, c2)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_empty_and_single_group(device, geometry):
    verts, edges, pos, sup, pq, pe = geometry
    e = warpfused.fused_channels(pq[:0], pe[:0], pos, sup, verts, edges, (FusedGroup("w2"),), device=str(device))
    assert e.shape == (0, 9)
    one = warpfused.fused_channels(pq, pe, pos, sup, verts, edges, (FusedGroup("w2"),), device=str(device), as_float64=True)
    ref = warpbc.edge_channels(pq, pe, pos, sup, verts, edges, "w2", device=str(device), stable=False)
    assert float((one - ref).abs().max()) <= 1e-14 * float(ref.abs().max())
