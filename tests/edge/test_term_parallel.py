"""Term-parallel evaluation of `warpbc.edge_channels` (docs/plan-wall-evaluation.md step 1b): one thread per (pair, term) + a fixed-order reduction instead of one thread per pair.

The reduction adds the terms in the order of the sequential kernel (edge terms, then vertex terms with `(ch + a) - b`), so the two paths differ only by the compiler's contraction of a*b + c
into an FMA: measured <= 3.2e-16 of the largest value in float64 (monomial and Chebyshev plans); channels made of one product per term ((3, 4) of the gradient kernels, (0,)) are exactly equal, the
unasked channels of a pruned plan exactly 0.
Tolerance stated before looking: 1e-14 * max|channel| (float64 contract of the exactness tests); determinism: two launches are torch.equal (no atomics).
"""
import numpy as np
import pytest
import torch
import warp as wp

from warpSPHBoundaries.edge import warpbc

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
TD = torch.float64


@pytest.fixture
def geometry(device):
    """an L-shaped loop (reflex corner; counter-clockwise), a degenerate zero-length edge, 300 random queries in [-1, 3]^2 (inside, near edges / vertices, far), support 0.35 - 0.7, every
    (query, edge) pair."""
    P = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    verts = torch.tensor(np.vstack([P, P[:1]]), dtype=TD, device=device)                    # the extra vertex duplicates vertex 0 -> edge (6, 0) has zero length
    edges = torch.tensor([[i, (i + 1) % 6] for i in range(6)] + [[6, 0]], dtype=torch.int32, device=device)
    rng = np.random.default_rng(11)
    n = 300
    pos = torch.tensor(rng.uniform(-1, 3, (n, 2)), dtype=TD, device=device)
    pos[:6] = torch.tensor(P, dtype=TD, device=device) + 1e-3                                # queries almost on the vertices
    sup = torch.tensor(rng.uniform(0.35, 0.7, n), dtype=TD, device=device)
    ne = len(edges)
    pair_q = torch.arange(n).repeat_interleave(ne).to(torch.int32).to(device)
    pair_e = torch.arange(ne).tile(n).to(torch.int32).to(device)
    return verts, edges, pos, sup, pair_q, pair_e


def both(geometry, kernel, device, **kw):
    verts, edges, pos, sup, pair_q, pair_e = geometry
    run = lambda mode: (setattr(warpbc, "TERM_PARALLEL", mode), warpbc.edge_channels(pair_q, pair_e, pos, sup, verts, edges, kernel, device=str(device), **kw).clone())[1]
    try:
        return run(False), run(True), run(True)
    finally:
        warpbc.TERM_PARALLEL = "auto"


@pytest.mark.parametrize("device", DEVICES)
def test_term_parallel_equals_sequential(device, geometry):
    """(a) monomial plan (stable=False) and Chebyshev plan (stable=(8, 6)), kernels w2 (all channels), lw2 (all), w2p5 and cone (gradient channels (3, 4)) and a (0,)- and (5, 6)-pruned plan:
    equal to 1e-14 of the scale, equal run to run, pruned channels exactly 0."""
    from warpSPHBoundaries.scene.viscosity import lap_factor
    from warpSPHBoundaries.scene import tensile
    lap_factor(1.0, "w2"); tensile._register("w2")
    for stable in (False, (8, 6)):
        for kernel, ch in (("w2", None), ("lw2", None), ("w2p5", (3, 4)), ("cone", (3, 4)), ("w2", (0,)), ("w2", (5, 6))):
            seq, par, par2 = both(geometry, kernel, device, stable=stable, channels=ch)
            sc = float(seq.abs().max())
            assert sc > 0.1, (kernel, ch, sc)
            d = float((seq - par).abs().max())
            assert d <= 1e-14 * sc, (stable, kernel, ch, d, sc)
            assert torch.equal(par, par2), (stable, kernel, ch)
            if ch is not None:
                rest = [k for k in range(9) if k not in ch]
                assert (par[:, rest] == 0).all(), (stable, kernel, ch)
            print(f"(a) stable={stable} {kernel} {ch}: max|seq - par| {d:.1e} (scale {sc:.2f})")


@pytest.mark.parametrize("device", DEVICES)
def test_auto_switch_and_single_term_plan(device, geometry):
    """(b) `TERM_PARALLEL = 'auto'` takes the term-parallel path up to TERM_PARALLEL_MAX_PAIRS pairs and the sequential one above (observed through the launch counts of the two
    kernels), and a plan with a single term (nT = 1) always takes the sequential kernel (no division by the term count)."""
    verts, edges, pos, sup, pair_q, pair_e = geometry
    calls = {"par": 0, "seq": 0}
    orig = wp.launch

    def counting(kernel, *a, **k):
        name = getattr(kernel, "key", "")
        if name.endswith("_edge_terms_kernel") or name.endswith("_edge_terms_cheb_kernel"): calls["par"] += 1
        if name.endswith("_edge_channels_kernel") or name.endswith("_edge_channels_cheb_kernel"): calls["seq"] += 1
        return orig(kernel, *a, **k)
    wp.launch = counting
    try:
        warpbc.edge_channels(pair_q, pair_e, pos, sup, verts, edges, "w2", device=str(device), stable=False)
        assert calls == {"par": 1, "seq": 0}, calls
        old = warpbc.TERM_PARALLEL_MAX_PAIRS
        warpbc.TERM_PARALLEL_MAX_PAIRS = len(pair_q) - 1
        try:
            calls.update(par=0, seq=0)
            warpbc.edge_channels(pair_q, pair_e, pos, sup, verts, edges, "w2", device=str(device), stable=False)
            assert calls == {"par": 0, "seq": 1}, calls
        finally:
            warpbc.TERM_PARALLEL_MAX_PAIRS = old
        assert not warpbc._use_term_parallel(1000, 1)
    finally:
        wp.launch = orig
