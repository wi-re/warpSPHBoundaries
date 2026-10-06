"""Adjacency (pair topology) / precompute (`PairMoments`) / evaluation (`sceneOperation`) split of the scene layer (docs/plan-wall-evaluation.md §1, step 2b).

`Scene.adjacency` holds integers and kernel-independent geometry only; `Scene.precompute` makes the per-pair integrals of one kernel (and channel set) over
it; `SceneAdjacency.restrict(index)` is the adjacency of a query subset at the same positions and supports.  Scene: the L-shaped body of test_cover_scene.py
(rotated, translated; reflex corner), 300 random queries in a box around it (inside the body, near the edges, far away) with per-query supports 0.45-0.6.

Tolerances (stated before looking): the GPU index_add accumulation is not bit-reproducible, so results of different routes agree to 1e-12 of their scale;
pair counts and candidate lists are integers and agree exactly.
"""
import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation

from warpSPHBoundaries.scene.scene import Body, BodyField, Scene, SurfaceRep, SceneAdjacency, PairMoments, sceneOperation
from warpSPHBoundaries.scene.viscosity import lap_factor

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64
LS = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
CENTER, ANGLE = (0.3, -0.2), 0.7


def props(op, kernel):
    return OperationProperties(kernel=kernel, operation=op, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)


def make(device, n=300, seed=5):
    lap_factor(1.0, "w2")                                                    # registers the 'lw2' kernel (lazy registration)
    sc = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CENTER, angle=ANGLE)], device)
    rng = np.random.default_rng(seed)
    pos = torch.as_tensor(rng.uniform(-1.2, 3.2, (n, 2)), dtype=F64, device=device)
    sup = torch.as_tensor(rng.uniform(0.45, 0.6, n), dtype=F64, device=device)
    return sc, pos, sup


def state(pos, sup, device):
    n = len(pos)
    return ParticleState(positions=pos, supports=sup, masses=torch.ones(n, dtype=F64, device=device), kinds=torch.zeros(n, dtype=torch.int32, device=device),
                         densities=torch.ones(n, dtype=F64, device=device))


def evaluate(sc, ps, kernel, ops, moments):
    """sceneOperation of every op in `ops` over `moments`, per body, flattened to numpy (a unit wall; the Gradient of a constant)."""
    out = []
    for op in ops:
        fld = BodyField(torch.tensor(1.0, dtype=F64, device=sc.device)) if op == WarpOperation.Gradient else BodyField(rho=1.0)
        out.append(sceneOperation(ps, props(op, kernel), sc, moments, None, [fld], perBody=True).cpu().numpy().ravel())
    return np.concatenate(out)


def close(a, b, tol=1e-12):
    s = max(float(np.abs(b).max()), 1e-300)
    d = float(np.abs(a - b).max())
    assert d <= tol * s, (d, s)
    return d


OPS = (WarpOperation.Density, WarpOperation.Gradient, WarpOperation.Covariance)


@pytest.mark.parametrize("device", DEVICES)
def test_adjacency_holds_no_kernel_and_serves_any_kernel(device):
    """(a) an adjacency has no kernel and no channels; the one built with the props of kernel 'cone' precomputes 'w2' (all channels) and 'lw2' exactly
    like the one-call route `pairMoments`, and the pair count of the topology is the same whatever the props said."""
    sc, pos, sup = make(device)
    ps = state(pos, sup, device)
    adj = sc.adjacency(ps, props(WarpOperation.Density, "cone"))
    assert isinstance(adj, SceneAdjacency) and not hasattr(adj, "kernel") and not hasattr(adj, "channels")
    adj2 = sc.adjacency(ps, props(WarpOperation.Density, "w2"))
    assert [len(t.qi) for t in adj.bodies[0].reps] == [len(t.qi) for t in adj2.bodies[0].reps] and torch.equal(adj.bodies[0].reps[0].e, adj2.bodies[0].reps[0].e)
    for kernel in ("w2", "lw2", "cone"):
        pm = sc.precompute(adj, props(WarpOperation.Density, kernel))
        assert isinstance(pm, PairMoments) and pm.kernel == kernel and pm.channels is None
        ref = sc.pairMoments(ps, props(WarpOperation.Density, kernel))
        d = close(evaluate(sc, ps, kernel, OPS, pm), evaluate(sc, ps, kernel, OPS, ref))
        print(f"(a) {kernel}: precompute over a shared adjacency vs pairMoments max|diff| {d:.1e}")
    pm34 = sc.precompute(adj, props(WarpOperation.Gradient, "cone"), channels=(3, 4))
    assert pm34.channels == frozenset({3, 4}) and pm34.stats["pairs"] <= sc.precompute(adj, props(WarpOperation.Density, "cone")).stats["pairs"]


@pytest.mark.parametrize("device", DEVICES)
def test_sceneoperation_takes_adjacency_moments_or_nothing(device):
    """(b) sceneOperation(.., None), (.., SceneAdjacency) and (.., PairMoments) are the same evaluation."""
    sc, pos, sup = make(device)
    ps = state(pos, sup, device)
    pr = props(WarpOperation.Gradient, "w2")
    adj = sc.adjacency(ps, pr)
    pm = sc.precompute(adj, pr)
    ref = evaluate(sc, ps, "w2", OPS, pm)
    for src in (None, adj):
        got = np.concatenate([sceneOperation(ps, props(op, "w2"), sc, src, None,
                                             [BodyField(torch.tensor(1.0, dtype=F64, device=device)) if op == WarpOperation.Gradient else BodyField(rho=1.0)],
                                             perBody=True).cpu().numpy().ravel() for op in OPS])
        close(got, ref)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("indicator_first", [False, True])
def test_restrict_equals_a_fresh_adjacency_of_the_subset(device, indicator_first):
    """(c) restrict(index) of the full adjacency, precomputed for w2 (all channels), lw2 (all) and cone / w2p5 (gradient channels), evaluates like the adjacency
    built from scratch for the subset positions.  The subset is an ascending subset of every kind of query (inside the body, within a support of an edge, far
    away); `indicator_first`: the full adjacency's indicator is computed (by a w2 precompute) before the restriction (the restriction slices it) or not
    (the restricted adjacency computes it itself)."""
    sc, pos, sup = make(device)
    ps = state(pos, sup, device)
    adj = sc.adjacency(ps, props(WarpOperation.Density, "w2"))
    if indicator_first:
        sc.precompute(adj, props(WarpOperation.Density, "w2"))
    ind = sc.bodies[0].reps[0].indicator(sc.bodies[0].pose.toLocal(pos))
    index = torch.nonzero((torch.arange(len(pos), device=device) % 3 != 1)).flatten()
    assert int((ind[index] != 0).sum()) > 10 and int((ind[index] == 0).sum()) > 10            # both inside and outside queries are in the subset
    sub = adj.restrict(index)
    assert sub.numQueries == len(index) and isinstance(sub, SceneAdjacency)
    ps_sub = state(pos[index], sup[index], device)
    fresh = sc.adjacency(ps_sub, props(WarpOperation.Density, "w2"))
    assert sub.bodies[0].cand.tolist() == fresh.bodies[0].cand.tolist()
    assert sub.stats["candidates"] == fresh.stats["candidates"]
    ta, tb = sub.bodies[0].reps[0], fresh.bodies[0].reps[0]
    # same (row, edge) pair set (the order is the cell-list order of the respective search)
    key = lambda t: sorted(zip(t.qi.tolist(), t.e.tolist()))
    assert key(ta) == key(tb)
    for kernel in ("w2", "lw2"):
        d = close(evaluate(sc, ps_sub, kernel, OPS, sc.precompute(sub, props(WarpOperation.Density, kernel))),
                  evaluate(sc, ps_sub, kernel, OPS, sc.precompute(fresh, props(WarpOperation.Density, kernel))))
        print(f"(c) indicator_first={indicator_first} {kernel}: restricted vs fresh max|diff| {d:.1e}")
    for kernel in ("cone", "w2p5"):
        if kernel == "w2p5":
            from warpSPHBoundaries.scene import tensile
            tensile._register("w2")
        pr = props(WarpOperation.Gradient, kernel)
        a = evaluate(sc, ps_sub, kernel, (WarpOperation.Gradient,), sc.precompute(sub, pr, channels=(3, 4)))
        b = evaluate(sc, ps_sub, kernel, (WarpOperation.Gradient,), sc.precompute(fresh, pr, channels=(3, 4)))
        close(a, b)
    # restricting twice and restricting to everything are the identity on the evaluation
    full = evaluate(sc, ps, "w2", OPS, sc.precompute(adj, props(WarpOperation.Density, "w2")))
    again = evaluate(sc, ps, "w2", OPS, sc.precompute(adj.restrict(torch.arange(len(pos), device=device)), props(WarpOperation.Density, "w2")))
    close(again, full)


@pytest.mark.parametrize("device", DEVICES)
def test_restrict_to_nothing_and_to_far_queries(device):
    """(d) an empty index gives an adjacency of zero queries (no pairs); queries beyond every support restrict to a body with no candidates and evaluate to 0."""
    sc, pos, sup = make(device)
    ps = state(pos, sup, device)
    adj = sc.adjacency(ps, props(WarpOperation.Density, "w2"))
    empty = adj.restrict(torch.zeros(0, dtype=torch.long, device=device))
    assert empty.numQueries == 0 and len(empty.bodies[0].cand) == 0
    far = torch.nonzero(sc.bodies[0].pose.toLocal(pos).norm(dim=1) > 20).flatten()
    assert len(far) == 0                                                    # none: the random box is near the body; use an explicit far point instead
    pos2 = torch.cat([pos, torch.tensor([[40.0, 40.0], [-30.0, 5.0]], dtype=F64, device=device)])
    sup2 = torch.cat([sup, torch.full((2,), 0.5, dtype=F64, device=device)])
    ps2 = state(pos2, sup2, device)
    adj2 = sc.adjacency(ps2, props(WarpOperation.Density, "w2"))
    sub = adj2.restrict(torch.tensor([len(pos), len(pos) + 1], device=device))
    assert len(sub.bodies[0].cand) == 0
    out = sceneOperation(state(pos2[-2:], sup2[-2:], device), props(WarpOperation.Density, "w2"), sc, sub, None, [BodyField(rho=1.0)], perBody=True)
    assert out.shape[-1] == 2 and float(out.abs().max()) == 0.0
