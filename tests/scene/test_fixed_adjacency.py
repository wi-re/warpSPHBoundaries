"""Fixed-capacity wall adjacency (`scene/fixedadj.py`, plan step 5) against the pair-list adjacency (`Scene.adjacency`) it replaces.

(a) the slots of the valid rows hold exactly the (row, edge) pairs of `SurfaceRep.topology`, in the same order; the indicator equals `indicatorFast` on the valid rows and is 0 on the others;
(b) every output of the fused evaluation and `cone_area` is torch.equal to the one on the old adjacency (the per-row sums are the same sums: dead slots add exact zeros) for the surface, the box
    obstacle and the box tank (scenes of test_fused_wall.py);
(c) the build does not synchronise with the host (`torch.cuda.set_sync_debug_mode('error')`), i.e. it can be captured in a CUDA graph;
(d) the solver with cfg.fixedAdjacency True / False: the same state after 60 steps (tank and dam break) to 1e-12 (the run-to-run spread of the fluid index_add sums is ~1e-15).
"""
import math
import os
import sys

import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import WarpOperation

sys.path.insert(0, os.path.dirname(__file__))
from test_fused_wall import make, make_box, props, FusedGroup, FusedWall, WallOutput   # noqa: E402

from warpSPHBoundaries.scene.fixedadj import fixed_adjacency   # noqa: E402

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64
GROUPS = (FusedGroup("w2"), FusedGroup("lw2"), FusedGroup("cone", (3, 4)), FusedGroup("w2p5", (3, 4)))
OUTS = (WallOutput("lam", 0, "lam"), WallOutput("G", 0, "g0"), WallOutput("Cov", 0, "cov"), WallOutput("A", 0, "a1g1"), WallOutput("lap", 1, "lap"), WallOutput("cover", 2, "g0"),
        WallOutput("tens", 3, "g0"))
SCENES = {"surface": lambda d: make(d, constant_support=0.5), "box-obstacle": lambda d: make_box(d, "inside"), "box-tank": lambda d: make_box(d, "outside")}


@pytest.mark.parametrize("device", DEVICES)
def test_slots_are_the_old_pairs_and_the_indicator_is_the_old_one(device):
    sc, ps, pos, _ = make(device, constant_support=0.5)
    old = sc.adjacency(ps, props(WarpOperation.Density, "w2"))
    new = fixed_adjacency(sc, ps, props(WarpOperation.Density, "w2"), 0.5)
    for bo, bn in zip(old.bodies, new.bodies):
        assert torch.equal(bn.valid[bo.cand], torch.ones(len(bo.cand), dtype=torch.bool, device=device)) and int(bn.valid.sum()) == len(bo.cand)
        to, tn = bo.reps[0], bn.reps[0]
        K = tn.K
        slots = tn.e.reshape(-1, K)
        for k, r in enumerate(bo.cand.tolist()):
            got = [int(v) for v in slots[r].tolist() if v >= 0]
            want = [int(v) for v in to.e[to.qi == k].tolist()]
            assert got == want, (r, got, want)
        rep = bo.body.reps[0]
        assert torch.equal(tn.ind[bo.cand], to.indicator(rep, bo.lpos))
        assert float(tn.ind[~bn.valid].abs().max()) == 0.0 if int((~bn.valid).sum()) else True


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scene", list(SCENES))
def test_fused_outputs_are_bit_equal_to_the_pair_list_adjacency(device, scene):
    sc, ps, pos, a1 = SCENES[scene](device)
    old = FusedWall(sc, sc.adjacency(ps, props(WarpOperation.Density, "w2")), GROUPS)
    new = FusedWall(sc, fixed_adjacency(sc, ps, props(WarpOperation.Density, "w2"), 0.5), GROUPS)
    a, b = old.evaluate(OUTS, a1=a1), new.evaluate(OUTS, a1=a1)
    for o in OUTS:
        assert torch.equal(a[o.name], b[o.name]), (scene, o.name, float((a[o.name] - b[o.name]).abs().max()))
    assert float(a["lam"].abs().max()) > 0.5
    axes = torch.as_tensor(np.random.default_rng(9).uniform(-1, 1, (len(pos), 2)), dtype=F64, device=device)
    for al in (math.pi / 6, math.pi):
        assert torch.equal(old.cone_area(axes, al), new.cone_area(axes, al)), (scene, al)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="host-sync detection needs CUDA")
@pytest.mark.parametrize("scene", list(SCENES))
def test_build_does_not_synchronise(scene):
    sc, ps, pos, _ = SCENES[scene]("cuda:0")
    p = props(WarpOperation.Density, "w2")
    fixed_adjacency(sc, ps, p, 0.5)                                         # warm-up: static structures, module loads
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        adj = fixed_adjacency(sc, ps, p, 0.5)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert adj.numQueries == len(pos)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["tank", "dambreak"])
def test_solver_state_does_not_depend_on_the_adjacency_kind(device, case):
    from warpSPHBoundaries.sim import cases
    res = {}
    for fixed in (False, True):
        if case == "tank":
            sim = cases.hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=cases.DeltaSPHConfig(noPen="impulse", shifting=True, fixedAdjacency=fixed))[0]
        else:
            sim = cases.marrone_dambreak(nx=30, shifting=True, noPen="impulse", device=device, fixedAdjacency=fixed)[0]
        for _ in range(60):
            sim.step()
        res[fixed] = (sim.x.clone(), sim.v.clone(), sim.rho.clone())
    for a, b in zip(res[False], res[True]):
        assert float((a - b).abs().max()) <= 1e-12, float((a - b).abs().max())      # the fluid pair sums (index_add) are not deterministic: same-code run-to-run spread ~1e-15
