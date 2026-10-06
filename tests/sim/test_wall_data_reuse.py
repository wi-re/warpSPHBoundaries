"""Reuse of the wall adjacency of no_penetration in the next step's first RHS (WORK-008 T8.3).

`_wall_state` keeps `(x, body poses, adjacency, pair moments, lam, G, Hvec, kinds)` of its last build and reuses `adj, pm, lam, G` when the
positions, the poses, the supports and the kinds are unchanged; `lam` and `G` do not depend on the densities or the gravity, so
a cache hit is valid for a different `rho` and `g` (`A`, which depends on `g`, is always recomputed from the cached or fresh
adjacency).  Nothing in the solver modifies `lam` or `G` in place (the read-only contract).

Tolerances (stated before looking):
  (e) host-call counts, exact (deterministic): 3 Scene.adjacency + 9 Scene.precompute per step with the cache, 4 + 10 without (the near-wall consumers take
      `restrict` of the wall adjacency; before the adjacency / precompute split the same schedule was 9 / 10 buildAdjacency calls).
  (f) max|Δ| <= 1e-12 for x, v, rho: the same-code GPU run-to-run spread was measured at 8.7e-16 in v and 0 in x, rho, so 1e-12
      is three orders above it and six below the 1e-6 effect a stale cache would have.
  (g) a cache hit is torch.equal to a fresh evaluation; a rebuild is max|Δ| <= 1e-13 * max|fresh| (the index_add_ atomics of the
      wall operations are not guaranteed bit-reproducible); the invalidation cases assert the stale-cache error is visible
      (|Δlam| > 1e-4, reviewer measured 8.3e-3 / 8.9e-3) and the g-change moves A by > 1e-2 (reviewer measured 0.59).
"""
import pytest
import torch
import warp as wp

from warpSPHBoundaries.sim.deltasph2d import DeltaSPHConfig
from warpSPHBoundaries.sim.cases import hydrostatic_tank
from warpSPHBoundaries.scene.scene import Scene

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
TD = torch.float64

CFG = lambda device, fused=True: DeltaSPHConfig(noPen="impulse", shifting=True, fusedWall=fused, graphStep=False)       # host-call counts: no graph replay


def _disable_cache(sim):
    """wrap sim._wall_state (what rhs and _wall_data call) so the cache is cleared before every call (forces a fresh build each time)."""
    orig = sim._wall_state

    def no_cache(x, rho):
        sim._wallCache = None
        return orig(x, rho)

    sim._wall_state = no_cache
    return orig


def _counting_adjacency():
    """a (counter, restore) pair: counter[0] counts Scene.adjacency and fixed_adjacency calls, counter[1] Scene.precompute calls, counter[2] the stage-1 launches of the fused wall evaluation
    (`FusedWall` constructions) (class-level, restored by calling restore())."""
    from warpSPHBoundaries.scene import fused as F
    counter = [0, 0, 0]
    orig_a, orig_p, orig_f = Scene.adjacency, Scene.precompute, F.FusedWall.__init__

    def counting_a(self, *a, **k):
        counter[0] += 1
        return orig_a(self, *a, **k)

    def counting_p(self, *a, **k):
        counter[1] += 1
        return orig_p(self, *a, **k)

    def counting_f(self, *a, **k):
        counter[2] += 1
        return orig_f(self, *a, **k)

    from warpSPHBoundaries.scene import provider as D
    orig_x = D.fixed_adjacency                                          # the boundary provider builds the fused path adjacency with the fixed-capacity builder (cfg.fixedAdjacency)

    def counting_x(*a, **k):
        counter[0] += 1
        return orig_x(*a, **k)

    Scene.adjacency, Scene.precompute, F.FusedWall.__init__, D.fixed_adjacency = counting_a, counting_p, counting_f, counting_x

    def restore():
        Scene.adjacency, Scene.precompute, F.FusedWall.__init__, D.fixed_adjacency = orig_a, orig_p, orig_f, orig_x

    return counter, restore


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_reuse_saves_one_adjacency_per_step(device, fused):
    """(e) over 4 steps after a warm-up step, host-call counts (deterministic): per step with the cache / with it disabled
      fused wall evaluation (default):  3 adjacency, 3 stage-1 launch families (FusedWall), 0 precompute  /  4, 4, 0
      sceneOperation path (fusedWall=False):  3 adjacency, 0, 9 precompute  /  4, 0, 10."""
    expect_cache, expect_nocache = ((3.0, 3.0, 0.0), (4.0, 4.0, 0.0)) if fused else ((3.0, 0.0, 9.0), (4.0, 0.0, 10.0))
    sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=CFG(device, fused))
    sim.step()                                                          # warm-up
    counter, restore = _counting_adjacency()
    try:
        for _ in range(4):
            sim.step()
    finally:
        restore()
    got = (counter[0] / 4, counter[2] / 4, counter[1] / 4)
    assert got == expect_cache, got
    print(f"(e) fused={fused} with cache: adjacency / fused / precompute per step = {got} (== {expect_cache})")
    sim2, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=CFG(device, fused))
    sim2.step()                                                         # warm-up
    orig_wd = _disable_cache(sim2)
    counter2, restore2 = _counting_adjacency()
    try:
        for _ in range(4):
            sim2.step()
    finally:
        restore2()
        sim2._wall_state = orig_wd
    got2 = (counter2[0] / 4, counter2[2] / 4, counter2[1] / 4)
    assert got2 == expect_nocache, got2
    print(f"(e) fused={fused} without cache: {got2} (== {expect_nocache})")


@pytest.mark.parametrize("device", DEVICES)
def test_reuse_does_not_change_the_dynamics(device):
    """(f) two fresh sims, 8 steps each, one with the cache disabled: max|Δ| <= 1e-12 for x, v, rho (the cache returns the same
    lam, G, A; the only difference is the same-code GPU run-to-run spread, measured 8.7e-16 in v and 0 in x, rho)."""
    simA, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=CFG(device))
    simB, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=CFG(device))
    orig_wd = _disable_cache(simB)
    try:
        for _ in range(8):
            simA.step()
            simB.step()
    finally:
        simB._wall_state = orig_wd
    for name in ("x", "v", "rho"):
        d = float((getattr(simA, name) - getattr(simB, name)).abs().max())
        assert d <= 1e-12, (name, d)
        print(f"(f) {name}: max|Δ| = {d:.2e} (<= 1e-12)")


@pytest.mark.parametrize("device", DEVICES)
def test_reuse_invalidation(device):
    """(g) after one step, a second _wall_data at the same state builds nothing; each invalidation case is compared with a fresh
    evaluation (_wallCache = None): a hit is torch.equal, a rebuild <= 1e-13 * max|fresh|; a position/pose/support change rebuilds
    once and the stale cache would have been wrong by |Δlam| > 1e-4 (measured 8.3e-3 / 8.9e-3); a rho change does not rebuild
    (lam, G, A are torch.equal to fresh); a g change does not rebuild but moves A by > 1e-2 (measured 0.59) while lam, G stay
    torch.equal to fresh."""
    sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=CFG(device))
    counter, restore = _counting_adjacency()
    try:
        sim.step()                                                      # one step: no_penetration leaves the cache at the final x
        c0 = counter[0]
        lam1, G1, A1 = sim._wall_data(sim.x, sim.rho)
        b1 = counter[0] - c0
        c0 = counter[0]
        lam2, G2, A2 = sim._wall_data(sim.x, sim.rho)
        b2 = counter[0] - c0
        assert b2 == 0, ("second call built", b2)
        print(f"(g) two _wall_data at the same state: builds = {b1}, {b2} (second builds nothing)")

        def fresh(x, rho):
            sim._wallCache = None
            c0 = counter[0]
            r = sim._wall_data(x, rho)
            return r, counter[0] - c0

        # (i) a position change (x + 1e-3 in y): exactly one build; the stale cache would have been wrong by |Δlam| > 1e-4
        sim._wallCache = None
        lam0, G0, A0 = sim._wall_data(sim.x, sim.rho)                   # clean cache at the current state
        old_lam = sim._wallCache[4].clone()
        x_i = sim.x.clone()
        x_i[:, 1] += 1e-3
        c0 = counter[0]
        lam, G, A = sim._wall_data(x_i, sim.rho)
        builds = counter[0] - c0
        assert builds == 1, builds
        d_lam = float((lam - old_lam).abs().max())
        assert d_lam > 1e-4, d_lam
        ref, _ = fresh(x_i, sim.rho)
        assert float((lam - ref[0]).abs().max()) <= 1e-13 * float(ref[0].abs().max())
        print(f"(g)(i) x + 1e-3 in y: {builds} build, |Δlam| vs stale cache = {d_lam:.2e} (> 1e-4)")

        # (ii) the body rotated by 1e-3 rad: one build; |Δlam| > 1e-4
        sim._wallCache = None
        lam0, G0, A0 = sim._wall_data(sim.x, sim.rho)
        old_lam = sim._wallCache[4].clone()
        body = sim.scene.bodies[0]
        old_angle = float(body.angle)
        body.angle = old_angle + 1e-3
        try:
            c0 = counter[0]
            lam, G, A = sim._wall_data(sim.x, sim.rho)
            builds = counter[0] - c0
            assert builds == 1, builds
            d_lam = float((lam - old_lam).abs().max())
            assert d_lam > 1e-4, d_lam
            ref, _ = fresh(sim.x, sim.rho)
            assert float((lam - ref[0]).abs().max()) <= 1e-13 * float(ref[0].abs().max())
            print(f"(g)(ii) body + 1e-3 rad: {builds} build, |Δlam| vs stale cache = {d_lam:.2e} (> 1e-4)")
        finally:
            body.angle = old_angle

        # (iii) the supports scaled by 1.01: one build
        sim._wallCache = None
        sim._wall_data(sim.x, sim.rho)
        old_Hvec = sim.Hvec.clone()
        sim.Hvec = sim.Hvec * 1.01
        try:
            c0 = counter[0]
            lam, G, A = sim._wall_data(sim.x, sim.rho)
            builds = counter[0] - c0
            assert builds == 1, builds
            ref, _ = fresh(sim.x, sim.rho)
            assert float((lam - ref[0]).abs().max()) <= 1e-13 * float(ref[0].abs().max())
            print(f"(g)(iii) Hvec * 1.01: {builds} build")
        finally:
            sim.Hvec = old_Hvec

        # (iv) a density change (1.1 * rho): no build; lam, G, A torch.equal to a fresh evaluation (they do not depend on rho)
        sim._wallCache = None
        lam0, G0, A0 = sim._wall_data(sim.x, sim.rho)
        c0 = counter[0]
        lam, G, A = sim._wall_data(sim.x, 1.1 * sim.rho)
        builds = counter[0] - c0
        assert builds == 0, builds
        ref, _ = fresh(sim.x, 1.1 * sim.rho)
        assert torch.equal(lam, ref[0]) and torch.equal(G, ref[1]) and torch.equal(A, ref[2])
        print(f"(g)(iv) 1.1 * rho: {builds} builds, lam/G/A torch.equal to fresh")

        # (v) a gravity change: no build; A equals a fresh evaluation and differs from the old A by > 1e-2; lam, G torch.equal
        sim._wallCache = None
        lam0, G0, A0 = sim._wall_data(sim.x, sim.rho)
        old_A = A0.clone()
        old_g = sim.g.clone()
        sim.g = torch.tensor([0.5, -9.0], dtype=TD, device=device)
        try:
            c0 = counter[0]
            lam, G, A = sim._wall_data(sim.x, sim.rho)
            builds = counter[0] - c0
            assert builds == 0, builds
            ref, _ = fresh(sim.x, sim.rho)
            assert torch.equal(A, ref[2])
            assert torch.equal(lam, ref[0]) and torch.equal(G, ref[1])
            d_A = float((A - old_A).abs().max())
            assert d_A > 1e-2, d_A
            print(f"(g)(v) g = [0.5, -9.0]: {builds} builds, A torch.equal to fresh, |ΔA| vs old g = {d_A:.2e} (> 1e-2)")
        finally:
            sim.g = old_g
    finally:
        restore()
    print("(g) invalidation: position / pose / support rebuild once (stale cache wrong by > 1e-4); rho / g do not rebuild")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["dambreak", "sloshing"])
def test_box_domain_fused_equals_scene_operation_path(device, case):
    """the `box` domain (BoxRep: corner tables, the exact polygon for the cone group) runs the fused wall evaluation and gives the state of the `sceneOperation` path (cfg.fusedWall = False) to round-off
    after 60 steps: positions 1e-12, velocities 1e-10 (stated before looking; the two paths evaluate the same tables and differ in summation order)."""
    from warpSPHBoundaries.sim import cases
    res = {}
    for fused in (False, True):
        kw = dict(nx=30, shifting=True, noPen="impulse") if case == "dambreak" else dict(nx=40)
        sim = (cases.marrone_dambreak if case == "dambreak" else cases.sloshing_tank)(domain="box", device=device, fusedWall=fused, graphStep=False, **kw)[0]
        for _ in range(60):
            sim.step()
        res[fused] = (sim.x.clone(), sim.v.clone(), sim._wallCache is not None and type(sim._wallCache[2]).__name__)
    assert res[True][2] == "FusedWall" and res[False][2] != "FusedWall"
    assert float((res[True][0] - res[False][0]).abs().max()) <= 1e-12
    assert float((res[True][1] - res[False][1]).abs().max()) <= 1e-10
