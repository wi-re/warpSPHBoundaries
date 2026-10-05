"""Reuse of the wall adjacency of no_penetration in the next step's first RHS (WORK-008 T8.3).

`_wall_data` keeps `(x, body poses, adjacency, lam, G, Hvec, kinds)` of its last build and reuses `adj, lam, G` when the
positions, the poses, the supports and the kinds are unchanged; `lam` and `G` do not depend on the densities or the gravity, so
a cache hit is valid for a different `rho` and `g` (`A`, which depends on `g`, is always recomputed from the cached or fresh
adjacency).  Nothing in the solver modifies `lam` or `G` in place (the read-only contract).

Tolerances (stated before looking):
  (e) host-call counts, exact (deterministic): 9 buildAdjacency/step with the cache, 10 without (reviewer measured 9.00 / 10.00).
  (f) max|Δ| <= 1e-12 for x, v, rho: the same-code GPU run-to-run spread was measured at 8.7e-16 in v and 0 in x, rho, so 1e-12
      is three orders above it and six below the 1e-6 effect a stale cache would have.
  (g) a cache hit is torch.equal to a fresh evaluation; a rebuild is max|Δ| <= 1e-13 * max|fresh| (the index_add_ atomics of the
      wall operations are not guaranteed bit-reproducible); the invalidation cases assert the stale-cache error is visible
      (|Δlam| > 1e-4, reviewer measured 8.3e-3 / 8.9e-3) and the g-change moves A by > 1e-2 (reviewer measured 0.59).
"""
import numpy as np
import pytest
import torch
import warp as wp

from edgebound.sim.deltasph2d import DeltaSPHConfig
from edgebound.sim.cases import hydrostatic_tank
from edgebound.scene.scene import Scene

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
TD = torch.float64

CFG = lambda device: DeltaSPHConfig(noPen="impulse", shifting=True)


def _disable_cache(sim):
    """wrap sim._wall_data so the cache is cleared before every call (forces a fresh build each time)."""
    orig = sim._wall_data

    def no_cache(x, rho):
        sim._wallCache = None
        return orig(x, rho)

    sim._wall_data = no_cache
    return orig


def _counting_adjacency():
    """a (counter, restore) pair: counts Scene.buildAdjacency calls (class-level, restored by calling restore())."""
    counter = [0]
    orig = Scene.buildAdjacency

    def counting(self, *a, **k):
        counter[0] += 1
        return orig(self, *a, **k)

    Scene.buildAdjacency = counting

    def restore():
        Scene.buildAdjacency = orig

    return counter, restore


@pytest.mark.parametrize("device", DEVICES)
def test_reuse_saves_one_adjacency_per_step(device):
    """(e) over 4 steps after a warm-up step: exactly 9 buildAdjacency/step with the cache, 10/step with it disabled
    (host-call counts, deterministic)."""
    # with the cache (default)
    sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=CFG(device))
    sim.step()                                                          # warm-up
    counter, restore = _counting_adjacency()
    try:
        for _ in range(4):
            sim.step()
    finally:
        restore()
    per_step = counter[0] / 4
    assert per_step == 9.0, per_step
    print(f"(e) with cache: {counter[0]} buildAdjacency over 4 steps = {per_step:.2f}/step (== 9)")
    # with the cache disabled
    sim2, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=CFG(device))
    sim2.step()                                                         # warm-up
    orig_wd = _disable_cache(sim2)
    counter2, restore2 = _counting_adjacency()
    try:
        for _ in range(4):
            sim2.step()
    finally:
        restore2()
        sim2._wall_data = orig_wd
    per_step2 = counter2[0] / 4
    assert per_step2 == 10.0, per_step2
    print(f"(e) without cache: {counter2[0]} buildAdjacency over 4 steps = {per_step2:.2f}/step (== 10)")


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
        simB._wall_data = orig_wd
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
        old_lam = sim._wallCache[3].clone()
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
        old_lam = sim._wallCache[3].clone()
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
