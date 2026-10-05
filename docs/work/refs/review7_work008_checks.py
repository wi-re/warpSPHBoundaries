# NOTE: run on the tree WITH review7_work008_proto.diff applied, from python/ with PYTHONPATH=.
# reviewer probe for WORK-008 (run on the tree with docs/work/refs/review7_work008_proto.diff applied): every number the work document quotes for its tests
import torch, numpy as np
from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
from edgebound.sim import cases, deltasph2d as D
from edgebound.scene import scene as S
from edgebound.edge import warpbc
from edgebound.scene.scene import BodyField, sceneOperation
dev = "cuda:0"
# ---- T8.3 on the small tank (the fixture of tests/sim/test_deltasph.py: hydrostatic_tank dp=0.04, noPen impulse, shifting on)
def tank(**kw):
    sim, info = cases.hydrostatic_tank(dp=0.04, domain="surface", device=dev, cfg=D.DeltaSPHConfig(noPen="impulse", shifting=True, **kw))
    return sim
def counted(sim):
    n = {"b": 0}
    orig = S.Scene.pairMoments
    def wrap(self, *a, **k):
        n["b"] += 1; return orig(self, *a, **k)
    S.Scene.pairMoments = wrap
    return n, orig
for use_cache in (True, False):
    sim = tank()
    for _ in range(3): sim.step()
    if not use_cache:
        o = sim._wall_data
        sim._wall_data = lambda x, rho, o=o: (setattr(sim, "_wallCache", None), o(x, rho))[1]
    n, orig = counted(sim)
    for _ in range(4): sim.step()
    S.Scene.pairMoments = orig
    print("T8.3 small tank, cache %-5s: pairMoments calls per step = %.2f" % (use_cache, n["b"] / 4))
# bit-identity over 8 steps, cache on vs off (two fresh sims, the same initial state)
def run(use_cache, steps=8):
    sim = tank()
    if not use_cache:
        o = sim._wall_data
        sim._wall_data = lambda x, rho, o=o: (setattr(sim, "_wallCache", None), o(x, rho))[1]
    for _ in range(steps): sim.step()
    return sim.x, sim.v, sim.rho
a, b = run(True), run(False)
print("T8.3 8 steps: x, v, rho torch.equal cache on/off:", [bool(torch.equal(p, q)) for p, q in zip(a, b)])
# invalidation
sim = tank(); sim.step(); sim.step()
n, orig = counted(sim)
sim._wall_data(sim.x, sim.rho); sim._wall_data(sim.x, sim.rho); c1 = n["b"]
sim._wall_data(sim.x + 1e-3, sim.rho); c2 = n["b"]
sim.scene.bodies[0].angle = float(sim.scene.bodies[0].angle) + 1e-3; sim._wall_data(sim.x + 1e-3, sim.rho); c3 = n["b"]
S.Scene.pairMoments = orig
print("T8.3 builds: same x twice (after one miss) %d ; moved x +1 = %d ; rotated body +1 = %d   (expected pattern: 1 or 0, 1, 1)" % (c1, c2 - c1, c3 - c2))
# gravity changes, positions do not: A must follow the new g, lam and G come from the cache
sim = tank(); sim.step()
l0, G0, A0 = sim._wall_data(sim.x, sim.rho)
sim.g = torch.tensor([0.5, -9.0], dtype=torch.float64, device=dev)
l1, G1, A1 = sim._wall_data(sim.x, sim.rho)
sim._wallCache = None
l2, G2, A2 = sim._wall_data(sim.x, sim.rho)
print("T8.3 new g: A1 == fresh A2 %s ; A1 != A0 %s ; lam, G equal %s %s" % (bool(torch.equal(A1, A2)), bool((A1 != A0).any()), bool(torch.equal(l1, l2)), bool(torch.equal(G1, G2))))
# ---- T8.2 guard
sim = tank(); x = sim.x; H = sim.H
ps = ParticleState(positions=x, supports=sim.Hvec, masses=torch.full_like(sim.rho, sim.m), kinds=sim.kinds, densities=sim.rho)
pr = OperationProperties(kernel="cone", operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
adj = sim.scene.pairMoments(ps, pr, channels=(3, 4))
one = BodyField(torch.tensor(1.0, dtype=torch.float64, device=dev))
full = sim.scene.pairMoments(ps, pr)
ok = sceneOperation(ps, pr, sim.scene, adj, None, [one], perBody=True); ref = sceneOperation(ps, pr, sim.scene, full, None, [one], perBody=True)
print("T8.2 pruned Gradient == full Gradient (torch.equal):", bool(torch.equal(ok, ref)), " adj.channels", sorted(adj.channels), " full.channels", full.channels)
def tryit(label, f):
    try:
        f(); print("T8.2 guard %-34s -> NO ERROR (bad)" % label)
    except ValueError as e:
        print("T8.2 guard %-34s -> ValueError" % label)
prD = OperationProperties(kernel="cone", operation=WarpOperation.Density, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
prC = OperationProperties(kernel="cone", operation=WarpOperation.Covariance, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
tryit("Density", lambda: sceneOperation(ps, prD, sim.scene, adj, None, [BodyField(rho=1.0)], perBody=True))
tryit("Covariance", lambda: sceneOperation(ps, prC, sim.scene, adj, None, [one], perBody=True))
tryit("Gradient perQuery a1", lambda: sceneOperation(ps, pr, sim.scene, adj, None, [BodyField(torch.zeros(len(x), dtype=torch.float64, device=dev), torch.zeros(len(x), 2, dtype=torch.float64, device=dev), rho=1.0, perQuery=True)], perBody=True))
tryit("Gradient returnReaction", lambda: sceneOperation(ps, pr, sim.scene, adj, None, [one], returnReaction=True))
tryit("channels without 3,4 + Gradient", lambda: sceneOperation(ps, pr, sim.scene, sim.scene.pairMoments(ps, pr, channels=(0,)), None, [one], perBody=True))
# ---- T8.1 / T8.2 plan arithmetic
for kern in ("cone", "lw2", "w2"):
    p = warpbc._device_plan(kern, dev, (3, 4)); print("T8.2 plan terms (3,4) %-5s E %d V %d" % (kern, p.nE, p.nV))
pf = warpbc._device_plan("w2", dev); print("T8.2 full plan w2: E %d V %d" % (pf.nE, pf.nV))
print("T8.1 plan cache identity:", warpbc._device_plan("w2", dev) is warpbc._device_plan("w2", dev), warpbc._device_plan("w2", dev, (3, 4)) is warpbc._device_plan("w2", dev, [4, 3]))
