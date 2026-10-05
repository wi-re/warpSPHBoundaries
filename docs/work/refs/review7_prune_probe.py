# NOTE: run on the tree WITH review7_work008_proto.diff applied (`git apply docs/work/refs/review7_work008_proto.diff`), from python/ with PYTHONPATH=.
# reviewer probe for WORK-008: cover / tensile with the pruned adjacency (channels 3,4) vs the full adjacency: bit-identity and cost (dam-break state)
import time, torch
from edgebound.sim.deltasph2d import marrone_dambreak
from edgebound.scene.cover import cover_vector_scene
from edgebound.scene.tensile import tensile_vector_scene
from edgebound.scene import scene as S
from edgebound.edge import warpbc
sim, _ = marrone_dambreak(nx=67, shifting=True, noPen="impulse")
for _ in range(300): sim.step()
x, rho, H = sim.x, sim.rho, sim.H
lam, G, A = sim._wall_data(x, rho)
near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
def T(f, n=30):
    f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t0) / n * 1e3
orig = S.Scene.buildAdjacency
def full(*a, **k):
    k.pop("channels", None); return orig(*a, **k)
res = {}
for name, f in (("cover", lambda: cover_vector_scene(sim.scene, x[near], H)), ("tensile w2", lambda: tensile_vector_scene(sim.scene, x[near], H, "w2")), ("tensile w4", lambda: tensile_vector_scene(sim.scene, x[near], H, "w4"))):
    S.Scene.buildAdjacency = orig; tp = T(f); op = f()
    S.Scene.buildAdjacency = full; tf = T(f); of = f()
    S.Scene.buildAdjacency = orig
    print("%-11s pruned %6.2f ms   full %6.2f ms   bit-identical %s  (max|diff| %.1e)" % (name, tp, tf, bool(torch.equal(op, of)), float((op - of).abs().max())))
