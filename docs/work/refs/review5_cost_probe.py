# reviewer probe for WORK-006: per-call cost of each exact wall operation on a developed dam-break state, and calls per step
import time, math, torch
from edgebound.sim.cases import marrone_dambreak
from edgebound.scene.cover import cover_vector_scene
from edgebound.scene.cone_area import cone_area_scene
from edgebound.scene.tensile import tensile_vector_scene
from edgebound.scene.viscosity import lap_lambda_scene
from edgebound.scene import scene as S
sim,_ = marrone_dambreak(nx=67, shifting=True, noPen="impulse")
for _ in range(300): sim.step()
x, rho = sim.x, sim.rho
lam, G, A = sim._wall_data(x, rho)
near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
c = torch.nn.functional.normalize(torch.randn(len(near),2,dtype=torch.float64,device=x.device),dim=1)
H = sim.H
def t(f, n=20):
    f(); torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1e3
nb = S.Scene.pairMoments
cnt = {"adj":0}
def wrap(*a, **k):
    cnt["adj"] += 1; return nb(*a, **k)
print("N", len(x), "near", len(near), "edges", sum(len(getattr(r,'edges',[])) for b in sim.scene.bodies for r in b.reps))
rows = [("cover_vector_scene", lambda: cover_vector_scene(sim.scene, x[near], H), 3),
        ("cone_area_scene (th/2)", lambda: cone_area_scene(sim.scene, x[near], c, sim.cfg.barecascoThreshold/2, H), 3),
        ("cone_area_scene (pi)", lambda: cone_area_scene(sim.scene, x[near], c, math.pi, H), 3),
        ("tensile_vector_scene w2", lambda: tensile_vector_scene(sim.scene, x[near], H, "w2"), 1),
        ("lap_lambda_scene w2", lambda: lap_lambda_scene(sim.scene, x[near], H, "w2"), 2),
        ("_wall_data (existing, per call)", lambda: sim._wall_data(x, rho), 3)]
tot = 0
for name, f, calls in rows:
    S.Scene.pairMoments = wrap; cnt["adj"]=0; f(); S.Scene.pairMoments = nb
    ms = t(f)
    print("%-32s %7.2f ms/call  adjacency builds/call %d   calls/step %d  -> %6.2f ms/step" % (name, ms, cnt["adj"], calls, ms*calls))
    if not name.startswith("_wall"): tot += ms*calls
print("sum of exact ops per step: %.1f ms" % tot)
