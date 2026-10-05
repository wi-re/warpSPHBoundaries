# evaluation schedule of one step: every Scene.adjacency / Scene.precompute / sceneOperation call, by consumer, position set, kernel, channels, operation.
# counts only (no timings): safe on a loaded GPU.   python docs/work/refs/schedule_probe.py [dambreak|sloshing]
import sys, inspect, hashlib, collections
import torch
from edgebound.scene import scene as S
from edgebound.sim import cases

case = sys.argv[1] if len(sys.argv) > 1 else "dambreak"
sim = cases.marrone_dambreak(nx=40, shifting=True, noPen="impulse")[0] if case == "dambreak" else cases.sloshing_tank(nx=60)[0]
for _ in range(5): sim.step()

log, ids = [], {}
pos_store = {}
def pid(ps):
    pos_store.setdefault(len(ids), None)
    h = hashlib.md5(ps.positions.detach().cpu().numpy().tobytes()).hexdigest()[:6]
    if h not in ids: ids[h] = "P%d" % len(ids); pos_store[ids[h]] = ps.positions.detach().clone()
    return ids[h]
def caller():
    out = []
    for f in inspect.stack()[2:]:
        if "edgebound" in f.filename and not f.function.startswith(("buildAdjacency", "sceneOperation", "_wall_op", "B", "O")):
            out.append(f.function)
        if len(out) == 3: break
    return "<".join(out)
oA, oP, oO = S.Scene.adjacency, S.Scene.precompute, S.sceneOperation
def A(self, ps, props):
    a = oA(self, ps, props)
    log.append(("adj", pid(ps), caller(), "-", None, "", id(a)))
    return a
def P(self, adjacency, props, channels=None):
    m = oP(self, adjacency, props, channels)
    log.append(("pre", "-", caller(), S.kernelName(props.kernel), None if channels is None else tuple(channels), "", id(m)))
    return m
def O(ps, props, scene, adjacency=None, qv=None, flds=None, returnReaction=False, perBody=False):
    log.append(("op", pid(ps), caller(), S.kernelName(props.kernel), None, "%s/%s%s" % (props.operation.name, props.gradientMode.name, "+reaction" if returnReaction else ""), id(adjacency) if adjacency is not None else 0))
    return oO(ps, props, scene, adjacency, qv, flds, returnReaction, perBody)
S.Scene.adjacency = A; S.Scene.precompute = P; S.sceneOperation = O
import edgebound.sim.deltasph2d as D, edgebound.scene.cover as C, edgebound.scene.tensile as T, edgebound.scene.viscosity as V
for m in (D, C, T, V):
    if hasattr(m, "sceneOperation"): m.sceneOperation = O
log.clear(); ids.clear()
sim.step()
print("case", case, "N =", len(sim.x))
for r in log: print("%-5s %-3s %-52s %-5s ch=%-7s %-22s adj=%x" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6] & 0xffff))
print("adjacencies", sum(r[0] == "adj" for r in log), "precomputes", sum(r[0] == "pre" for r in log), "position sets", len(ids), "ops", sum(r[0] == "op" for r in log))
names = sorted((k for k in pos_store if isinstance(k, str)), key=lambda k: int(k[1:]))
print("query set sizes:", {a: len(pos_store[a]) for a in names})
print("max |x_a - x_b| / H between same-size sets (H = %.4g):" % float(sim.H))
for a in names:
    print(a, " ".join(("%8.2e" % (float((pos_store[a] - pos_store[b]).abs().max()) / float(sim.H))) if len(pos_store[a]) == len(pos_store[b]) else "     -   " for b in names))
# is every small query set a row-subset (same positions, same supports) of the full set at the same positions?
sup_store = {}
for a, b in [p for p in (("P1", "P0"), ("P3", "P2"), ("P5", "P4")) if p[0] in pos_store]:
    A, Bm = pos_store[a], pos_store[b]
    d = torch.cdist(A, Bm, compute_mode='donot_use_mm_for_euclid_dist'); m, idx = d.min(1)
    print("%s subset of %s: max row distance %.1e; %d of %d rows matched exactly" % (a, b, float(m.max()), int((m == 0).sum()), len(A)))
