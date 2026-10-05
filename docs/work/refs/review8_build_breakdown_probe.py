# run: cd python && PYTHONPATH=. python ../docs/work/refs/review8_build_breakdown_probe.py  (GPU idle: the numbers depend on load)
# reviewer probe (REVIEW-008): per-call cost of the Scene.adjacency / Scene.precompute calls/step (before 2b: 9 buildAdjacency), by caller, kernel, channels; dam break nx 67, step 300+
import time, collections, sys, torch
from edgebound.sim.cases import marrone_dambreak
from edgebound.scene.scene import Scene
sim, _ = marrone_dambreak(nx=67, shifting=True, noPen="impulse")
for _ in range(300): sim.step()
rows = []
oA, oP = Scene.adjacency, Scene.precompute
def timed_a(self, ps, pr):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    a = oA(self, ps, pr)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1e3
    rows.append(("adjacency", sys._getframe(1).f_code.co_name, "-", None, dt, None, len(ps.positions)))
    return a
def timed_p(self, adj, pr, channels=None):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    m = oP(self, adj, pr, channels)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1e3
    rows.append(("precompute", sys._getframe(1).f_code.co_name, m.kernel, None if channels is None else tuple(channels), dt, m.stats.get("pairs"), adj.numQueries))
    return m
Scene.adjacency, Scene.precompute = timed_a, timed_p
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(20): sim.step()
torch.cuda.synchronize(); tot = (time.perf_counter() - t0) / 20 * 1e3
Scene.adjacency, Scene.precompute = oA, oP
agg = collections.defaultdict(list)
for r in rows: agg[(r[0], r[1], r[2], r[3])].append(r[4])
print("step (with sync wrappers) %.2f ms; adjacencies/step %.1f, precomputes/step %.1f" % (tot, sum(r[0] == "adjacency" for r in rows) / 20, sum(r[0] == "precompute" for r in rows) / 20))
s = 0
for k, v in agg.items():
    print("%-10s %-22s kernel %-5s channels %-9s calls/step %.2f  ms/call %.2f  ms/step %.2f" % (k[0], k[1], k[2], k[3], len(v) / 20, sum(v) / len(v), sum(v) / 20)); s += sum(v) / 20
print("sum of adjacency + precompute %.2f ms/step = %.0f%% of the step" % (s, 100 * s / tot))
