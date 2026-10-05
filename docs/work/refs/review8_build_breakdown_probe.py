# run: cd python && PYTHONPATH=. python ../docs/work/refs/review8_build_breakdown_probe.py  (GPU idle: the numbers depend on load)
# reviewer probe (REVIEW-008): per-call cost of the 9 Scene.buildAdjacency calls/step, by caller, kernel, channels; dam break nx 67, step 300+
import time, collections, sys, torch
from edgebound.deltasph2d import marrone_dambreak
from edgebound.scene import Scene
sim, _ = marrone_dambreak(nx=67, shifting=True, noPen="impulse")
for _ in range(300): sim.step()
rows = []
orig = Scene.buildAdjacency
def timed(self, ps, pr, channels=None):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    a = orig(self, ps, pr, channels)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1e3
    caller = sys._getframe(1).f_code.co_name
    rows.append((caller, a.kernel, None if channels is None else tuple(channels), dt, a.stats.get("pairs"), len(ps.positions)))
    return a
Scene.buildAdjacency = timed
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(20): sim.step()
torch.cuda.synchronize(); tot = (time.perf_counter() - t0) / 20 * 1e3
Scene.buildAdjacency = orig
agg = collections.defaultdict(list)
for r in rows: agg[(r[0], r[1], r[2])].append(r[3])
print("step (with sync wrappers) %.2f ms; builds/step %.1f" % (tot, len(rows) / 20))
s = 0
for k, v in agg.items():
    print("%-22s kernel %-5s channels %-9s calls/step %.2f  ms/call %.2f  ms/step %.2f" % (k[0], k[1], k[2], len(v) / 20, sum(v) / len(v), sum(v) / 20)); s += sum(v) / 20
print("sum of builds %.2f ms/step = %.0f%% of the step" % (s, 100 * s / tot))
print("pairs/query-count of the last build:", rows[-1][4:], )
