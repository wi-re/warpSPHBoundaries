# reviewer probe for WORK-006: ms/step, default (polar) vs all four exact switches, dam break nx=67 and sloshing nx=200
import time, sys, torch
from edgebound.sim.cases import marrone_dambreak, sloshing_tank
def run(build, warm, timed, **kw):
    sim, info = build(**kw)
    for _ in range(warm): sim.step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(timed): sim.step()
    torch.cuda.synchronize(); return (time.time()-t0)/timed*1e3, len(sim.x)
ALL = dict(coverExact=True, coneExact=True, tensileExact=True, viscosityExact=True)
for name, build, warm, timed in (("dambreak nx67", lambda **k: marrone_dambreak(nx=67, shifting=True, noPen="impulse", **k), 100, 100),
                                 ("sloshing nx200", lambda **k: sloshing_tank(nx=200, shifting=True, noPen="impulse", **k), 50, 50)):
    for lab, kw in (("default", {}), ("all four exact", ALL)):
        ms, n = run(build, warm, timed, **kw)
        print("%-15s %-15s N=%d  %.2f ms/step" % (name, lab, n, ms), flush=True)
