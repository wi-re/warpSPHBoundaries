"""Snapshot run of the English wedge / flat tank for videos:  python scripts/deltasph_snap.py wedge|tank <dp> <T> <out.npz>   (snapshots every 1/30 s: t, x, v, rho, p, lo, hi, r, h, poly)."""
import sys
import time

import numpy as np

from edgebound.sim.deltasph2d import english_wedge, hydrostatic_tank

what, dp, T, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
sim, info = english_wedge(dp=dp) if what == "wedge" else hydrostatic_tank(dp=dp)
snaps = dict(t=[], x=[], v=[], rho=[], p=[])
nxt, t0 = 0.0, time.time()
while sim.time < T:
    if sim.time >= nxt - 1e-12:
        for k, v in (("t", sim.time), ("x", sim.x.cpu().numpy().copy()), ("v", sim.v.cpu().numpy().copy()), ("rho", sim.rho.cpu().numpy().copy()), ("p", sim.pressure().cpu().numpy().copy())):
            snaps[k].append(v)
        nxt += 1.0 / 30
    sim.step()
bed = info["bed"]
extra = dict(lo=np.array([-1.2, bed]), hi=np.array([1.2, bed + 0.6]), r=dp / 2, h=4 * dp)
if what == "wedge":
    extra["poly"] = info["tri"]
np.savez(out, **{k: np.array(v) for k, v in snaps.items()}, **extra)
print(f"{what} dp={dp}: {len(snaps['t'])} snapshots, wall {time.time() - t0:.0f} s", flush=True)
