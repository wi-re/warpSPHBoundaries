"""Float32 gate: the production precision (warpSPHCore_PRECISION=float32, a process-wide setting, hence two subprocesses) against float64 on the dam break and the sloshing tank.

Same cases, same steps, graph step on in both.  Measured (2026-10-06, 200 - 300 steps): positions 5e-8, velocities 4e-6, kinetic energy 2e-7 relative.  Gates (a factor ~10 above, stated before a change
can erode them): max |dx| <= 1e-6 (the particle spacing is 1e-2 .. 1e-3), max |dv| <= 5e-5 of the peak speed scale (velocities of order 1), kinetic-energy series within 2e-6 relative.
"""
import json
import os
import subprocess
import sys

import pytest
import warp as wp

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA")

CODE = r'''
import json, sys, numpy as np, torch
import warpSPHBoundaries
from warpSPHBoundaries.sim import cases
out = {}
for name, maker, steps, kw in (("dambreak", cases.marrone_dambreak, 150, dict(nx=40, shifting=True, noPen="impulse")), ("sloshing", cases.sloshing_tank, 120, dict(nx=60))):
    sim = maker(**kw)[0]
    ke = []
    for k in range(steps):
        sim.step()
        if k % 10 == 0:
            ke.append(float(sim.kinetic()))
    out[name] = dict(x=sim.x.cpu().tolist(), v=sim.v.cpu().tolist(), ke=ke, graph=bool(sim._graphed))
json.dump(out, open(sys.argv[1], "w"))
'''


def run(precision, path):
    env = dict(os.environ, warpSPHCore_PRECISION=precision)
    subprocess.run([sys.executable, "-c", CODE, str(path)], check=True, env=env, capture_output=True)
    return json.load(open(path))


def test_float32_matches_float64(tmp_path):
    import numpy as np
    a, b = run("float64", tmp_path / "a.json"), run("float32", tmp_path / "b.json")
    for name in a:
        assert a[name]["graph"] and b[name]["graph"], name
        dx = np.abs(np.array(a[name]["x"]) - np.array(b[name]["x"])).max()
        dv = np.abs(np.array(a[name]["v"]) - np.array(b[name]["v"])).max()
        ke_a, ke_b = np.array(a[name]["ke"]), np.array(b[name]["ke"])
        dke = np.abs(ke_a - ke_b).max() / np.abs(ke_a).max()
        print(f"{name}: max|dx| {dx:.1e} max|dv| {dv:.1e} KE rel {dke:.1e}")
        assert dx <= 1e-6 and dv <= 5e-5 and dke <= 2e-6, (name, dx, dv, dke)
