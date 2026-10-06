# ms/step of the dam break (nx 67) and sloshing (nx 100) on the idle GPU, plus the final state saved for a cross-precision comparison.
#   warpSPHCore_PRECISION=float64 python scripts/studies/step_time_probe.py       then       warpSPHCore_PRECISION=float32 python scripts/studies/step_time_probe.py   (prints f32 vs f64 state differences)
#   python scripts/studies/step_time_probe.py --old-wall     runs with cfg.fusedWall = False (the sceneOperation reference path)
# Reference numbers (2026-10-05, RTX PRO 6000 Blackwell idle): f64 dam break 20.2 / sloshing 12.0 ms/step, f32 17.4 / 9.4 (before step 3: 24.8 / 15.2, before 1b: 30.8 / 22.2).
import os, sys, time
import numpy as np, torch
import edgebound                                                   # noqa: F401  (float64 unless warpSPHCore_PRECISION is set)
from edgebound import paths
from edgebound.edge import precision as PR
from edgebound.sim import cases

fused = "--old-wall" not in sys.argv
graph = "--graph" in sys.argv                                    # fluidWarp + graphStep: the whole step replayed as a CUDA graph (sim/graphstep.py)
tag = PR.real.__name__
out = {}
for name, maker, steps, kw in (("dambreak", cases.marrone_dambreak, 300, dict(nx=67, shifting=True, noPen="impulse")), ("sloshing", cases.sloshing_tank, 200, dict(nx=100))):
    sim = maker(fusedWall=fused, **(dict(fluidWarp=True, graphStep=True) if graph else {}), **kw)[0]
    for _ in range(10): sim.step()
    torch.cuda.synchronize(); t0 = time.time(); ke = []
    for k in range(steps):
        sim.step()
        if k % 20 == 0: ke.append(float(sim.kinetic()))
    torch.cuda.synchronize()
    out[name] = dict(x=sim.x.cpu().numpy(), v=sim.v.cpu().numpy(), ke=np.array(ke), ms=(time.time() - t0) / steps * 1e3)
    print(tag, ("graph" if graph else "fusedWall") if fused else "sceneOperation", name, "%.1f ms/step" % out[name]["ms"], flush=True)
d = paths.tmp_dir(); os.makedirs(d, exist_ok=True)
np.savez(os.path.join(d, "step_time_%s.npz" % tag), **{f"{n}_{k}": v for n, dd in out.items() for k, v in dd.items()})
ref = os.path.join(d, "step_time_float64.npz")
if tag != "float64" and os.path.exists(ref):
    r = np.load(ref)
    for n in out:
        print(f"{n}: {tag} vs float64 max|dx| {np.abs(out[n]['x'] - r[f'{n}_x']).max():.2e}  max|dv| {np.abs(out[n]['v'] - r[f'{n}_v']).max():.2e}  KE series rel diff {np.abs(out[n]['ke'] - r[f'{n}_ke']).max() / np.abs(r[f'{n}_ke']).max():.2e}")
