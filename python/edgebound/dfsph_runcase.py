"""Dam-break snapshot runs for the videos / omniSPH comparison:  cd .tmp/omni (a directory with a `cfg` symlink to omniSPH/cfg for the omni backend);
   python -m edgebound.dfsph_runcase <backend> <r> <T> <out.npz> [key=value ...]
backend: omni | volume | surface | sdf   (ours with that domain representation) | hex (dam break into a spinning hexagon, calibrated lattice, surface loops)  extra key=value go to DFSPHConfig (ours) / are ignored (omni).
Dam break (column 0.2 x 0.8 m at the left, box 1.6 x 1.0 m).  Saves snapshots every `fps` simulated seconds: x, v, rho, p."""
import sys, time

import numpy as np

from . import dfsph_ref as R
from .dfsph2d import DFSPH2D, DFSPHConfig, domain_scene, lattice_calibration

backend, r, T, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
kw = {}
for a in sys.argv[5:]:
    k, v = a.split("=")
    kw[k] = v if v in ("hydrostatic", "linear", "mirror", "wall", "none", "surface", "volume", "sdf") else (v == "True" if v in ("True", "False") else float(v) if "." in v or "e" in v else int(v))
fps = kw.pop("fps", 60)
calib = kw.pop("calib", 0)                      # 1: calibrated rest lattice (V', wall mass 1/S, first row at 0.551 dx); 0: omniSPH conventions
fmin, fmax, top, right = (0.1, 0.1), (0.3, 0.9), 1.0, 1.6
omega = kw.pop("omega", 3.0)
bodies = []
if backend == "hex":                                   # the dam break into a spinning hexagon of dfsph_validation.obstacle (calibrated lattice, surface loop domain)
    from .dfsph_cases import tank_with_obstacle
    sim, info = tank_with_obstacle(L=1.6, H=1.0, fill=0.8, fluidWidth=0.2, Rh=0.06, center=(0.9, 0.11), omega=omega, device="cuda:0", r=r)
    sim.cfg.__dict__.update(kw)
    n, h, lo, hi = len(sim.x), info["h"], info["lo"], np.array([1.6, 1.0])
    c = None
else:
    c = R.omni_case(r, fmin, fmax, top=top, right=right)
if c is not None:
    n, h = c["n"], float(c["h"][0])
    sp = np.array([c["dx"], c["dy"]])
    lo = np.array(fmin) - sp
    hi = np.array([right, top])
snaps = dict(t=[], x=[], v=[], rho=[], p=[], body=[])
dev = "cuda:0"
t0 = time.time()
nextSnap = 0.0
if backend == "omni":
    import omnySPH
    sim = omnySPH.SPHSimulation(R._yaml(r, fmin, fmax, lo - c["epsAdj"], hi + c["epsAdj"], 0.06))
    t = 0.0
    while t < T:
        if t >= nextSnap - 1e-12:
            st = R.omni_state(sim, n)
            for k_, v_ in (("t", t), ("x", st["x"]), ("v", st["v"]), ("rho", st["rho"]), ("p", st["p"])):
                snaps[k_].append(v_)
            nextSnap += 1.0 / fps
        sim.timestep()
        t = sim.getScalar("sim.time")
elif backend == "hex":
    while sim.time < T:
        if sim.time >= nextSnap - 1e-12:
            for k_, v_ in (("t", sim.time), ("x", sim.x.cpu().numpy().copy()), ("v", sim.v.cpu().numpy().copy()), ("rho", sim.rho.cpu().numpy().copy()), ("p", sim.p.cpu().numpy().copy())):
                snaps[k_].append(v_)
            b = sim.scene.bodies[1]
            snaps["body"].append(np.array([float(b.center[0]), float(b.center[1]), float(b.angle)]))
            nextSnap += 1.0 / fps
        sim.step()
else:
    x, V = c["x"], c["V"]
    wm = 1.0
    if calib:
        cal = lattice_calibration(c["dx"], c["dy"], h)
        V, wm = cal["V"], cal["mu"]
        lo = x.min(0) - np.array([cal["dwallX"], cal["dwallY"]])
    cfg = DFSPHConfig(wallMass=wm, **kw)
    sim = DFSPH2D(x, np.zeros_like(x), V, c["h"], domain_scene(backend, lo, hi, h, dev), cfg, dev)
    while sim.time < T:
        if sim.time >= nextSnap - 1e-12:
            for k_, v_ in (("t", sim.time), ("x", sim.x.cpu().numpy().copy()), ("v", sim.v.cpu().numpy().copy()), ("rho", sim.rho.cpu().numpy().copy()), ("p", sim.p.cpu().numpy().copy())):
                snaps[k_].append(v_)
            nextSnap += 1.0 / fps
        sim.step()
snaps = {k: np.array(v) for k, v in snaps.items() if len(v)}
np.savez(out, **snaps, lo=lo, hi=hi, r=r, h=h)
print(f"{backend} r={r} N={n} T={T}: {len(snaps['t'])} snapshots, wall {time.time() - t0:.0f} s", flush=True)
