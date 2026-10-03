"""Reviewer's probe for WORK-004 (throw-away; write your own code).  Sloshing (SPHERIC 10, nx = 200, Wendland C4) truncated at T s: default config vs the exact C4 wall tensile
term (stable `w4p5` kernel monkeypatched into warpbc.edge_channels, solver `shift` patched as in tensile_c4_probe.py).  Prints the KE max-relative difference to the stored
reference slosh_B_nx200 (the harness' secondary check, limit 5 %), between the two runs, and the wall time.   usage: python slosh_gate_probe.py T default|exact"""
import sys, time, types
sys.path.insert(0, "python"); sys.path.insert(0, "docs/work/refs")
import numpy as np
T, which = float(sys.argv[1]), sys.argv[2]
from edgebound import deltasph2d, deltasph_validation as dv
from edgebound.deltasph_regress import _ke_relmax
if which == "exact":
    import importlib
    ns = {}
    exec(open("docs/work/refs/tensile_c4_probe.py").read().split("# (0)")[0], ns)          # registers w4p5, patches warpbc.edge_channels, defines T_scene
    src = open("python/edgebound/deltasph2d.py").read()
    src = src.replace('                    if cfg.kernel != KernelFunctions.Wendland2:  raise NotImplementedError("tensileExact: Wendland C2 only (C4 needs the Chebyshev plan)")\n', "")
    src = src.replace("T = tensile_vector_scene(self.scene, x[near], H)", "T = _TE(self.scene, x[near], H)")
    mod = types.ModuleType("edgebound.deltasph2d_probe"); mod.__package__ = "edgebound"; mod.__dict__["_TE"] = ns["T_scene"]
    exec(compile(src, "deltasph2d_probe", "exec"), mod.__dict__)
    dv.sloshing_tank = lambda *a, **k: (lambda sim_info: (setattr(sim_info[0], "__class__", mod.DeltaSPH2D), sim_info)[1])(deltasph2d.sloshing_tank(*a, **k))
kw = dict(tensileExact=True) if which == "exact" else {}
t0 = time.time()
sim, info, res = dv.run_sloshing(nx=200, T=T, shifting=True, noPen="impulse", verbose=False, **kw)
wall = time.time() - t0
series = {"t": np.asarray(res["t"]), "ke": np.asarray(res["kineticEnergy"])}
rel, n = _ke_relmax(series, "slosh_B_nx200_series.npz")
np.savez(".tmp/slosh_probe_%s_T%g.npz" % (which, T), t=series["t"], ke=series["ke"], p=np.asarray(res["sensorPressureProbe"]), vmax=np.asarray(res["maxVelocity"]), rhomin=np.asarray(res["minDensity"]), rhomax=np.asarray(res["maxDensity"]))
print("%s T=%g: steps %d  wall %.0f s  KE rel max vs slosh_B = %.4e (%d samples)  maxVel %.3f  rho [%.4f, %.4f]" % (
    which, T, res["steps"], wall, rel, n, np.asarray(res["maxVelocity"]).max(), np.asarray(res["minDensity"]).min(), np.asarray(res["maxDensity"]).max()), flush=True)
