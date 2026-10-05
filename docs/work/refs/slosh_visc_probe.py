"""Reviewer's probe for WORK-005 (throw-away): sloshing (SPHERIC 10, nx = 200, Wendland C4) truncated at T s with cfg.viscosityExact = True (prototype switch, see lap_scene_probe.py);
prints KE max-rel vs the stored reference slosh_B_nx200 and the wall time.   usage (repo root): python docs/work/refs/slosh_visc_probe.py 1.5"""
import sys, time
sys.path.insert(0, "python")
import numpy as np
from edgebound.sim import validation as dv
from edgebound.sim.validation import ke_relmax
from edgebound import paths
_ke_relmax = lambda series, name: ke_relmax(series, str(paths.results_dir() / 'deltasph' / name))
T = float(sys.argv[1]); t0 = time.time()
sim, info, res = dv.run_sloshing(nx=200, T=T, shifting=True, noPen="impulse", verbose=False, viscosityExact=True)
series = {"t": np.asarray(res["t"]), "ke": np.asarray(res["kineticEnergy"])}
rel, n = _ke_relmax(series, "slosh_B_nx200_series.npz")
print("viscosityExact T=%g: steps %d wall %.0f s  KE rel max vs slosh_B = %.4e (%d samples)  maxVel %.4f  rho [%.5f, %.5f]" % (
    T, res["steps"], time.time() - t0, rel, n, np.asarray(res["maxVelocity"]).max(), np.asarray(res["minDensity"]).min(), np.asarray(res["maxDensity"]).max()), flush=True)
