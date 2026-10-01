"""Validation of DFSPH2D on the scene boundary layer.   cd .tmp/omni (a directory with a `cfg` symlink to omniSPH/cfg);  python -m edgebound.dfsph_validation [omni|reps|all]

 omni : tank settling and dam break against the compiled omniSPH (identical initial particles, omniSPH conventions: V = pi r^2, wall face one spacing outside the
        block, Wendland C2, same DFSPH loop); walls = omniSPH's triangle slabs as a `VolumeRep`.
 reps : the same calibrated rest lattice / dam break with the domain as VolumeRep, SurfaceRep, SdfRep (tier 3 + corner fallback) and four half planes (control).
"""
import sys
import time

import numpy as np
import torch

from . import dfsph_ref as R
from .dfsph2d import DFSPH2D, DFSPHConfig, domain_scene, lattice_calibration


def _series(sim, rows):
    rows.append((sim.time, float(sim.v.norm(dim=1).max()), float(sim.x[:, 1].mean()), float(sim.x[:, 0].max()), float(sim.x[:, 0].mean()),
                 sim.wallForce[0].item(), sim.wallForce[1].item()))


def _at(a, t):
    return a[min(np.searchsorted(a[:, 0], t - 1e-12), len(a) - 1)]


def vs_omni(case, r=0.005, T=0.6, every=10, dev="cuda:0"):
    import omnySPH
    fmin, fmax, top, right = ((0.02, 0.02), (0.98, 0.25), 0.5, None) if case == "tank" else ((0.1, 0.1), (0.3, 0.9), 1.0, 1.6)
    c = R.omni_case(r, fmin, fmax, top=top, right=right)
    sp = np.array([c["dx"], c["dy"]])
    lo = np.array(fmin) - sp
    hi = np.array([fmax[0] + sp[0] if right is None else right, top])
    so = omnySPH.SPHSimulation(R._yaml(r, fmin, fmax, lo - c["epsAdj"], hi + c["epsAdj"], 0.06))
    n, h = c["n"], float(c["h"][0])
    out = {}
    for wp in ["hydrostatic", "linear"]:
        sim = DFSPH2D(c["x"], np.zeros_like(c["x"]), c["V"], c["h"], domain_scene("volume", lo, hi, h, dev), DFSPHConfig(wallPressure=wp), dev)
        rows = []
        while sim.time < T:
            sim.step()
            _series(sim, rows)
        out["ours/" + wp] = np.array(rows)
    rows, k, t = [], 0, 0.0
    while t < T:
        so.timestep()
        k += 1
        t = so.getScalar("sim.time")
        if k % every == 0:
            st = R.omni_state(so, n)
            rows.append((t, np.linalg.norm(st["v"], axis=1).max(), st["x"][:, 1].mean(), st["x"][:, 0].max(), st["x"][:, 0].mean(), np.nan, np.nan))
    out["omniSPH"] = np.array(rows)
    print(f"\n### {case}: r = {r}, N = {n}, h = {h:.4f}\n")
    times = [0.1, 0.2, 0.3, 0.4, 0.5, T] if case == "dam" else [0.05, 0.1, 0.2, T]
    print("| model | " + " | ".join(f"t={t:.2f}: front x / mean y / v_max" for t in times) + " |")
    print("|---|" + "---|" * len(times))
    for k_, a in out.items():
        print(f"| {k_} | " + " | ".join("%.4f / %.4f / %.2f" % (_at(a, t)[3], _at(a, t)[2], _at(a, t)[1]) for t in times) + " |")
    return out


def reps(r=0.005, T=0.6, dev="cuda:0", snaps=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
    fmin, fmax, top, right = (0.1, 0.1), (0.3, 0.9), 1.0, 1.6
    c = R.omni_case(r, fmin, fmax, top=top, right=right)
    x, h = c["x"], float(c["h"][0])
    cal = lattice_calibration(c["dx"], c["dy"], h)
    lo = x.min(0) - np.array([cal["dwallX"], cal["dwallY"]])
    hi = np.array([right, top])
    res = {}
    for kind in ["surface", "volume", "sdf", "halfplanes"]:
        sim = DFSPH2D(x, np.zeros_like(x), cal["V"], c["h"], domain_scene(kind, lo, hi, h, dev), DFSPHConfig(wallMass=cal["mu"]), dev)
        rows, sn, t0 = [], {}, time.time()
        while sim.time < T - 1e-12:
            sim.step()
            _series(sim, rows)
            for s in snaps:
                if s not in sn and sim.time >= s - 1e-12:
                    sn[s] = sim.x.cpu().numpy().copy()
        res[kind] = (np.array(rows), sn, time.time() - t0)
    print(f"\n### representations, calibrated lattice, dam break r = {r}\n")
    print("| domain representation | pair-set type | time |" + " | ".join(f"t={s:.1f}" for s in snaps) + " |")
    print("|---|---|---|" + "---|" * len(snaps))
    kinds = {"surface": "edge terms + indicator", "volume": "triangle slabs (P1 weights)", "sdf": "tier 3 planar + corner fallback surface", "halfplanes": "4 x tier 3 half plane (corners double counted)"}
    for k, (a, sn, wall) in res.items():
        d = [np.sqrt(((sn[s] - res["surface"][1][s]) ** 2).sum(1).mean()) for s in snaps]
        print(f"| {k} | {kinds[k]} | {wall:.0f} s | " + " | ".join("%.1e" % v for v in d) + " |")
    print("\n(rms particle displacement from the `surface` run)\n")
    print("| t | " + " | ".join(res) + " |   (front x)")
    print("|---|" + "---|" * len(res))
    for s in snaps:
        print(f"| {s:.1f} | " + " | ".join("%.4f" % _at(a, s)[3] for a, _, _ in res.values()) + " |")
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("omni", "all"):
        vs_omni("tank", T=0.3)
        vs_omni("dam", T=0.6)
    if what in ("reps", "all"):
        reps()
