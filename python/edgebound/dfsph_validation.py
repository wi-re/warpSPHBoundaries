"""Validation of DFSPH2D on the scene boundary layer.   cd .tmp/omni (a directory with a `cfg` symlink to omniSPH/cfg);  python -m edgebound.dfsph_validation [omni|reps|obstacle|all]

 omni : tank settling and dam break against the compiled omniSPH (identical initial particles, omniSPH conventions: V = pi r^2, wall face one spacing outside the
        block, Wendland C2, same DFSPH loop); walls = omniSPH's triangle slabs as a `VolumeRep`.
 obstacle : forces on a hexagon (Archimedes, spinning, dam break into it), no omniSPH needed
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


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] != "obstacle"):
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("omni", "all"):
        vs_omni("tank", T=0.3)
        vs_omni("dam", T=0.6)
    if what in ("reps", "all"):
        reps()


# ----------------------------------------------------------------------------------------------------------------------------- bodies in the fluid
def _forces(sim):
    t = np.array([h["t"] for h in sim.history])
    F = np.array([h["pressure"] + h["friction"] for h in sim.history])           # [T, bodies, 2]
    return t, F


def obstacle(dev="cuda:0", out_png=None):
    """forces on a submerged hexagon: Archimedes for a fixed body, a spinning body, momentum bookkeeping, and a dam break running into a spinning hexagon."""
    from .dfsph_cases import tank_with_obstacle
    print("\n### submerged hexagon (R = 0.06 m, A = %.5f m^2) in a tank at rest, r = 0.005, 0.6 s\n" % (1.5 * np.sqrt(3) * 0.06 ** 2))
    print("| case | rep | mean F_y on the body (t > 0.2) | buoyancy rho g A | mean F_x | sum of forces on all bodies / (-weight) | max momentum-balance residual |")
    print("|---|---|---|---|---|---|---|")
    for omega in (0.0, 3.0):
        for rep in ("surface", "volume"):
            sim, info = tank_with_obstacle(omega=omega, rep=rep, device=dev)
            while sim.time < 0.6:
                sim.step()
            t, F = _forces(sim)
            m = t > 0.2
            fo, fd = F[m, 1].mean(0), F[m, 0].mean(0)
            print(f"| omega = {omega} rad/s | {rep} | {fo[1]:.5f} | {info['buoyancy']:.5f} | {fo[0]:+.5f} | {(fo + fd)[1] / -info['weight']:.5f} | {max(h['balance'] for h in sim.history):.1e} |")
    # dam break into a spinning hexagon
    sim2, info = tank_with_obstacle(L=1.6, H=1.0, fill=0.8, fluidWidth=0.2, Rh=0.06, center=(0.9, 0.11), omega=3.0, device=dev, r=0.006)
    nkeep = len(sim2.x)
    while sim2.time < 1.0:
        sim2.step()
    t, F = _forces(sim2)
    fo = F[:, 1]
    i = int(np.argmax(np.abs(fo[:, 0])))
    print(f"\ndam break into a hexagon spinning at 3 rad/s (r = 0.006, {nkeep} particles): peak horizontal force on the hexagon {fo[i, 0]:+.4f} N/m at t = {t[i]:.3f} s,"
          f" peak vertical {fo[np.argmax(np.abs(fo[:, 1])), 1]:+.4f}; max momentum-balance residual {max(h['balance'] for h in sim2.history):.1e}")
    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        ax[0].plot(t, fo[:, 0], label="F_x"); ax[0].plot(t, fo[:, 1], label="F_y"); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("force on the hexagon [N/m]"); ax[0].legend()
        xs = sim2.x.cpu().numpy()
        ax[1].scatter(xs[:, 0], xs[:, 1], s=2, c=sim2.v.norm(dim=1).cpu().numpy()); ax[1].set_aspect("equal"); ax[1].set_title(f"t = {sim2.time:.2f} s")
        fig.tight_layout(); fig.savefig(out_png, dpi=120)
    return sim2


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "obstacle":
    import os
    obstacle(out_png=os.path.join(os.path.dirname(__file__), "..", "..", "results", "dfsph", "dam_hexagon.png"))
