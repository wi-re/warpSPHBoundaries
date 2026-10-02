"""Validation of DeltaSPH2D against warpSPH's `sun2017DeltaSPH`/`deltaSPH` + mDBC.   python -m edgebound.deltasph_validation tank [dp] [T] [domain]

 sloshing : SPHERIC test case 10 (warpSPH `examples/sloshingTank`): rolling tank as rotating gravity, Sensor-1 pressure
 dambreak : Marrone et al. 2011 s.3.1 (warpSPH `probe_deltaSPHMarrone.py`): probes P1-P3, front, KE
 wedge : the same with the sharp wedge on the bed (`probe_englishWedge.py --wedge`): face / apex / base-corner bands of the probe
 tank : English et al. 2022 s.4.1 still water in a flat tank (warpSPH `scripts/probe_englishWedge.py --no-wedge`): per-particle p/(rho0 g H) against the hydrostatic line, kinetic energy history.
        Scored exactly as the probe's `_score`: bulk = 2 dx below the surface and off every wall, near wall = within 2 dx of the bed or a side wall, settled KE = mean over the last 25 % of the record.
"""
import math
import sys
import time

import numpy as np
import torch

from .deltasph2d import english_wedge, hydrostatic_tank, marrone_dambreak, sloshing_probes, sloshing_tank, triangle_distance, wall_probes


def score_tank(sim, info, dp, L=2.4, t=None, ke=None):
    H, g, rho0 = info["Hwater"], info["g"], 1.0
    pgH = rho0 * g * H
    z = (sim.x[:, 1] - info["bed"]).cpu().numpy()
    x = sim.x[:, 0].cpu().numpy()
    p = sim.pressure().cpu().numpy()
    resid = (p - rho0 * g * np.clip(H - z, 0.0, None)) / pgH
    wm = 2.0 * dp
    bulk = (z < H - wm) & (z > wm) & (np.abs(x) < L / 2 - wm)
    near = (z <= wm) | (np.abs(x) >= L / 2 - wm)
    out = dict(rmseBulk=float(np.sqrt(np.mean(resid[bulk] ** 2))), rmseNear=float(np.sqrt(np.mean(resid[near] ** 2))), maxNear=float(np.abs(resid[near]).max()),
               rhoMin=float(sim.rho.min()), rhoMax=float(sim.rho.max()), resid=resid, z=z, x=x)
    if ke is not None:
        t, ke = np.asarray(t), np.asarray(ke)
        tail = t > 0.75 * t[-1]
        half = t > 0.5 * t[-1]
        out.update(keSettled=float(ke[tail].mean()), keMax=float(ke.max()), keTrend=float(np.polyfit(t[half], ke[half], 1)[0]), keBudget=1e-4 * pgH * len(z) * dp * dp)
    return out


def run_tank(dp=0.02, T=4.0, domain="surface", every=50, verbose=True, **cfgkw):
    sim, info = hydrostatic_tank(dp=dp, domain=domain)
    sim.cfg.__dict__.update(cfgkw)
    t, ke = [], []
    t0 = time.time()
    k = 0
    while sim.time < T:
        sim.step()
        k += 1
        if k % every == 0:
            t.append(sim.time)
            ke.append(sim.kinetic())
            if verbose and k % (every * 20) == 0:
                print(f"t={sim.time:.3f} KE={ke[-1]:.2e} vmax={float(sim.v.norm(dim=1).max()):.3f} rho[{float(sim.rho.min()):.4f},{float(sim.rho.max()):.4f}] {time.time() - t0:.0f}s", flush=True)
    s = score_tank(sim, info, dp, t=t, ke=ke)
    s.update(steps=k, wall=time.time() - t0, t=np.array(t), ke=np.array(ke))
    return sim, info, s


def report_tank(s):
    ok = [("hydrostatic profile, bulk", s["rmseBulk"] <= 0.03, f"RMSE {s['rmseBulk']:.4f} of rho g H  (want <= 0.03)"),
          ("hydrostatic profile, near wall/bed", s["rmseNear"] <= 0.08, f"RMSE {s['rmseNear']:.4f}, max |resid| {s['maxNear']:.3f}  (want RMSE <= 0.08)"),
          ("settled kinetic energy small", s["keSettled"] <= s["keBudget"], f"KE tail-mean {s['keSettled']:.3e}  (want <= {s['keBudget']:.3e}); transient peak {s['keMax']:.3e}"),
          ("kinetic energy not growing", s["keTrend"] <= 0.0 or s["keTrend"] < 1e-3 * s["keSettled"], f"2nd-half dKE/dt {s['keTrend']:.3e}"),
          ("weakly compressible", s["rhoMin"] > 0.99 and s["rhoMax"] < 1.05, f"rho in [{s['rhoMin']:.4f}, {s['rhoMax']:.4f}]")]
    for name, good, msg in ok:
        print(("  PASS  " if good else "  FAIL  ") + f"{name} | {msg}")
    return all(g for _, g, _ in ok)


def score_wedge(sim, info, dp, t=None, ke=None):
    """the probe's wedge bands: face (0.3 dp < d < 3 dp off the sloped faces), apex (within 4 dp, outside), base corners (within 4 dp), on top of `score_tank`."""
    out = score_tank(sim, info, dp, L=info["L"], t=t, ke=ke)
    H, g = info["Hwater"], info["g"]
    xy = sim.x.cpu().numpy()
    resid = out["resid"]
    dW = triangle_distance(xy, info["tri"])
    face = (dW > 0.3 * dp) & (dW < 3.0 * dp)
    R = 4.0 * dp
    apex = info["tri"][2]
    near_apex = (np.hypot(xy[:, 0] - apex[0], xy[:, 1] - apex[1]) < R) & (dW > 0)
    dc = np.minimum(np.hypot(xy[:, 0] - info["tri"][0][0], xy[:, 1] - info["tri"][0][1]), np.hypot(xy[:, 0] - info["tri"][1][0], xy[:, 1] - info["tri"][1][1]))
    near_corner = (dc < R) & (dW > 0)
    rm = lambda m: float(np.sqrt(np.mean(resid[m] ** 2))) if m.any() else float("nan")
    mx = lambda m: float(np.abs(resid[m]).max()) if m.any() else float("nan")
    out.update(rmseFace=rm(face), rmseApex=rm(near_apex), maxApex=mx(near_apex), rmseCorner=rm(near_corner), maxCorner=mx(near_corner))
    return out


def run_wedge(dp=0.02, T=4.0, domain="surface", every=50, verbose=True, **cfgkw):
    sim, info = english_wedge(dp=dp, domain=domain)
    sim.cfg.__dict__.update(cfgkw)
    t, ke = [], []
    t0 = time.time()
    k = 0
    while sim.time < T:
        sim.step()
        k += 1
        if k % every == 0:
            t.append(sim.time)
            ke.append(sim.kinetic())
            if verbose and k % (every * 20) == 0:
                print(f"t={sim.time:.3f} KE={ke[-1]:.2e} vmax={float(sim.v.norm(dim=1).max()):.3f} rho[{float(sim.rho.min()):.4f},{float(sim.rho.max()):.4f}] {time.time() - t0:.0f}s", flush=True)
    s = score_wedge(sim, info, dp, t=t, ke=ke)
    s.update(steps=k, wall=time.time() - t0, t=np.array(t), ke=np.array(ke))
    return sim, info, s


def report_wedge(s):
    ok = report_tank(s)
    rows = [("wedge faces", s["rmseFace"] <= 0.05, f"RMSE {s['rmseFace']:.4f} in a 3-dx band off the sloped faces  (want <= 0.05)"),
            ("apex", s["rmseApex"] <= 0.03, f"RMSE {s['rmseApex']:.4f} (max {s['maxApex']:.3f}) within 4 dx of the apex  (want RMSE <= 0.03)"),
            ("base corners", s["rmseCorner"] <= 0.03, f"RMSE {s['rmseCorner']:.4f} (max {s['maxCorner']:.3f}) within 4 dx of a base corner  (want RMSE <= 0.03)")]
    for name, good, msg in rows:
        print(("  PASS  " if good else "  FAIL  ") + f"{name} | {msg}")
    return ok and all(g for _, g, _ in rows)


def run_dambreak(nx=67, T=1.9, every=10, snapDt=None, out=None, verbose=True, **cfgkw):
    """Marrone 3.1: probe series P1-P3 (disc-averaged first-order MLS at the impact wall and 1 dx in), kinetic energy, front, mean height, density range; optional snapshots every `snapDt` s.
    Saved to `out` (.npz) with the keys of warpSPH's probe output where they exist (`tStar`, `pProbe{k}Star`, `pProbe{k}In1Star`, `kineticEnergy`, `maxVelocity`, `minDensity`, `maxDensity`)."""
    sim, info = marrone_dambreak(nx=nx, **cfgkw)
    rows, snaps, nxt, k, t0 = [], dict(t=[], x=[], v=[], rho=[], p=[]), 0.0, 0, time.time()
    while sim.time < T:
        if snapDt and sim.time >= nxt - 1e-12:
            for key, val in (("t", sim.time), ("x", sim.x.cpu().numpy().copy()), ("v", sim.v.cpu().numpy().copy()), ("rho", sim.rho.cpu().numpy().copy()), ("p", sim.pressure().cpu().numpy().copy())):
                snaps[key].append(val)
            nxt += snapDt
        sim.step()
        k += 1
        if k % every == 0:
            pw, pi = wall_probes(sim, info)
            xs = sim.x
            rows.append([sim.time, sim.time * math.sqrt(info["g"] / info["H"]), sim.kinetic(), float(sim.v.norm(dim=1).max()), float(sim.rho.min()), float(sim.rho.max()),
                         float(xs[:, 0].max()), float(xs[:, 1].mean()), float(sim.dt), *pw, *pi])
            if verbose and k % (every * 100) == 0:
                print(f"t={sim.time:.3f} t*={rows[-1][1]:.2f} KE={rows[-1][2]:.3f} vmax={rows[-1][3]:.2f} rho[{rows[-1][4]:.4f},{rows[-1][5]:.4f}] P*={pw.round(3)} {time.time() - t0:.0f}s", flush=True)
    a = np.array(rows)
    res = dict(t=a[:, 0], tStar=a[:, 1], kineticEnergy=a[:, 2], maxVelocity=a[:, 3], minDensity=a[:, 4], maxDensity=a[:, 5], front=a[:, 6], meanY=a[:, 7], dt=a[:, 8], steps=k, wall=time.time() - t0)
    for q in range(3):
        res[f"pProbe{q}Star"], res[f"pProbe{q}In1Star"] = a[:, 9 + q], a[:, 12 + q]
    if out:
        np.savez(out, **res, **({f"snap_{key}": np.array(v) for key, v in snaps.items()} if snapDt else {}), lo=np.array([info["xl"], info["yb"]]), hi=np.array([info["xr"], info["yt"]]), r=info["dx"] / 2, h=4 * info["dx"])
    return sim, info, res


def report_dambreak(res, ref=None):
    ts = res["tStar"]
    print(f"{res['steps']} steps, {res['wall']:.0f} s; rho in [{res['minDensity'].min():.4f}, {res['maxDensity'].max():.4f}], max|v| {res['maxVelocity'].max():.2f}")
    for q, name in enumerate(("P1", "P2", "P3")):
        for suffix, label in (("Star", "wall"), ("In1Star", "1dx in")):
            for src, r_ in (("ours", res), ("warpSPH", ref)):
                if r_ is None:
                    continue
                p, t = np.asarray(r_[f"pProbe{q}{suffix}"]), np.asarray(r_["tStar"])
                ok = np.isfinite(p)
                arr = t[ok][np.argmax(p[ok] > 0.05)] if (ok & (p > 0.05)).any() else float("nan")
                m1, m2 = ok & (t >= 3.2) & (t <= 4.8), ok & (t >= 5.2) & (t <= 6.1)
                print(f"  {name} {label:6s} {src:8s} first P*>0.05 at t*={arr:.2f} | mean[3.2,4.8] {p[m1].mean() if m1.any() else float('nan'):.3f} | mean[5.2,6.1] {p[m2].mean() if m2.any() else float('nan'):.3f} | max {np.nanmax(p):.3f} at t*={t[ok][np.nanargmax(p[ok])]:.2f}")


def run_sloshing(nx=200, T=7.0, every=10, snapDt=None, out=None, verbose=True, **cfgkw):
    """SPHERIC test case 10: Sensor-1 pressure series every `every` steps (dt = 1e-4: 1 ms), kinetic energy, density range, roll angle, optional snapshots.  Keys follow warpSPH's series file where they exist
    (`t`, `sensorPressureProbe` = the Gaussian Tait probe, `rollAngleDeg`, `kineticEnergy`, `maxVelocity`, `minDensity`, `maxDensity`); `sensorPressureMLS` is the wall MLS probe."""
    sim, info = sloshing_tank(nx=nx, **cfgkw)
    rows, snaps, nxt, k, t0 = [], dict(t=[], x=[], v=[], rho=[], p=[]), 0.0, 0, time.time()
    troll, throll, _ = info["roll"]
    while sim.time < T:
        if snapDt and sim.time >= nxt - 1e-12:
            for key, val in (("t", sim.time), ("x", sim.x.cpu().numpy().copy()), ("v", sim.v.cpu().numpy().copy()), ("rho", sim.rho.cpu().numpy().copy()), ("p", sim.pressure().cpu().numpy().copy())):
                snaps[key].append(val)
            nxt += snapDt
        sim.step()
        k += 1
        if k % every == 0:
            pg, pm = sloshing_probes(sim, info)
            rows.append([sim.time, pg, pm, math.degrees(float(np.interp(sim.time, troll, throll))), sim.kinetic(), float(sim.v.norm(dim=1).max()), float(sim.rho.min()), float(sim.rho.max())])
            if verbose and k % (every * 200) == 0:
                print(f"t={sim.time:.3f} p_probe={pg:.0f} Pa roll={rows[-1][3]:.2f} deg KE={rows[-1][4]:.3e} vmax={rows[-1][5]:.2f} rho[{rows[-1][6]:.3f},{rows[-1][7]:.3f}] {time.time() - t0:.0f}s", flush=True)
    a = np.array(rows)
    res = dict(t=a[:, 0], sensorPressureProbe=a[:, 1], sensorPressureMLS=a[:, 2], rollAngleDeg=a[:, 3], kineticEnergy=a[:, 4], maxVelocity=a[:, 5], minDensity=a[:, 6], maxDensity=a[:, 7], steps=k, wall=time.time() - t0)
    if out:
        np.savez(out, **res, **({f"snap_{key}": np.array(v) for key, v in snaps.items()} if snapDt else {}), lo=np.array([-info["L"] / 2, 0.0]), hi=np.array([info["L"] / 2, 0.25]), r=info["dx"] / 2, h=4 * info["dx"])
    return sim, info, res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "tank"
    if what == "tank":
        dp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
        T = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
        dom = sys.argv[4] if len(sys.argv) > 4 else "surface"
        sim, info, s = run_tank(dp, T, dom)
        print(f"\nflat tank, dp = {dp}, domain = {dom}, {s['steps']} steps, {s['wall']:.0f} s")
        report_tank(s)
    if what == "sloshing":
        nx = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        T = float(sys.argv[3]) if len(sys.argv) > 3 else 7.0
        out = sys.argv[4] if len(sys.argv) > 4 else None
        kw = {k: (v == "True" if v in ("True", "False") else (v if not v.replace(".", "").replace("e-", "").isdigit() else float(v))) for k, v in (a.split("=") for a in sys.argv[5:])}
        sim, info, res = run_sloshing(nx, T, snapDt=(1 / 60) if out else None, out=out, **kw)
        print(f"{res['steps']} steps, {res['wall']:.0f} s")
    if what == "dambreak":
        nx = int(sys.argv[2]) if len(sys.argv) > 2 else 67
        T = float(sys.argv[3]) if len(sys.argv) > 3 else 1.9
        out = sys.argv[4] if len(sys.argv) > 4 else None
        kw = {k: (v == "True" if v in ("True", "False") else (v if not v.replace(".", "").replace("e-", "").isdigit() else float(v))) for k, v in (a.split("=") for a in sys.argv[5:])}
        sim, info, res = run_dambreak(nx, T, snapDt=(1 / 60) if out else None, out=out, **kw)
        report_dambreak(res)
    if what == "wedge":
        dp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
        T = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
        dom = sys.argv[4] if len(sys.argv) > 4 else "surface"
        sim, info, s = run_wedge(dp, T, dom)
        print(f"\nEnglish wedge, dp = {dp}, domain = {dom}, {len(sim.x)} particles, {s['steps']} steps, {s['wall']:.0f} s")
        report_wedge(s)
