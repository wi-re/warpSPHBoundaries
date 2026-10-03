"""Truncated-run regression harness for the delta+-SPH solver (`deltasph2d.py`).

The full validation cases take 5-90 min each; this runs short variants so a later
refactor can be checked in minutes.  It reuses the `deltasph_validation` runners
(`run_tank` / `run_dambreak` / `run_sloshing`) with truncated `T` and records a
small set of scalar metrics per case.

    python -m edgebound.deltasph_regress record [--cases tank,dambreak] [--baseline path]
    python -m edgebound.deltasph_regress check  [--cases tank,dambreak] [--baseline path]

`record` runs each selected case twice (tank, tank, dambreak, dambreak, ...) in one
process; the first run's metrics are the baseline and the per-metric absolute
run-to-run difference is the `spread`.  Tolerance per metric:

    tol = max(20 * spread, 1e-9 * |value|, 1e-12)

(the solver is deterministic only up to GPU reduction order, so `spread` is the
round-off floor and 20x it the guard).  `check` runs each case once and prints one
line per metric (name, value, baseline, tol, margin=|value-baseline|/tol, PASS/FAIL),
exiting 0 iff every metric is within tolerance.

A secondary physics check against the stored reference series is printed in BOTH
`record` and `check` but is NEVER part of the exit code (it is a finding, not a
pass/fail): for the dam break, the max over the common time range of
|KE - KE_B| / max(KE_B) vs `dambreak_B_nx67_series.npz` (interpolated on t) and
|arrival - 2.49| in t*; for sloshing the same KE check vs `slosh_B_nx200_series.npz`.
`--perturb` (negative control) runs the dam break with shifting=False; the defaults
are untouched.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from .deltasph_validation import run_dambreak, run_sloshing, run_tank

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "deltasph")
DEFAULT_BASELINE = os.path.join(RESULTS_DIR, "regress_baseline.json")


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _run_case(name, perturb=False):
    """Run one truncated case; return (metrics dict of floats, series dict for the physics check)."""
    if name == "tank":
        t0 = time.time()
        sim, info, s = run_tank(dp=0.02, T=1.0, domain="surface")
        wall = time.time() - t0
        m = {"rmseBulk": float(s["rmseBulk"]), "rmseNear": float(s["rmseNear"]), "keLast": float(np.asarray(s["ke"])[-1]),
             "rhoMin": float(s["rhoMin"]), "rhoMax": float(s["rhoMax"]), "steps": float(s["steps"])}
        return m, {"t": np.asarray(s["t"]), "ke": np.asarray(s["ke"])}, wall
    if name == "dambreak":
        t0 = time.time()
        # negative control (--perturb): the reference and the baseline both use shifting=True
        sim, info, res = run_dambreak(nx=67, T=0.65, shifting=not perturb, noPen="impulse")
        wall = time.time() - t0
        ts, ke = np.asarray(res["tStar"]), np.asarray(res["kineticEnergy"])
        p0 = np.asarray(res["pProbe0Star"])
        mask = p0 > 0.05
        arrival = float(ts[np.argmax(mask)]) if mask.any() else float("nan")
        m = {"ke_tstar1": float(np.interp(1.0, ts, ke)), "ke_tstar2": float(np.interp(2.0, ts, ke)), "ke_tstar25": float(np.interp(2.5, ts, ke)),
             "p0_arrival_tstar": arrival, "maxVelocityMax": float(np.asarray(res["maxVelocity"]).max()),
             "minDensityMin": float(np.asarray(res["minDensity"]).min()), "maxDensityMax": float(np.asarray(res["maxDensity"]).max()),
             "steps": float(res["steps"])}
        return m, {"t": np.asarray(res["t"]), "tStar": ts, "ke": ke, "arrival": arrival}, wall
    if name == "sloshing":
        t0 = time.time()
        sim, info, res = run_sloshing(nx=200, T=2.6, shifting=True, noPen="impulse")
        wall = time.time() - t0
        t, ke = np.asarray(res["t"]), np.asarray(res["kineticEnergy"])
        m = {"ke_t1": float(np.interp(1.0, t, ke)), "ke_t2": float(np.interp(2.0, t, ke)), "ke_t25": float(np.interp(2.5, t, ke)),
             "maxVelocityMax": float(np.asarray(res["maxVelocity"]).max()), "minDensityMin": float(np.asarray(res["minDensity"]).min()),
             "maxDensityMax": float(np.asarray(res["maxDensity"]).max()), "steps": float(res["steps"])}
        return m, {"t": t, "ke": ke}, wall
    raise ValueError("unknown case: " + name)


def _ke_relmax(series, refname):
    """max over the common time range of |KE - KE_ref| / max(KE_ref), KE_ref interpolated on our t.  (secondary check)."""
    ref = np.load(os.path.join(RESULTS_DIR, refname))
    t_o, ke_o = series["t"], series["ke"]
    t_r, ke_r = ref["t"], ref["kineticEnergy"]
    lo, hi = max(t_o.min(), t_r.min()), min(t_o.max(), t_r.max())
    sel = (t_o >= lo) & (t_o <= hi)
    if not sel.any():
        return float("nan"), 0
    ke_r_at = np.interp(t_o[sel], t_r, ke_r)
    denom = ke_r_at.max()
    if denom <= 0:
        return float("nan"), int(sel.sum())
    return float(np.max(np.abs(ke_o[sel] - ke_r_at) / denom)), int(sel.sum())


def _physics_lines(case, series):
    """Return the printed secondary-check lines (never in the exit code)."""
    out = []
    if case == "dambreak":
        rel, n = _ke_relmax(series, "dambreak_B_nx67_series.npz")
        out.append("  [physics] dam break KE vs dambreak_B_nx67: max|KE-KE_B|/max(KE_B) over common range = %.4e (%d samples)  %s"
                   % (rel, n, "(FINDING: > 5 %)" if rel > 0.05 else ""))
        arr = series["arrival"]
        out.append("  [physics] dam break P1 arrival = %.4f t*  (reference 2.49, |diff| = %.4f)  %s"
                   % (arr, abs(arr - 2.49), "(FINDING: > 0.05)" if (np.isfinite(arr) and abs(arr - 2.49) > 0.05) else ""))
    elif case == "sloshing":
        rel, n = _ke_relmax(series, "slosh_B_nx200_series.npz")
        out.append("  [physics] sloshing KE vs slosh_B_nx200: max|KE-KE_B|/max(KE_B) over common range = %.4e (%d samples)  %s"
                   % (rel, n, "(FINDING: > 5 %)" if rel > 0.05 else ""))
    return out


def _tol(spread, value):
    return max(20.0 * spread, 1e-9 * abs(value), 1e-12)


def cmd_record(cases, baseline_path):
    data, meta_wall = {}, {}
    for case in cases:                       # same process order: tank, tank, dambreak, dambreak, ...
        m1, s1, w1 = _run_case(case)
        m2, s2, w2 = _run_case(case)
        entry = {}
        for k, v in m1.items():
            spread = abs(m1[k] - m2[k])
            entry[k] = {"value": v, "spread": spread, "tol": _tol(spread, v)}
        data[case] = entry
        meta_wall[case] = {"run1_s": w1, "run2_s": w2}
        for ln in _physics_lines(case, s1):
            print(ln, flush=True)
    data["meta"] = {"gitHead": _git_head(), "date": time.strftime("%Y-%m-%d %H:%M:%S"), "wallSeconds": meta_wall,
                    "torch": torch.__version__, "warp": getattr(__import__("warp"), "__version__", "unknown")}
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(data, f, indent=2)
    print("\nwrote baseline: " + baseline_path)
    print("\nspread and relative tolerance (tol/|value|); flags metrics with rel tol > 5 % (weak):")
    for case in cases:
        for k, e in data[case].items():
            rel = (e["tol"] / abs(e["value"])) if e["value"] != 0 else float("inf")
            flag = "  <-- WEAK rel tol > 5 %" if rel > 0.05 else ""
            print("  %-9s %-18s value=% .10g spread=% .3e tol/|v|=% .3e%s" % (case, k, e["value"], e["spread"], rel, flag))
    return 0


def cmd_check(cases, baseline_path, perturb=False):
    with open(baseline_path) as f:
        base = json.load(f)
    all_pass = True
    for case in cases:
        m, series, wall = _run_case(case, perturb=perturb)
        for ln in _physics_lines(case, series):
            print(ln, flush=True)
        print("\n%s  (%.1f s wall%s)" % (case, wall, ", PERTURBED shifting=False" if (perturb and case == "dambreak") else ""))
        for k, v in m.items():
            bv, btol = base[case][k]["value"], base[case][k]["tol"]
            err = abs(v - bv)
            margin = err / btol if btol > 0 else (0.0 if err == 0 else float("inf"))
            ok = err <= btol
            all_pass = all_pass and ok
            print("  %-18s value=% .12g baseline=% .12g tol=% .3e margin=% .3f  %s" % (k, v, bv, btol, margin, "PASS" if ok else "FAIL"))
    print("\nOVERALL: %s" % ("PASS" if all_pass else "FAIL"))
    return 0 if all_pass else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "check"])
    ap.add_argument("--cases", default="tank,dambreak")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--perturb", action="store_true", help="negative control: run the dam break with shifting=False")
    a = ap.parse_args(argv)
    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    if a.mode == "record":
        return cmd_record(cases, a.baseline)
    return cmd_check(cases, a.baseline, perturb=a.perturb)


if __name__ == "__main__":
    sys.exit(main())
