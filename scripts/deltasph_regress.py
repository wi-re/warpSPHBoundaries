"""Truncated-run regression harness for the delta+-SPH solver (`deltasph2d.py`).

The full validation cases take 5-90 min each; this runs short variants so a later
refactor can be checked in minutes.  It reuses the `warpSPHBoundaries.sim.validation` runners
(`run_tank` / `run_dambreak` / `run_sloshing`) with truncated `T` and records a
small set of scalar metrics per case.

    python scripts/deltasph_regress.py record [--cases tank,dambreak] [--baseline path]
    python scripts/deltasph_regress.py check  [--cases tank,dambreak] [--baseline path]

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

`--cfg key=value[,key=value...]` (values: true/false -> bool, else float if parsable, else the string) is
passed as `**cfgkw` to the runners (e.g. `--cfg alpha=0.5`, `--cfg wallViscosityForm=pairwise`); the cfg dict is printed at
the top of `check` and stored under `meta` by `record`.  `--physics` (check only): the exit code then depends
ONLY on the physics gate (fixed limits, one line per item: name, value, limit, PASS/FAIL); the bit-level lines
are still printed, for information.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from warpSPHBoundaries.sim.validation import ke_relmax, run_dambreak, run_sloshing, run_tank

from warpSPHBoundaries import paths

REPO_ROOT = str(paths.REPO_ROOT)
RESULTS_DIR = str(paths.results_dir() / "deltasph")
DEFAULT_BASELINE = os.path.join(RESULTS_DIR, "regress_baseline.json")


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _run_case(name, perturb=False, **cfgkw):
    """Run one truncated case; return (metrics dict of floats, series dict for the physics check)."""
    if name == "tank":
        t0 = time.time()
        sim, info, s = run_tank(dp=0.02, T=1.0, domain="surface", **cfgkw)
        wall = time.time() - t0
        m = {"rmseBulk": float(s["rmseBulk"]), "rmseNear": float(s["rmseNear"]), "keLast": float(np.asarray(s["ke"])[-1]),
             "rhoMin": float(s["rhoMin"]), "rhoMax": float(s["rhoMax"]), "steps": float(s["steps"])}
        return m, {"t": np.asarray(s["t"]), "ke": np.asarray(s["ke"])}, wall
    if name == "dambreak":
        t0 = time.time()
        # negative control (--perturb): the reference and the baseline both use shifting=True
        sim, info, res = run_dambreak(nx=67, T=0.65, shifting=not perturb, noPen="impulse", **cfgkw)
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
        sim, info, res = run_sloshing(nx=200, T=2.6, shifting=True, noPen="impulse", **cfgkw)
        wall = time.time() - t0
        t, ke = np.asarray(res["t"]), np.asarray(res["kineticEnergy"])
        m = {"ke_t1": float(np.interp(1.0, t, ke)), "ke_t2": float(np.interp(2.0, t, ke)), "ke_t25": float(np.interp(2.5, t, ke)),
             "maxVelocityMax": float(np.asarray(res["maxVelocity"]).max()), "minDensityMin": float(np.asarray(res["minDensity"]).min()),
             "maxDensityMax": float(np.asarray(res["maxDensity"]).max()), "steps": float(res["steps"])}
        return m, {"t": t, "ke": ke}, wall
    raise ValueError("unknown case: " + name)


def _ke_relmax(series, refname):
    return ke_relmax(series, os.path.join(RESULTS_DIR, refname))


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


def _parse_cfg(s):
    """'key=value[,key=value...]' -> dict (the --cfg flag).  Value: true/false -> bool, else float(...) if parsable, else the string."""
    d = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        k, sep, v = item.partition("=")
        if not sep:
            raise ValueError("--cfg items must be key=value: " + item)
        v = v.strip()
        if v.lower() == "true":
            d[k.strip()] = True
        elif v.lower() == "false":
            d[k.strip()] = False
        else:
            try:
                d[k.strip()] = float(v)
            except ValueError:
                d[k.strip()] = v
    return d


def _gate_lines(case, m, series, base):
    """The physics gate (T2.3): one line per item (name, value, limit, PASS/FAIL); returns (lines, all_ok).
    The limits are fixed in docs/work/WORK-002.md -- do not change them."""
    out = []
    ok = True

    def add(name, value, okv, limit):
        nonlocal ok
        out.append("  [gate] %-20s value=% .12g  limit: %s  %s" % (name, value, limit, "PASS" if okv else "FAIL"))
        ok = ok and okv

    if case == "dambreak":
        rel, _ = _ke_relmax(series, "dambreak_B_nx67_series.npz")
        add("KE rel max vs B", rel, rel <= 0.05, "<= 0.05")
        arr = series["arrival"]
        darr = abs(arr - 2.49) if np.isfinite(arr) else float("inf")
        add("P1 arrival |diff|", darr, darr <= 0.05, "<= 0.05")
        add("minDensityMin", m["minDensityMin"], m["minDensityMin"] >= 0.97, ">= 0.97")
        add("maxDensityMax", m["maxDensityMax"], m["maxDensityMax"] <= 1.03, "<= 1.03")
        bsteps = base[case]["steps"]["value"]
        drift = abs(m["steps"] - bsteps) / bsteps
        add("steps drift", drift, drift <= 0.05, "<= 0.05 (baseline %d)" % int(bsteps))
    elif case == "tank":
        for k in ("rmseBulk", "rmseNear"):
            limit = 1.25 * base[case][k]["value"]
            add(k, m[k], m[k] <= limit, "<= %.10g (1.25 x baseline)" % limit)
        limit = 5.0 * base[case]["keLast"]["value"]
        add("keLast", m["keLast"], m["keLast"] <= limit, "<= %.10g (5 x baseline)" % limit)
        add("rhoMin", m["rhoMin"], m["rhoMin"] >= 0.99, ">= 0.99")
        add("rhoMax", m["rhoMax"], m["rhoMax"] <= 1.01, "<= 1.01")
    elif case == "sloshing":
        rel, _ = _ke_relmax(series, "slosh_B_nx200_series.npz")
        add("KE rel max vs B", rel, rel <= 0.05, "<= 0.05")
        add("minDensityMin", m["minDensityMin"], m["minDensityMin"] >= 0.97, ">= 0.97")
        add("maxDensityMax", m["maxDensityMax"], m["maxDensityMax"] <= 1.03, "<= 1.03")
    else:
        raise ValueError("unknown case: " + case)
    return out, ok


def _tol(spread, value):
    return max(20.0 * spread, 1e-9 * abs(value), 1e-12)


def cmd_record(cases, baseline_path, cfgkw=None):
    cfgkw = cfgkw or {}
    data, meta_wall = {}, {}
    for case in cases:                       # same process order: tank, tank, dambreak, dambreak, ...
        m1, s1, w1 = _run_case(case, **cfgkw)
        m2, s2, w2 = _run_case(case, **cfgkw)
        entry = {}
        for k, v in m1.items():
            spread = abs(m1[k] - m2[k])
            entry[k] = {"value": v, "spread": spread, "tol": _tol(spread, v)}
        data[case] = entry
        meta_wall[case] = {"run1_s": w1, "run2_s": w2}
        for ln in _physics_lines(case, s1):
            print(ln, flush=True)
    data["meta"] = {"gitHead": _git_head(), "date": time.strftime("%Y-%m-%d %H:%M:%S"), "wallSeconds": meta_wall,
                    "torch": torch.__version__, "warp": getattr(__import__("warp"), "__version__", "unknown"), "cfg": cfgkw}
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


def cmd_check(cases, baseline_path, perturb=False, physics=False, cfgkw=None):
    cfgkw = cfgkw or {}
    with open(baseline_path) as f:
        base = json.load(f)
    print("cfg: %s" % (cfgkw if cfgkw else "{} (defaults)"), flush=True)
    all_pass = True
    gate_pass = True
    for case in cases:
        m, series, wall = _run_case(case, perturb=perturb, **cfgkw)
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
        if physics:
            gln, gok = _gate_lines(case, m, series, base)
            for ln in gln:
                print(ln, flush=True)
            gate_pass = gate_pass and gok
    if physics:
        print("\nPHYSICS GATE: %s" % ("PASS" if gate_pass else "FAIL"))
        return 0 if gate_pass else 1
    print("\nOVERALL: %s" % ("PASS" if all_pass else "FAIL"))
    return 0 if all_pass else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "check"])
    ap.add_argument("--cases", default="tank,dambreak")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--perturb", action="store_true", help="negative control: run the dam break with shifting=False")
    ap.add_argument("--cfg", default="", help="config overrides key=value[,key=value...] (true/false -> bool, else float if parsable, else string), passed as **cfgkw to the runners")
    ap.add_argument("--physics", action="store_true", help="check: the exit code depends only on the physics gate (the bit-level lines are still printed)")
    a = ap.parse_args(argv)
    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    cfgkw = _parse_cfg(a.cfg)
    if a.mode == "record":
        return cmd_record(cases, a.baseline, cfgkw=cfgkw)
    return cmd_check(cases, a.baseline, perturb=a.perturb, physics=a.physics, cfgkw=cfgkw)


if __name__ == "__main__":
    sys.exit(main())
