"""Validation of DeltaSPH2D against warpSPH's `sun2017DeltaSPH`/`deltaSPH` + mDBC.   python scripts/deltasph_validation.py tank|wedge|dambreak|sloshing …  (runners: edgebound.sim.validation)

 sloshing : SPHERIC test case 10 (warpSPH `examples/sloshingTank`): rolling tank as rotating gravity, Sensor-1 pressure
 dambreak : Marrone et al. 2011 s.3.1 (warpSPH `probe_deltaSPHMarrone.py`): probes P1-P3, front, KE
 wedge : the same with the sharp wedge on the bed (`probe_englishWedge.py --wedge`): face / apex / base-corner bands of the probe
 tank : English et al. 2022 s.4.1 still water in a flat tank (warpSPH `scripts/probe_englishWedge.py --no-wedge`): per-particle p/(rho0 g H) against the hydrostatic line, kinetic energy history.
        Scored exactly as the probe's `_score`: bulk = 2 dx below the surface and off every wall, near wall = within 2 dx of the bed or a side wall, settled KE = mean over the last 25 % of the record.
"""
import sys

from edgebound.sim.validation import (report_dambreak, report_tank, report_wedge, run_dambreak, run_sloshing, run_tank, run_wedge)

def main():
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


if __name__ == "__main__":
    main()
