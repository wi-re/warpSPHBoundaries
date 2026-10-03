"""Free-surface detector comparison on real states: at each listed step count of a default-config run, evaluate the
free-surface detector's wall routes of `_detect_surface` on the SAME state -- all switches False vs the chosen
`--exact` switches True (`cover` -> `coverExact`, `cone` -> `coneExact`, `both` -> both) -- and report the
disagreement.  No pass/fail except: disagreements > 2 % of N_near at any snapshot is a finding.  The `max|Cw_q-Cw_e|`
column is printed only for `cover`/`both` (the cover-vector disagreement; `--exact cover` is the default and its
output is byte-for-byte the original T2.3 comparison).

    python -m edgebound.deltasph_detcmp [--case dambreak|tank] [--steps 0,500,1500,3000,5000] [--nx 67] [--exact cover|cone|both]

The dam break uses `marrone_dambreak(nx=...)` with the harness cfg (`shifting=True, noPen="impulse"`); the tank
uses `hydrostatic_tank(dp=0.02, domain="surface")` with the harness cfg.
"""
import argparse
import sys

import torch

from .cover import cover_vector_scene
from .deltasph2d import hydrostatic_tank, marrone_dambreak
from .dfsph2d import F64, neighbor_pairs


def _snapshot(sim, stepno, findings, exact="cover"):
    x = sim.x
    i, j, r = neighbor_pairs(x, sim.Hvec)
    lam, G, A = sim._wall_data(x, sim.rho)
    near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
    samples = None
    if len(near):
        samples = (near, sim._solid_samples(x, near))
    cfg = sim.cfg
    show_ratio = exact in ("cover", "both")
    # the two evaluations on the same state: all switches False vs the chosen switches True (cover -> coverExact, cone -> coneExact, both -> both); restored afterwards
    cfg.coverExact = False
    cfg.coneExact = False
    surf_q = sim._detect_surface(x, i, j, r, lam, samples)
    if exact in ("cover", "both"):
        cfg.coverExact = True
    if exact in ("cone", "both"):
        cfg.coneExact = True
    surf_e = sim._detect_surface(x, i, j, r, lam, samples)
    cfg.coverExact = False
    cfg.coneExact = False
    N, Nn = int(len(x)), int(len(near))
    nq, ne = int(surf_q.sum()), int(surf_e.sum())
    dis = (surf_q != surf_e)
    ndis = int(dis.sum())
    nw = cfg.wallMass / sim.dx ** 2
    ratio = float("nan")
    if show_ratio and Nn:
        _, (ins, u, rk, dr, dphi) = samples
        area = (rk * dr * dphi)[:, None]
        wt = ins.any(0).to(F64) * area[None]
        Cw_q = -nw * (wt[..., None] * u[None, None]).sum((1, 2))
        Cw_e = nw * cover_vector_scene(sim.scene, x[near], sim.H)
        ratio = float((Cw_q - Cw_e).abs().max()) / (nw * sim.H ** 2)
    if show_ratio:
        print("step %6d  N=%5d N_near=%5d  surface flags quad=%4d exact=%4d  disagree=%3d  max|Cw_q-Cw_e|/(n_w H^2)=% .3e"
              % (stepno, N, Nn, nq, ne, ndis, ratio), flush=True)
    else:
        print("step %6d  N=%5d N_near=%5d  surface flags quad=%4d exact=%4d  disagree=%3d"
              % (stepno, N, Nn, nq, ne, ndis), flush=True)
    if Nn and ndis > 0.02 * Nn:
        findings.append(stepno)
        print("  *** FINDING: disagreements > 2 %% of N_near at step %d" % stepno, flush=True)
    for k in torch.nonzero(dis).flatten()[:10].tolist():
        d, n, hit = sim.scene.signed_distance(x[[k]])
        print("    disagree #%d:  d/dx = % .3f   y = % .4f" % (k, float(d[0]) / sim.dx, float(x[k, 1])), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="dambreak", choices=["dambreak", "tank"])
    ap.add_argument("--steps", default="0,500,1500,3000,5000")
    ap.add_argument("--nx", type=int, default=67)
    ap.add_argument("--exact", default="cover", choices=["cover", "cone", "both"])
    a = ap.parse_args(argv)
    steps = sorted(int(s) for s in a.steps.split(",") if s.strip() != "")
    if a.case == "dambreak":
        sim, info = marrone_dambreak(nx=a.nx, shifting=True, noPen="impulse")
    else:
        sim, info = hydrostatic_tank(dp=0.02, domain="surface")
    findings = []
    k = 0
    for nxt in steps:
        while k < nxt:
            sim.step()
            k += 1
        _snapshot(sim, k, findings, exact=a.exact)
    print("\nfindings: %s" % (findings if findings else "none"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
