"""Free-surface comparison of the DFSPH projection variants (docs/plan-next-steps.md "DFSPH in closed periodic domains" -> free surfaces): the omniSPH-style default (divergence + density solve, p >= 0) against the
compact projection (sim/projection.py) with a Dirichlet free surface, the surface-aware shift and, optionally, a weak density term.

    column  a fluid block at rest in a box under gravity: the pressure against rho g (H - y) (bulk / near the walls, RMSE in units of rho g H), the kinetic energy, the drift of the mean height and of the top
    dam     the dam break of scripts/dfsph_validation.py (0.2 x 0.8 column, box to x = 1.6, omniSPH conventions with the calibrated lattice): front and mean height over time

    python scripts/studies/dfsph_freesurface.py column|dam [--variant omni|compact|compact+rho] [--dx 0.01] [--time 2]
"""
import argparse
import time

import numpy as np
import torch

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.sim.dfsph2d import DFSPH2D, DFSPHConfig, PACKING, domain_scene, lattice_calibration

VARIANTS = {
    "omni": dict(),
    "compact": dict(densitySolve=False, projection="compact", projectionTol=1e-8, shifting="fixed", shiftA=0.5),
    "compact-diff": dict(densitySolve=False, projection="compact", projectionTol=1e-8, shifting="fixed", shiftA=0.5, projectionGradient="difference"),
    "mirror": dict(densitySolve=False, projection="compact", projectionTol=1e-8, shifting="fixed", shiftA=0.5, projectionWall="mirror"),
    "mirror+rho": dict(densitySolve=False, projection="compact", projectionTol=1e-8, shifting="fixed", shiftA=0.5, projectionWall="mirror", projectionDensity=0.1),
    "compact+rho": dict(densitySolve=False, projection="compact", projectionTol=1e-8, shifting="fixed", shiftA=0.5, projectionDensity=0.1),
}


def block(x0, y0, nx, ny, dx, dy):
    X, Y = np.meshgrid(x0 + dx * np.arange(nx), y0 + dy * np.arange(ny), indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


def make(a, W, Hf, box, fit=False):
    dx = a.dx
    h = dx / PACKING
    cal = lattice_calibration(dx, dx, h)
    nx, ny = int(round(W / dx)), int(round(Hf / dx))
    lo = np.array([0.0, 0.0])
    x = block(cal["dwallX"], cal["dwallY"], nx, ny, dx, dx)
    hi = np.array([2 * cal["dwallX"] + (nx - 1) * dx if fit else box[0], box[1]])             # fit: the side walls at the rest distance of both outer columns
    cfg = DFSPHConfig(wallMass=cal["mu"], viscosity=a.nu, boundaryFriction=0.0 if a.nu > 0 else 5e-3, maxDt=a.maxDt, **{**VARIANTS[a.variant], **eval("dict(" + a.set + ")")})
    sim = DFSPH2D(x, np.zeros_like(x), cal["V"], h, domain_scene("surface", lo, hi, h, a.device), cfg, a.device)
    return sim, cal, x


def pressure(sim, variant):
    return sim.pDiv if variant.startswith("compact") else sim.p


def run_column(a):
    W, Hf = 0.4, 0.3
    sim, cal, x0 = make(a, W, Hf, (W, 0.6), fit=True)
    g = 9.81
    t0, k, ts, ke, my = time.time(), 0, [], [], []
    my0 = float(sim.x[:, 1].mean())
    while sim.time < a.time:
        sim.step()
        k += 1
        if k % 20 == 0:
            ts.append(sim.time)
            ke.append(float(0.5 * (sim.V * (sim.v ** 2).sum(1)).sum()))
            my.append(float(sim.x[:, 1].mean()))
        if k % 500 == 0:
            print(f"  t {sim.time:.3f} dt {sim.dt:.2e} iters {sim.iters} KE {ke[-1]:.3e} vmax {float(sim.v.norm(dim=1).max()):.3e} mean y drift / dx {(my[-1] - my0) / a.dx:+.4f}  {time.time() - t0:.0f} s", flush=True)
    xy = sim.x.cpu().numpy()
    p = pressure(sim, a.variant).cpu().numpy()
    top = xy[:, 1].max() + 0.5 * a.dx
    ref = g * (top - xy[:, 1])
    res = (p - ref) / (g * Hf)
    near = (xy[:, 1] < 2 * a.dx) | (xy[:, 0] < 2 * a.dx) | (xy[:, 0] > xy[:, 0].max() - 2 * a.dx)
    bulk = ~near & (xy[:, 1] < top - 2 * a.dx)
    rm = lambda m: float(np.sqrt(np.mean(res[m] ** 2)))
    half = len(ke) // 2
    print(f"column {a.variant} dx={a.dx} N={len(xy)} t={sim.time:.2f}: p RMSE / rho g H bulk {rm(bulk):.4f} near walls {rm(near):.4f} (max {np.abs(res[near]).max():.3f}); "
          f"KE tail {np.mean(ke[half:]):.3e} peak {max(ke):.3e}; mean-height drift {(my[-1] - my0) / a.dx:+.4f} dx; vmax {float(sim.v.norm(dim=1).max()):.2e}  ({time.time() - t0:.0f} s, {k} steps)")


def run_dam(a):
    W, Hf = 0.2, 0.8                                                                    # scripts/dfsph_validation.py: the block (0.1..0.3) x (0.1..0.9) in a box to x = 1.6, y = 1.0 (here shifted to the origin)
    sim, cal, x0 = make(a, W, Hf, (1.5, 0.9))
    t0, k, snaps, out = time.time(), 0, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), {}
    my0 = float(sim.x[:, 1].mean())
    while sim.time < a.time - 1e-12:
        sim.step()
        k += 1
        for s in snaps:
            if s not in out and sim.time >= s - 1e-12:
                out[s] = (float(sim.x[:, 0].max()), float(sim.x[:, 1].mean()), float(sim.v.norm(dim=1).max()), float(sim.rho.min()), float(sim.rho.max()))
                print(f"  t {s:.1f}: front {out[s][0]:.4f} mean y {out[s][1]:.4f} vmax {out[s][2]:.2f} rho [{out[s][3]:.3f}, {out[s][4]:.3f}] iters {sim.iters}  {time.time() - t0:.0f} s", flush=True)
    print(f"dam {a.variant} dx={a.dx}: " + " | ".join(f"t={s}: front {v[0]:.4f} meanY {v[1]:.4f}" for s, v in out.items()) + f"  ({time.time() - t0:.0f} s, {k} steps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=("column", "dam"))
    ap.add_argument("--variant", default="omni", choices=tuple(VARIANTS))
    ap.add_argument("--dx", type=float, default=0.01)
    ap.add_argument("--time", type=float, default=2.0)
    ap.add_argument("--nu", type=float, default=0.0)
    ap.add_argument("--maxDt", type=float, default=1e-3)
    ap.add_argument("--set", default="")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"column": run_column, "dam": run_dam}[a.case](a)


if __name__ == "__main__":
    main()
