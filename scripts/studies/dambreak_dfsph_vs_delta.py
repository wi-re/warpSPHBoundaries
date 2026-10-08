"""Dam break (column 0.2 x 0.8 at the left wall of a 1.6 x 1.0 box, the DFSPH validation geometry) with DFSPH2D (omniSPH-style default, calibrated lattice) and DeltaSPH2D (EOS) side by side: front position, mean height, max speed, kinetic energy and density range at fixed times (docs/dfsph-validation.md s.9).  The converged projection (pressureSolver='projection') is for closed / periodic flows: it has no free-surface treatment and
blows up (NaN within 12 steps) on this case.

    python scripts/studies/dambreak_dfsph_vs_delta.py dfsph|delta [--dx 0.01] [--time 1.0] [--set "..."]
"""
import argparse
import math
import time

import numpy as np

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.dfsph2d import DFSPH2D, DFSPHConfig, PACKING, domain_scene, lattice_calibration

W, Hf, L, H = 0.2, 0.8, 1.6, 1.0
G = 9.81


def block(x0, y0, nx, ny, dx):
    X, Y = np.meshgrid(x0 + dx * np.arange(nx), y0 + dx * np.arange(ny), indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


def make(a):
    dx = a.dx
    nx, ny = int(round(W / dx)), int(round(Hf / dx))
    extra = eval("dict(" + a.set + ")")
    if a.scheme == "dfsph":
        h = dx / PACKING
        cal = lattice_calibration(dx, dx, h)
        x = block(cal["dwallX"], cal["dwallY"], nx, ny, dx)
        cfg = DFSPHConfig(gravity=(0.0, -G), wallMass=cal["mu"], viscosity=0.0, boundaryFriction=5e-3, maxDt=1e-3, **extra)
        sim = DFSPH2D(x, np.zeros_like(x), cal["V"], h, domain_scene("surface", (0.0, 0.0), (L, H), h, a.device), cfg, a.device)
        return sim, lambda s: s.dt
    c0 = 10.0 * math.sqrt(G * Hf)
    x = block(0.5 * dx, 0.5 * dx, nx, ny, dx)
    cfg = DeltaSPHConfig(gravity=(0.0, -G), c0=c0, **extra)
    sim = DeltaSPH2D(x, np.zeros_like(x), np.ones(len(x)), dx, domain_scene("surface", (0.0, 0.0), (L, H), 4 * dx, a.device), cfg, a.device, support=4 * dx)
    return sim, lambda s: s.dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scheme", choices=("dfsph", "delta"))
    ap.add_argument("--dx", type=float, default=0.01)
    ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--set", default="")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    sim, _ = make(a)
    snaps = [round(0.1 * k, 1) for k in range(1, int(a.time * 10 + 1e-9) + 1)]
    t0, k, done = time.time(), 0, set()
    print(f"{a.scheme} dx={a.dx} N={len(sim.x)}", flush=True)
    while sim.time < a.time - 1e-12:
        sim.step()
        k += 1
        for s in snaps:
            if s not in done and sim.time >= s - 1e-12:
                done.add(s)
                x, v = sim.x, sim.v
                speed = float(v.norm(dim=1).max())
                if not np.isfinite(speed):
                    print(f"  t {s:.1f}: NaN", flush=True)
                    return
                print(f"  t {s:.1f}: front {float(x[:, 0].max()):.4f} meanY {float(x[:, 1].mean()):.4f} vmax {speed:.2f} KE/m {0.5 * float((v ** 2).sum(1).mean()):.4f} "
                      f"rho [{float(sim.rho.min()):.3f}, {float(sim.rho.max()):.3f}]  {time.time() - t0:.0f} s", flush=True)
    print(f"done {a.scheme}: {k} steps, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
