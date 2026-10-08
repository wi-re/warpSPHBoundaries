"""Stability ladder of the omniSPH-style DFSPH in periodic domains (docs/plan-next-steps.md "DFSPH track", D3): the disk array with a body force blew up at t ~ 7 with rho_min ~ 0.7 next to the disk, so step back
(as warpSPH's VD+PS track did: periodic, stressed flows need a different set-up than free-surface ones) and add one ingredient per rung:

    tgv       Taylor-Green vortex in the periodic unit box, no walls, no forcing: u = U (-cos kx sin ky, sin kx cos ky), k = 2 pi; kinetic energy decays as exp(-4 nu k^2 t) (Morris, calibrated nu).
              Checks: decay rate / analytic, monotone KE, the density band.
    obstacle  a static disk in the periodic cell, no forcing, an initial velocity field (`--init uniform`: U e_x, `tgv`): the flow must decay to rest with a bounded density band.

Both take `--clamp` / `--noClamp` (DFSPHConfig.densityClamp: p >= 0 in the density solve, omniSPH's free-surface default, against the signed pressure).

    python scripts/studies/dfsph_periodic.py tgv [--n 48] [--nu 0.0185] [--U 0.1] [--time 5] [--noClamp]
    python scripts/studies/dfsph_periodic.py obstacle [--init uniform] [--R 0.2] [--time 10] [--noClamp]
"""
import argparse
import math
import time

import numpy as np
import torch

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, Scene
from warpSPHBoundaries.sim.dfsph2d import DFSPH2D, DFSPHConfig, PACKING, carve, lattice_calibration


def lattice(n, dx):
    X, Y = np.meshgrid(dx * (np.arange(n) + 0.5), dx * (np.arange(n) + 0.5), indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


def tgv_field(pos, U):
    k = 2 * math.pi
    return U * np.stack([-np.cos(k * pos[:, 0]) * np.sin(k * pos[:, 1]), np.sin(k * pos[:, 0]) * np.cos(k * pos[:, 1])], 1)


def run(sim, a, label, dx, decay=None):
    t0 = time.time()
    ke0 = float((sim.V * (sim.v ** 2).sum(1)).sum()) * 0.5
    ts, ke, rmin, rmax = [0.0], [ke0], [float(sim.rho.min()) if sim.rho is not None else 1.0], [1.0]
    k, rises = 0, 0
    while sim.time < a.time:
        sim.step()
        k += 1
        e = float((sim.V * (sim.v ** 2).sum(1)).sum()) * 0.5
        rises += e > ke[-1] * (1 + 1e-9)
        ts.append(sim.time); ke.append(e); rmin.append(float(sim.rho.min())); rmax.append(float(sim.rho.max()))
        vmax = float(sim.v.norm(dim=1).max())
        if k % a.every == 0 or not math.isfinite(e) or vmax > 10 * a.U:
            print(f"  t {sim.time:.3f} dt {sim.dt:.2e} iters {sim.iters} err {sim.err:.1e} vmax {vmax:.3e} KE/KE0 {e / ke0:.4e} rho [{rmin[-1]:.4f}, {rmax[-1]:.4f}] "
                  f"p [{float(sim.p.min()):+.2e}, {float(sim.p.max()):+.2e}]  {time.time() - t0:.0f} s", flush=True)
        if not math.isfinite(e) or vmax > 10 * a.U:
            print(f"{label}: BLEW UP at t {sim.time:.3f} (step {k})")
            return
    ts, ke = np.array(ts), np.array(ke)
    msg = f"{label}: t {sim.time:.2f}, {k} steps, KE/KE0 {ke[-1] / ke0:.3e}, KE rises {rises}, rho band [{min(rmin[1:]):.4f}, {max(rmax[1:]):.4f}] (final [{rmin[-1]:.4f}, {rmax[-1]:.4f}])"
    if decay is not None:
        sel = (ke > 1e-6 * ke0) & (ts > 0.05 * ts[-1])
        rate = -np.polyfit(ts[sel], np.log(ke[sel]), 1)[0]
        msg += f", KE decay rate / analytic {rate / decay:.4f}"
    print(msg + f"  ({time.time() - t0:.0f} s)")


def cfg(a, cal, **kw):
    return DFSPHConfig(gravity=(0.0, 0.0), viscosity=a.nu, boundaryFriction=0.0, wallMass=cal["mu"], maxDt=a.maxDt, cfl=0.4, recordForces=False, densityClamp=a.clamp,
                       periodic=Periodic((0, 0), (1, 1)), **{**eval("dict(" + a.set + ")"), **kw})


def run_tgv(a):
    dx = 1.0 / a.n
    h = dx / PACKING
    cal = lattice_calibration(dx, dx, h)
    pos = lattice(a.n, dx)
    if a.jitter > 0:
        pos = pos + a.jitter * dx * np.random.default_rng(0).uniform(-1, 1, pos.shape)
    sim = DFSPH2D(pos, tgv_field(pos, a.U), cal["V"], h, None, cfg(a, cal), a.device)
    run(sim, a, f"tgv n={a.n} nu={a.nu} U={a.U} jitter={a.jitter} clamp={a.clamp}", dx, decay=4 * a.nu * 2 * (2 * math.pi) ** 2 / 2)


def run_obstacle(a):
    dx = 1.0 / a.n
    h = dx / PACKING
    cal = lattice_calibration(dx, dx, h)
    disk = lambda: Body(bodyId=0, center=(0.5, 0.5), reps=[DiskArrayRep([(0.0, 0.0)], [a.R])])
    pos = carve(lattice(a.n, dx), [disk()], h, cal["lamCell" if a.carve == "cell" else "lamRow0"], a.device)
    print(f"carve {a.carve}: N {len(pos)}, N dx^2 / fluid area {len(pos) * dx * dx / (1 - math.pi * a.R ** 2):.4f}")
    vel = tgv_field(pos, a.U) if a.init == "tgv" else np.tile([a.U, 0.0], (len(pos), 1))
    sim = DFSPH2D(pos, vel, cal["V"], h, Scene([disk()], a.device), cfg(a, cal), a.device)
    run(sim, a, f"obstacle R={a.R} n={a.n} nu={a.nu} init={a.init} U={a.U} clamp={a.clamp} carve={a.carve} N={len(pos)}", dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=("tgv", "obstacle"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--nu", type=float, default=0.0185)
    ap.add_argument("--U", type=float, default=0.1)
    ap.add_argument("--R", type=float, default=0.2)
    ap.add_argument("--init", choices=("uniform", "tgv"), default="uniform")
    ap.add_argument("--carve", choices=("row0", "cell"), default="row0")
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--time", type=float, default=5.0)
    ap.add_argument("--maxDt", type=float, default=0.01)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--clamp", dest="clamp", action="store_true", default=True)
    ap.add_argument("--noClamp", dest="clamp", action="store_false")
    ap.add_argument("--set", default="", help="extra DFSPHConfig fields")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"tgv": run_tgv, "obstacle": run_obstacle}[a.case](a)


if __name__ == "__main__":
    main()
