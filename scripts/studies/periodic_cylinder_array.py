"""Rungs 2 and 4 of the validation ladder of docs/plan-next-steps.md: a square periodic array of cylinders, driven by a uniform body force, Stokes regime.

Steady state: the total force of the fluid on the cylinder (wall loads: pressure + wall viscous + no-penetration impulse) equals the body force on the fluid, F = rho f N dx^2 (the body force on the particles present; momentum balance, rung 2).  The drag
coefficient K = F / (mu U) with U the superficial velocity (the mean fluid velocity over the whole cell, the solid counting zero) against the Sangani-Acrivos / Hasimoto square-array expansion
K = 4 pi / (-1/2 ln c - 0.738 + c - 0.887 c^2 + 2.038 c^3), c = pi R^2 / L^2 (rung 4).  That K is the drag of a mean pressure gradient G, F = G L^2: the gradient acts on the whole cell (fluid AND solid).
The body force here acts on the fluid only, so the same flow (same U) carries the drag F = f (1 - c) L^2: the reference for this driver is (1 - c) K_SA (checked with the Fourier solve of
stokes_array_ref.py with the force on the fluid only: U unchanged, K ratio 0.8743 = 1 - c).  mu = rho nu_eff, nu_eff = measured from a shear-wave decay of the same discretisation (`--nu`, default the measured law
nu_eff = about 0.96 alpha c0 H / (8 xi) (`shear_nu`); `--nu` fixes the fluid across resolutions.

    python scripts/studies/periodic_cylinder_array.py --n 48 --R 0.2 --alpha 0.5 --f 0.03 --time 40
"""
import argparse
import math
import time

import numpy as np
import torch

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, Scene, SurfaceRep
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.loads import LoadHistory

F64 = torch.float64


# grid-aligned square of side a in the unit cell, body force on the fluid only: K = F / (mu U) from the Fourier-penalisation Stokes solve (stokes_array_ref.py --square), N = 192 / 384 / 576:
# 26.60 / 26.10 / 25.98, extrapolated (order ~1.5) 25.84; the value below is the mean of the finest and the extrapolation (uncertainty ~0.3 %)
SQUARE_REF = {round(1.0 / 3.0, 6): 25.91}


def sangani_acrivos(c):
    return 4 * math.pi / (-0.5 * math.log(c) - 0.738 + c - 0.887 * c ** 2 + 2.038 * c ** 3)


def shear_nu(n, alpha, c0, device, Hfac=4.0, T=0.3, visc="alpha", cal=1.0):
    """effective kinematic viscosity of the discretisation: decay of a sine shear wave in the periodic box (no walls)."""
    dx = 1.0 / n
    X, Y = np.meshgrid(dx * (np.arange(n) + .5), dx * (np.arange(n) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    vel = np.zeros_like(pos)
    vel[:, 0] = 0.1 * np.sin(2 * math.pi * pos[:, 1])
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=c0, alpha=alpha, periodic=Periodic((0, 0), (1, 1)), graphStep=False, shifting=False, fluidViscosity=visc, morrisCalibration=cal)
    sim = DeltaSPH2D(pos, vel, np.ones(len(pos)), dx, None, cfg, device, support=Hfac * dx)
    ts, A = [], []
    for k in range(int(T / sim.dt)):
        sim.step()
        if k % 20 == 0:
            ts.append(sim.time)
            A.append(float((sim.v[:, 0] * torch.sin(2 * math.pi * sim.x[:, 1])).mean() * 2))
    return -np.polyfit(ts, np.log(A), 1)[0] / (2 * math.pi) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--R", type=float, default=0.2)
    ap.add_argument("--shape", default="disk", choices=("disk", "square"), help="square: a grid-aligned square of side --a (walls half a spacing from the lattice rows when a n is an integer; no curvature, no cut lattice)")
    ap.add_argument("--a", type=float, default=1.0 / 3.0, help="side of the square")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--nu", type=float, default=None, help="target kinematic viscosity: alpha = nu 8 xi / (c0 H) (the same fluid at every resolution)")
    ap.add_argument("--c0", type=float, default=10.0)
    ap.add_argument("--f", type=float, default=0.03)
    ap.add_argument("--time", type=float, default=40.0)
    ap.add_argument("--H", type=float, default=4.0, help="support in units of dx")
    ap.add_argument("--wall", default="noslip", help="wallViscosityForm: noslip (flux form), noslipMirror (exact Laplacian, antisymmetric mirror), noslipCurv (flux form with the second-order wall gradient)")
    ap.add_argument("--pack", type=int, default=0, help="body-fitted packing: this many iterations of DeltaSPH2D.pack (relax the particles to zero wall-consistency residual) before the run")
    ap.add_argument("--wpv", action="store_true", help="wallPressureViscous: the viscous term in the wall pressure condition")
    ap.add_argument("--rho", type=float, default=1.0, help="initial density (the mean pressure level P = c0^2 (rho - rho0): the Antuono switch and the wall pressure clamp act on its sign)")
    ap.add_argument("--consistent", action="store_true", help="pressureConsistent: the pressure force exact for a uniform pressure near the wall")
    ap.add_argument("--Pb", type=float, default=0.0, help="background pressure P_b added to the equation of state (density untouched)")
    ap.add_argument("--noWallForce", action="store_true", help="leave the body force out of the wall pressure condition (the old behaviour)")
    ap.add_argument("--visc", default="alpha", choices=("alpha", "morris"), help="fluid viscous operator (cfg.fluidViscosity)")
    ap.add_argument("--cal", type=float, default=1.0, help="cfg.morrisCalibration (Wendland C2: 0.985)")
    ap.add_argument("--complement", action="store_true", help="force cfg.complementMoments on (automatic with --visc morris)")
    ap.add_argument("--tables", action="store_true", help="force the table closure (cfg.complementMoments = False)")
    ap.add_argument("--set", default="", help="extra DeltaSPHConfig fields, e.g. \"pressureSolver='projection', fluidWarp=False, graphStep=False, ddt=False\"")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dx = 1.0 / a.n
    if a.nu is not None:
        a.alpha = a.nu * 8 * 2.821384729 / (a.c0 * a.H * dx)
    X, Y = np.meshgrid(dx * (np.arange(a.n) + .5), dx * (np.arange(a.n) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    if a.shape == "disk":
        pos = pos[np.linalg.norm(pos - 0.5, axis=1) >= a.R + 0.5 * dx]
        c = math.pi * a.R ** 2
        Kref = (1.0 - c) * sangani_acrivos(c)                             # the body force acts on the fluid only (F = f (1 - c)), Sangani-Acrivos is per cell (F = G L^2)
        rep = DiskArrayRep([(0.0, 0.0)], [a.R])
    else:
        pos = pos[np.abs(pos - 0.5).max(1) > 0.5 * a.a]
        c = a.a ** 2
        Kref = SQUARE_REF.get(round(a.a, 6), float("nan"))
        h = 0.5 * a.a
        rep = SurfaceRep.polygon([(-h, -h), (h, -h), (h, h), (-h, h)])
    nu = shear_nu(a.n, a.alpha, a.c0, a.device, a.H, visc=a.visc, cal=a.cal)
    print(f"{a.shape} N={len(pos)} c={c:.4f} nu_eff={nu:.5f} (alpha c0 H / 8 xi = {a.alpha * a.c0 * a.H * dx / (8 * 2.821384729):.5f})  K_ref={Kref:.3f}")
    scene = Scene([Body(bodyId=0, center=(0.5, 0.5), reps=[rep])], a.device)
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=a.c0, alpha=a.alpha, periodic=Periodic((0, 0), (1, 1)), bodyForce=(a.f, 0.0), bodyForceAtWall=not a.noWallForce, wallPressureViscous=a.wpv, pressureConsistent=a.consistent, backgroundPressure=a.Pb, graphStep=True, shifting=True, wallViscosityForm=a.wall, fluidViscosity=a.visc, morrisCalibration=a.cal, complementMoments=(False if a.tables else (True if a.complement else None)))
    for k, val in eval("dict(" + a.set + ")").items():                                         # extra DeltaSPHConfig fields (e.g. pressureSolver='projection', fluidWarp=False, graphStep=False, ddt=False)
        setattr(cfg, k, val)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, scene, cfg, a.device, support=a.H * dx)
    sim.rho = sim.rho * a.rho
    if a.pack:
        sim.pack(np.ones(len(pos), dtype=bool), iters=a.pack, verbose=True)
    hist = LoadHistory()
    Fbal = sim.cfg.rho0 * a.f * len(pos) * dx * dx                         # the body force on the fluid particles actually present (the cut lattice fills the fluid region to ~1 %)
    t0, k = time.time(), 0
    while sim.time < a.time:
        sim.step()
        hist.record(sim)
        k += 1
        if k % 1000 == 0:
            F = hist.total()[-200:, 0].mean()
            U = float(sim.v[:, 0].sum()) * dx * dx
            print(f"t={sim.time:7.2f} U={U:.5f} F={F:.5f} (balance {Fbal:.5f}, {F / Fbal:.4f}) K={F / (nu * U):.3f} Fy={hist.total()[-200:, 1].mean():+.1e}  {time.time() - t0:.0f}s", flush=True)
    F = hist.total()[-500:, 0].mean()
    U = float(sim.v[:, 0].sum()) * dx * dx
    print(f"final: F/balance={F / Fbal:.4f}  K={F / (nu * U):.3f}  K_ref={Kref:.3f}  ratio={F / (nu * U) / Kref:.4f}")


if __name__ == "__main__":
    main()
