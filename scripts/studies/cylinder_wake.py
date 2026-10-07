"""Rung 5 of the validation ladder of docs/plan-next-steps.md: flow past one body (a disk or a grid-aligned square) driven by a prescribed-velocity frame in a periodic box (the warpSPH `movingObstacle` setup).

Box [0, Lx] x [0, Ly] (in D), periodic in both directions; the frame is the band |x| < b, |y| < b around the seams (sim/pinned.py: the velocity is the free stream U there, the particles stay fluid); the body sits at (xb, Ly / 2), xb
measured from the seam.  The run starts impulsively (uniform stream outside the body).  Output: C_D (mean over the second half), C_L (rms, amplitude), the Strouhal number of the lift, the recirculation length behind the body from the
centreline velocity at the end (steady cases), and the series in .tmp.  Re = U D / nu with nu the effective viscosity of the alpha form, nu_eff = NU_FAC alpha c0 H / (8 xi) (NU_FAC from the shear-wave decay along the lattice axes,
docs/plan-next-steps.md; along the diagonal it is ~8 % larger at H = 4 dx).

Literature for the UNCONFINED circular cylinder (the frame + periodic images give a blockage D / (Ly - 2 b), which raises C_D and St by a few %):
  Re 20: C_D 2.0-2.09, L_w / D 0.91-0.94 (Dennis & Chang 1970, Fornberg 1980);  Re 40: C_D 1.50-1.55, L_w / D 2.2-2.35;  Re 100: C_D 1.33-1.35, C_L amplitude 0.32-0.34, St 0.164-0.166 (Williamson 1996).
Square (unconfined; values to be verified against the papers before use as a gate): Re 100 St ~0.145-0.15, C_D ~1.45-1.5 (Sohankar et al. 1998, Sen et al. 2011).

    warpSPHCore_PRECISION=float32 python scripts/studies/cylinder_wake.py --Re 100 --res 20 --time 150
"""
import argparse
import math
import os
import time

import numpy as np
import torch

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries import paths
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, Scene, SurfaceRep
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.loads import LoadHistory, coefficients, strouhal
from warpSPHBoundaries.sim.pinned import Pinned

XI = 2.821384729
NU_FAC = 0.957

REF = {("disk", 20): dict(CD=(2.0, 2.09), Lw=(0.91, 0.94)), ("disk", 40): dict(CD=(1.50, 1.55), Lw=(2.2, 2.35)), ("disk", 100): dict(CD=(1.33, 1.35), CLamp=(0.32, 0.34), St=(0.164, 0.166))}


def recirculation_length(x, v, xc, yc, half, dx):
    """length behind the body (in the units of x) over which the centreline velocity is negative: particles within one spacing of the centreline, binned by dx, from the rear of the body to the first bin with u > 0."""
    m = (np.abs(x[:, 1] - yc) < dx) & (x[:, 0] > xc + half)
    if not m.any():
        return 0.0
    s = x[m, 0] - (xc + half)
    u = v[m, 0]
    k = (s / dx).astype(int)
    nb = int(k.max()) + 1
    cnt = np.bincount(k, minlength=nb)
    ub = np.bincount(k, weights=u, minlength=nb) / np.maximum(cnt, 1)
    neg = (ub < 0) & (cnt > 0)
    if not neg[:3].any():
        return 0.0
    first_pos = int(np.argmax(~neg & (cnt > 0)))
    if first_pos == 0 or not (~neg[first_pos:]).any():
        return 0.0
    u0, u1 = ub[first_pos - 1], ub[first_pos]                                       # linear zero crossing between the bin centres
    return ((first_pos - 0.5) + (-u0) / (u1 - u0)) * dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="disk", choices=("disk", "square"))
    ap.add_argument("--Re", type=float, default=100.0)
    ap.add_argument("--res", type=int, default=20, help="D / dx")
    ap.add_argument("--Lx", type=float, default=30.0)
    ap.add_argument("--Ly", type=float, default=15.0)
    ap.add_argument("--band", type=float, default=1.5, help="half width of the frame slabs (in D)")
    ap.add_argument("--xb", type=float, default=10.0, help="body centre from the seam x = 0 (in D)")
    ap.add_argument("--c0", type=float, default=10.0, help="speed of sound in units of U")
    ap.add_argument("--H", type=float, default=4.0, help="support in units of dx")
    ap.add_argument("--time", type=float, default=150.0, help="in D / U")
    ap.add_argument("--wall", default="noslipMoment")
    ap.add_argument("--consistent", action="store_true")
    ap.add_argument("--noShift", action="store_true")
    ap.add_argument("--visc", default="alpha", choices=("alpha", "morris"), help="fluid viscous operator; alpha: nu from the lattice-AXIS factor NU_FAC (the alpha form is anisotropic, the flow sees ~4 %% more at H = 4 dx); morris: calibrated, isotropic")
    ap.add_argument("--cal", type=float, default=0.985, help="cfg.morrisCalibration for --visc morris (Wendland C2)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    D, U = 1.0, 1.0
    dx = D / a.res
    nx, ny = int(round(a.Lx / dx)), int(round(a.Ly / dx))
    Lx, Ly = nx * dx, ny * dx
    X, Y = np.meshgrid(dx * (np.arange(nx) + .5), dx * (np.arange(ny) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    xc, yc = a.xb, 0.5 * Ly
    if a.shape == "disk":
        pos = pos[np.linalg.norm(pos - [xc, yc], axis=1) >= 0.5 * D + 0.5 * dx]
        rep = DiskArrayRep([(0.0, 0.0)], [0.5 * D])
    else:
        yc = round(yc / dx) * dx                                                   # walls on the half-spacing lines of the lattice (res even)
        xc = round(xc / dx) * dx
        pos = pos[np.abs(pos - [xc, yc]).max(1) > 0.5 * D]
        h = 0.5 * D
        rep = SurfaceRep.polygon([(-h, -h), (h, -h), (h, h), (-h, h)])
    nu = U * D / a.Re
    H = a.H * dx
    alpha = nu / (NU_FAC if a.visc == "alpha" else 1.0) * 8 * XI / (a.c0 * U * H)          # Morris: nu = alpha c0 H / (8 xi) exactly (the operator is divided by the calibration)
    box = Periodic((0.0, 0.0), (Lx, Ly))
    band = Pinned(slabs=((0, 0.0, a.band * D), (1, 0.0, a.band * D)), velocity=(U, 0.0))
    scene = Scene([Body(bodyId=0, center=(xc, yc), reps=[rep])], a.device)
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=a.c0 * U, alpha=alpha, periodic=box, pinned=band, pressureConsistent=a.consistent, graphStep=True, shifting=not a.noShift, wallViscosityForm=a.wall, fluidViscosity=a.visc, morrisCalibration=a.cal)
    vel = np.tile([U, 0.0], (len(pos), 1))
    sim = DeltaSPH2D(pos, vel, np.ones(len(pos)), dx, scene, cfg, a.device, support=H)
    blockage = D / (Ly - 2 * a.band * D)
    print(f"{a.shape} Re={a.Re:g} D/dx={a.res} N={len(pos)} box {Lx:g} x {Ly:g} band +-{a.band:g} blockage {blockage:.3f}  nu={nu:.4g} alpha={alpha:.4f} dt={sim.dt:.3g}  {a.wall}{' consistent' if a.consistent else ''} {a.visc}", flush=True)
    hist = LoadHistory()
    t0, k = time.time(), 0
    while sim.time < a.time:
        sim.step()
        hist.record(sim)
        k += 1
        if k % 2000 == 0:
            t, Fl = hist.series()
            cd, cl = coefficients(Fl[-2000:], 1.0, U, D)
            print(f"t={sim.time:7.2f} C_D={cd.mean():.4f} C_L={cl.mean():+.4f} (range {cl.min():+.3f} {cl.max():+.3f}) vmax={float(sim.v.norm(dim=1).max()):.3f}  {time.time() - t0:.0f}s", flush=True)
    t, Fl = hist.series()
    cd, cl = coefficients(Fl, 1.0, U, D)
    m = t >= 0.5 * a.time
    x, v = sim.x.cpu().numpy(), sim.v.cpu().numpy()
    xw = np.stack([np.mod(x[:, 0], Lx), np.mod(x[:, 1], Ly)], 1)
    Lw = recirculation_length(xw, v, xc, yc, 0.5 * D, dx) / D
    clr = cl[m]
    St = strouhal(t, cl, D, U, t0=0.5 * a.time) if clr.std() > 1e-2 else float("nan")
    print(f"final ({a.shape}, Re {a.Re:g}, D/dx {a.res}): C_D={cd[m].mean():.4f}  C_L rms={clr.std():.4f} amp={0.5 * (np.percentile(clr, 99) - np.percentile(clr, 1)):.4f}  St={St:.4f}  L_w/D={Lw:.3f}  ({time.time() - t0:.0f}s, {k} steps)")
    ref = REF.get((a.shape, int(a.Re)))
    if ref:
        print("  unconfined literature:", "  ".join(f"{n} {lo:g}-{hi:g}" for n, (lo, hi) in ref.items()))
    d = paths.tmp_dir()
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"wake_{a.shape}_Re{a.Re:g}_r{a.res}_{a.visc}{a.tag}.npz")
    np.savez(out, t=t, CD=cd, CL=cl, x=x, v=v, rho=sim.rho.cpu().numpy())
    print("  series:", out)


if __name__ == "__main__":
    main()
