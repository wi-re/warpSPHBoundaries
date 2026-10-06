"""Rungs 2 and 4 of the validation ladder of docs/plan-next-steps.md: a square periodic array of cylinders, driven by a uniform body force, Stokes regime.

Steady state: the total force of the fluid on the cylinder (wall loads: pressure + wall viscous + no-penetration impulse) equals the body force on the fluid, F = rho f N dx^2 (the body force on the particles present; momentum balance, rung 2).  The drag
coefficient K = F / (mu U) with U the superficial velocity (the mean fluid velocity over the whole cell, the solid counting zero) against the Sangani-Acrivos / Hasimoto square-array expansion
K = 4 pi / (-1/2 ln c - 0.738 + c - 0.887 c^2 + 2.038 c^3), c = pi R^2 / L^2 (rung 4).  mu = rho nu_eff, nu_eff = measured from a shear-wave decay of the same discretisation (`--nu`, default the measured law
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
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, Scene
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.loads import LoadHistory

F64 = torch.float64


def sangani_acrivos(c):
    return 4 * math.pi / (-0.5 * math.log(c) - 0.738 + c - 0.887 * c ** 2 + 2.038 * c ** 3)


def shear_nu(n, alpha, c0, device, Hfac=4.0, T=0.3):
    """effective kinematic viscosity of the discretisation: decay of a sine shear wave in the periodic box (no walls)."""
    dx = 1.0 / n
    X, Y = np.meshgrid(dx * (np.arange(n) + .5), dx * (np.arange(n) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    vel = np.zeros_like(pos)
    vel[:, 0] = 0.1 * np.sin(2 * math.pi * pos[:, 1])
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=c0, alpha=alpha, periodic=Periodic((0, 0), (1, 1)), graphStep=False, shifting=False)
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
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--nu", type=float, default=None, help="target kinematic viscosity: alpha = nu 8 xi / (c0 H) (the same fluid at every resolution)")
    ap.add_argument("--c0", type=float, default=10.0)
    ap.add_argument("--f", type=float, default=0.03)
    ap.add_argument("--time", type=float, default=40.0)
    ap.add_argument("--H", type=float, default=4.0, help="support in units of dx")
    ap.add_argument("--wall", default="noslip")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dx = 1.0 / a.n
    if a.nu is not None:
        a.alpha = a.nu * 8 * 2.821384729 / (a.c0 * a.H * dx)
    X, Y = np.meshgrid(dx * (np.arange(a.n) + .5), dx * (np.arange(a.n) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    pos = pos[np.linalg.norm(pos - 0.5, axis=1) >= a.R + 0.5 * dx]
    c = math.pi * a.R ** 2
    nu = shear_nu(a.n, a.alpha, a.c0, a.device, a.H)
    print(f"N={len(pos)} c={c:.4f} nu_eff={nu:.5f} (alpha c0 H / 8 xi = {a.alpha * a.c0 * a.H * dx / (8 * 2.821384729):.5f})  K_SA={sangani_acrivos(c):.3f}")
    scene = Scene([Body(bodyId=0, center=(0.5, 0.5), reps=[DiskArrayRep([(0.0, 0.0)], [a.R])])], a.device)
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=a.c0, alpha=a.alpha, periodic=Periodic((0, 0), (1, 1)), bodyForce=(a.f, 0.0), graphStep=True, shifting=True, wallViscosityForm=a.wall)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, scene, cfg, a.device, support=a.H * dx)
    hist = LoadHistory()
    Fbal = sim.cfg.rho0 * a.f * len(pos) * dx * dx                         # the body force on the fluid particles actually present (the cut lattice fills the fluid region to ~1 %)
    t0, k = time.time(), 0
    while sim.time < a.time:
        sim.step()
        hist.record(sim)
        k += 1
        if k % 1000 == 0:
            F = hist.total()[-200:, 0].mean()
            U = float(sim.v[:, 0].mean()) * (1.0 - c)
            print(f"t={sim.time:7.2f} U={U:.5f} F={F:.5f} (balance {Fbal:.5f}, {F / Fbal:.4f}) K={F / (nu * U):.3f} Fy={hist.total()[-200:, 1].mean():+.1e}  {time.time() - t0:.0f}s", flush=True)
    F = hist.total()[-500:, 0].mean()
    U = float(sim.v[:, 0].mean()) * (1.0 - c)
    print(f"final: F/balance={F / Fbal:.4f}  K={F / (nu * U):.3f}  K_SA={sangani_acrivos(c):.3f}  ratio={F / (nu * U) / sangani_acrivos(c):.3f}")


if __name__ == "__main__":
    main()
