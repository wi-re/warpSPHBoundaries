"""Rung 3 of the validation ladder of docs/plan-next-steps.md: plane Poiseuille flow between two plates, periodic in x, driven by a body force.

Plates are `BoxRep` bodies that SPAN the periodic box (a wall that is itself periodic).  Steady state: u(y) = f y (W - y) / (2 nu), mean f W^2 / (12 nu), the plate loads sum to rho f W L.  The profile is compared with the
parabola of the viscosity measured from a shear-wave decay of the same discretisation (`periodic_cylinder_array.shear_nu`); the ratio of the fitted amplitude to the parabola is the wall-model check (free slip would
flatten the profile, a no-slip wall that is too weak does too).

    python scripts/studies/periodic_channel.py --n 48 --W 0.5 --wall noslipMirror
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import warpSPHBoundaries  # noqa: F401
from periodic_cylinder_array import shear_nu
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, BoxRep, Scene
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.loads import LoadHistory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="particles per unit length (box length 1)")
    ap.add_argument("--W", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--c0", type=float, default=10.0)
    ap.add_argument("--f", type=float, default=0.05)
    ap.add_argument("--time", type=float, default=12.0)
    ap.add_argument("--H", type=float, default=4.0)
    ap.add_argument("--wall", default="noslipMirror")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dx = 1.0 / a.n
    nx, ny = a.n, int(round(a.W / dx))
    W = ny * dx
    X, Y = np.meshgrid(dx * (np.arange(nx) + .5), dx * (np.arange(ny) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    nu = shear_nu(a.n, a.alpha, a.c0, a.device, a.H)
    plate = lambda yc: Body(bodyId=0, center=(0.5, yc), reps=[BoxRep((-0.8, -0.15), (0.8, 0.15))])
    scene = Scene([plate(-0.15), plate(W + 0.15)], a.device)
    for i, b in enumerate(scene.bodies):
        b.bodyId = i
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=a.c0, alpha=a.alpha, periodic=Periodic((0, -9), (1, 9), (True, False)), bodyForce=(a.f, 0.0), graphStep=True, shifting=True, wallViscosityForm=a.wall)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, scene, cfg, a.device, support=a.H * dx)
    hist = LoadHistory()
    while sim.time < a.time:
        sim.step()
        hist.record(sim)
    y = sim.x[:, 1].cpu().numpy()
    u = sim.v[:, 0].cpu().numpy()
    exact = a.f * y * (W - y) / (2 * nu)
    amp = float((u * exact).sum() / (exact * exact).sum())
    umax = a.f * W * W / (8 * nu)
    bins = np.linspace(0, W, 11)
    prof = [u[(y >= b0) & (y < b1)].mean() / umax for b0, b1 in zip(bins[:-1], bins[1:])]
    F = sum(hist.total(b)[-300:, 0].mean() for b in range(2))
    Fbal = a.f * len(pos) * dx * dx
    print(f"n={a.n} W={W:.4f} N={len(pos)} nu_shear={nu:.5f} wall={a.wall}")
    print(f"profile amplitude / parabola(nu_shear) = {amp:.4f}   (1 = the wall is no-slip at y=0, W with the bulk viscosity)")
    print(f"u/umax over 10 bins: {np.round(prof, 3).tolist()}  (parabola: {np.round([(4 * ((i + .5) / 10) * (1 - (i + .5) / 10)) for i in range(10)], 3).tolist()})")
    print(f"plate loads F = {F:.5f}   body force on the fluid = {Fbal:.5f}   ratio {F / Fbal:.4f}")


if __name__ == "__main__":
    main()
