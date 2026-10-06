"""Curved no-slip walls with an exact solution: Taylor-Couette flow between a rotating inner cylinder (a convex wall, fluid outside) and a fixed outer cylinder (a concave wall, fluid inside), low Re.

u_theta(r) = omega r1^2 (r2^2 / r - r) / (r2^2 - r1^2); torque of the fluid on the inner cylinder per unit depth T = -4 pi mu omega r1^2 r2^2 / (r2^2 - r1^2), mu = rho nu with nu = the shear-wave viscosity
of the discretisation.  Reported: the profile amplitude (least squares of u_theta against the exact one), the torque ratio, and the torques of both walls (they are equal and opposite in steady state).

    python scripts/studies/taylor_couette.py --n 48 --wall noslipCurv
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
from warpSPHBoundaries.scene.implicitBodies import DiskBody
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, ImplicitRep, Scene
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.loads import LoadHistory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="particles per unit length")
    ap.add_argument("--r1", type=float, default=0.2)
    ap.add_argument("--r2", type=float, default=0.5)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--nu", type=float, default=0.0185)
    ap.add_argument("--c0", type=float, default=10.0)
    ap.add_argument("--H", type=float, default=4.0)
    ap.add_argument("--time", type=float, default=25.0)
    ap.add_argument("--wall", default="noslipCurv")
    ap.add_argument("--lever", default="particle", help="wallFrictionLever: contact | particle")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dx = 1.0 / a.n
    alpha = a.nu * 8 * 2.821384729 / (a.c0 * a.H * dx)
    g = np.arange(-int(a.r2 / dx) - 1, int(a.r2 / dx) + 2)
    X, Y = np.meshgrid(dx * (g + 0.5), dx * (g + 0.5), indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    r = np.linalg.norm(P, axis=1)
    pos = P[(r >= a.r1 + 0.5 * dx) & (r <= a.r2 - 0.5 * dx)]
    nu = shear_nu(a.n, alpha, a.c0, a.device, a.H)
    inner = Body(bodyId=0, center=(0.0, 0.0), angularVelocity=a.omega, reps=[DiskArrayRep([(0.0, 0.0)], [a.r1])])
    outer = Body(bodyId=1, center=(0.0, 0.0), reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=a.r2, solid="outside"))])
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=a.c0, alpha=alpha, graphStep=False, shifting=True, wallViscosityForm=a.wall, wallFrictionLever=a.lever)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, Scene([inner, outer], a.device), cfg, a.device, support=a.H * dx)
    hist = LoadHistory()
    while sim.time < a.time:
        sim.step()
        hist.record(sim)
    x = sim.x.cpu().numpy()
    v = sim.v.cpu().numpy()
    rr = np.linalg.norm(x, axis=1)
    ut = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / rr
    ex = a.omega * a.r1 ** 2 * (a.r2 ** 2 / rr - rr) / (a.r2 ** 2 - a.r1 ** 2)
    amp = float((ut * ex).sum() / (ex * ex).sum())
    T_ex = -4 * math.pi * nu * a.omega * a.r1 ** 2 * a.r2 ** 2 / (a.r2 ** 2 - a.r1 ** 2)
    T1 = hist.total(0)[-300:, 2].mean()
    T2 = hist.total(1)[-300:, 2].mean()
    bins = np.linspace(a.r1, a.r2, 9)
    prof = [ut[(rr >= b0) & (rr < b1)].mean() / ex[(rr >= b0) & (rr < b1)].mean() for b0, b1 in zip(bins[:-1], bins[1:])]
    print(f"n={a.n} N={len(pos)} nu_shear={nu:.5f} wall={a.wall} (Re = {a.omega * a.r1 * (a.r2 - a.r1) / nu:.1f})")
    absu = [(round(float(ut[(rr >= b0) & (rr < b1)].mean()), 4), round(float(ex[(rr >= b0) & (rr < b1)].mean()), 4)) for b0, b1 in zip(bins[:-1], bins[1:])]
    M = np.stack([rr, 1.0 / rr], 1)
    Ap, Bp = np.linalg.lstsq(M, ut, rcond=None)[0]
    T_fit = -4 * math.pi * nu * Bp                                      # the torque the measured profile transmits, 2 pi r^2 mu (u' - u / r) = -4 pi mu B'
    print(f"profile fit u = A' r + B' / r: A' = {Ap:.4f} (exact {-a.omega * a.r1 ** 2 / (a.r2 ** 2 - a.r1 ** 2):.4f}), B' = {Bp:.5f} (exact {a.omega * a.r1 ** 2 * a.r2 ** 2 / (a.r2 ** 2 - a.r1 ** 2):.5f}); torque of the fitted profile {T_fit:+.5f}; u at the inner wall from the fit {Ap * a.r1 + Bp / a.r1:.4f} (wall {a.omega * a.r1:.4f}), at the outer {Ap * a.r2 + Bp / a.r2:+.4f}")
    print(f"u_theta (measured, exact) by radial bin: {absu}")
    print(f"profile amplitude / exact = {amp:.4f}; by radial bin (inner -> outer): {np.round(prof, 3).tolist()}")
    print(f"torque on the inner cylinder {T1:+.5f}  exact {T_ex:+.5f}  ratio {T1 / T_ex:.4f};  torque on the outer {T2:+.5f} (about its centre; = -inner {-T1:+.5f}: ratio {T2 / -T1:.4f})")


if __name__ == "__main__":
    main()
