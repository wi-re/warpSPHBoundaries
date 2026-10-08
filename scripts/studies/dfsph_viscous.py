"""Viscous validation of the omniSPH-style DFSPH (docs/plan-next-steps.md "DFSPH track", D3): Morris viscosity + the shared no-slip wall closure (complement moments), periodic domains, a body force.

DFSPH conventions: rest density 1, lattice spacing dx = PACKING h (h the support, H / dx = 2.5), calibrated volumes and first-row wall distance (`lattice_calibration`), maxDt raised to the viscous / CFL limit (incompressible: no
acoustic limit).  Metrics as in the delta+ studies:

    shear    decay of a sine shear wave in a periodic box (no walls): nu_eff / nu (the Morris calibration of this lattice; 1 = calibrated)
    channel  plane Poiseuille between plates, periodic in x, body force f: profile amplitude against the parabola of nu (long-wave = the calibrated nu)
    array    a disk in a periodic unit cell (square array), Stokes, body force on the fluid only: superficial velocity U_x against the Sangani-Acrivos reference U_ref = f (1 - c) / (nu (1 - c) K_SA), and K = F / (nu U)
    couette  Taylor-Couette between two cylinders, the inner one rotating: profile amplitude against the exact u = A r + B / r, the torque on the inner cylinder

    python scripts/studies/dfsph_viscous.py shear | channel | array | couette  [--n 48] [--nu 0.0185]
"""
import argparse
import math
import time

import numpy as np
import torch

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, BoxRep, DiskArrayRep, Scene
from warpSPHBoundaries.sim.dfsph2d import DFSPH2D, DFSPHConfig, PACKING, carve, lattice_calibration

F64 = torch.float64


def sangani_acrivos(c):
    return 4 * math.pi / (-0.5 * math.log(c) - 0.738 + c - 0.887 * c ** 2 + 2.038 * c ** 3)


def base_cfg(a, cal, **kw):
    return DFSPHConfig(gravity=(0.0, 0.0), viscosity=a.nu, boundaryFriction=0.0, wallMass=cal["mu"], maxDt=a.maxDt, cfl=0.4, recordForces=True,
                       densityClamp=getattr(a, "clamp", True), **{**eval("dict(" + (getattr(a, "set", None) or "") + ")"), **kw})


def lattice(nx, ny, dx, x0=0.0, y0=0.0):
    X, Y = np.meshgrid(x0 + dx * (np.arange(nx) + 0.5), y0 + dx * (np.arange(ny) + 0.5), indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


def run_shear(a):
    dx = 1.0 / a.n
    h = dx / PACKING
    cal = lattice_calibration(dx, dx, h)
    pos = lattice(a.n, a.n, dx)
    vel = np.zeros_like(pos)
    vel[:, 0] = 0.05 * np.sin(2 * math.pi * pos[:, 1])
    sim = DFSPH2D(pos, vel, cal["V"], h, None, base_cfg(a, cal, periodic=Periodic((0, 0), (1, 1))), a.device)
    ts, A = [], []
    while sim.time < 0.25 / a.nu * 0.1:
        sim.step()
        ts.append(sim.time)
        A.append(float((sim.v[:, 0] * torch.sin(2 * math.pi * sim.x[:, 1])).mean() * 2))
    nu = -np.polyfit(ts, np.log(A), 1)[0] / (2 * math.pi) ** 2
    print(f"shear wave n={a.n} (H/dx {h / dx:.2f}): nu_eff / nu = {nu / a.nu:.4f}  (Morris calibration {sim._morris_cal():.4f}; |k| dx = {2 * math.pi * dx:.3f})")


def run_channel(a):
    dx = 1.0 / a.n
    h = dx / PACKING
    cal = lattice_calibration(dx, dx, h)
    ny = int(round(a.W / dx))
    dw = cal["dwallY"]
    W = 2 * dw + (ny - 1) * dx
    pos = lattice(a.n, ny, dx, y0=dw - 0.5 * dx)
    plate = lambda yc, k: Body(bodyId=k, center=(0.5, yc), reps=[BoxRep((-0.8, -0.15), (0.8, 0.15))])
    scene = Scene([plate(-0.15, 0), plate(W + 0.15, 1)], a.device)
    sim = DFSPH2D(pos, np.zeros_like(pos), cal["V"], h, scene, base_cfg(a, cal, periodic=Periodic((0, -9), (1, 9), (True, False)), bodyForce=(a.f, 0.0)), a.device)
    t0 = time.time()
    while sim.time < a.time:
        sim.step()
    y = sim.x[:, 1].cpu().numpy()
    u = sim.v[:, 0].cpu().numpy()
    exact = a.f * y * (W - y) / (2 * a.nu)
    amp = float((u * exact).sum() / (exact * exact).sum())
    F = sum(sum(hh["viscous"][:, 0]) + sum(hh["pressure"][:, 0]) for hh in sim.history[-200:]) / 200
    print(f"channel n={a.n}, W={W:.4f}: amplitude / parabola(nu) = {amp:.4f}   plate force / body force on the fluid = {F / (a.f * float(sim.V.sum())):.4f}   ({time.time() - t0:.0f} s, dt {sim.dt:.2e})")


def run_array(a):
    dx = 1.0 / a.n
    h = dx / PACKING
    cal = lattice_calibration(dx, dx, h)
    pos = lattice(a.n, a.n, dx)
    if a.shape == "square":                                       # grid-aligned: k lattice columns / rows removed, the faces at the rest distance dwall from the kept rows -> no cut cells, N dx^2 = the fluid area
        k = int(round(a.a / dx))
        if (a.n - k) % 2:
            k += 1
        half = 0.5 * (k + 1) * dx - cal["dwallY"]
        body = lambda: Body(bodyId=0, center=(0.5, 0.5), reps=[BoxRep((-half, -half), (half, half))])
        pos = pos[np.maximum(np.abs(pos[:, 0] - 0.5), np.abs(pos[:, 1] - 0.5)) > half]
        c = (2 * half) ** 2
        Kref = a.Kref if a.Kref else float("nan")                 # fluid-only K of this c (stokes_array_ref.py --square --fluid-only)
        Uref = a.f * (1 - c) / (a.nu * Kref)
        print(f"square side {2 * half / dx:.4f} dx = {2 * half:.6f}, c = {c:.6f}, N = {len(pos)}, N dx^2 / (1 - c) = {len(pos) * dx * dx / (1 - c):.5f}")
    else:
        body = lambda: Body(bodyId=0, center=(0.5, 0.5), reps=[DiskArrayRep([(0.0, 0.0)], [a.R])])
        pos = carve(pos, [body()], h, cal["lamCell" if a.carve == "cell" else "lamRow0"], a.device)
        c = math.pi * a.R ** 2
        Kref = (1 - c) * sangani_acrivos(c)
        Uref = a.f / (a.nu * sangani_acrivos(c))                  # superficial velocity of the reference (drag per cell f L^2 = nu U K_SA)
    scene = Scene([body()], a.device)
    sim = DFSPH2D(pos, np.zeros_like(pos), cal["V"], h, scene, base_cfg(a, cal, periodic=Periodic((0, 0), (1, 1)), bodyForce=(a.f, 0.0)), a.device)
    t0 = time.time()
    k = 0
    while sim.time < a.time:
        sim.step()
        k += 1
        if k % 500 == 0:
            print(f"  t {sim.time:.3f} dt {sim.dt:.2e} iters {sim.iters} vmax {float(sim.v.norm(dim=1).max()):.3e} rho [{float(sim.rho.min()):.4f}, {float(sim.rho.max()):.4f}] rhoSum [{float(getattr(sim, 'rhoSum', sim.rho).min()):.4f}, {float(getattr(sim, 'rhoSum', sim.rho).max()):.4f}] "
                  f"U_x / U_ref {float(sim.v[:, 0].sum()) * dx * dx / Uref:.4f}  p [{float(sim.p.min()):.2e}, mean {float(sim.p.mean()):.2e}, {float(sim.p.max()):.2e}]  {time.time() - t0:.0f} s", flush=True)
    if a.save:
        np.savez(a.save, x=sim.x.cpu().numpy(), v=sim.v.cpu().numpy(), c=c, dx=dx, nu=a.nu, f=a.f)
    U = float(sim.v[:, 0].sum()) * dx * dx
    F = np.mean([hh["viscous"][0, 0] + hh["pressure"][0, 0] for hh in sim.history[-200:]])
    Fbal = a.f * float(sim.V.sum())
    Fd = np.mean([hh["pressureDiv"][0, 0] for hh in sim.history[-200:]])
    Fv = np.mean([hh["viscous"][0, 0] for hh in sim.history[-200:]])
    print(f"  drag split / balance: viscous {Fv / Fbal:.4f}  divergence pressure {Fd / Fbal:.4f}  density pressure {(F - Fv - Fd) / Fbal:.4f}")
    print(f"array {a.shape} n={a.n}, c={c:.4f}, N={len(pos)}: U_x / U_ref = {U / Uref:.4f}   K / K_ref = {F / (a.nu * U) / Kref:.4f}   F / balance = {F / Fbal:.4f}   "
          f"({time.time() - t0:.0f} s, {len(sim.history)} steps)")


def run_couette(a):
    dx = 1.0 / a.n
    h = dx / PACKING
    cal = lattice_calibration(dx, dx, h)
    r1, r2, om = a.r1, a.r2, a.omega
    pos = lattice(int(2 * r2 / dx) + 2, int(2 * r2 / dx) + 2, dx, x0=-r2 - dx, y0=-r2 - dx)
    rr = np.linalg.norm(pos, axis=1)
    pos = pos[(rr > r1 + cal["dwallY"] - 0.01 * dx) & (rr < r2 - cal["dwallY"] + 0.01 * dx)]
    from warpSPHBoundaries.scene.implicitBodies import DiskBody
    from warpSPHBoundaries.scene.scene import ImplicitRep
    inner = Body(bodyId=0, center=(0.0, 0.0), angularVelocity=om, reps=[DiskArrayRep([(0.0, 0.0)], [r1])])
    outer = Body(bodyId=1, center=(0.0, 0.0), reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=r2, solid="outside"))])
    scene = Scene([inner, outer], a.device)
    sim = DFSPH2D(pos, np.zeros_like(pos), cal["V"], h, scene, base_cfg(a, cal), a.device)
    t0 = time.time()
    while sim.time < a.time:
        sim.step()
        inner.angle = 0.0                                             # an axisymmetric disk: keep the pose fixed (the wall velocity is what matters)
    x, v = sim.x.cpu().numpy(), sim.v.cpu().numpy()
    r = np.linalg.norm(x, axis=1)
    ut = (-x[:, 1] * v[:, 0] + x[:, 0] * v[:, 1]) / r
    A = -om * r1 ** 2 / (r2 ** 2 - r1 ** 2)
    B = om * r1 ** 2 * r2 ** 2 / (r2 ** 2 - r1 ** 2)
    ex = A * r + B / r
    amp = float((ut * ex).sum() / (ex * ex).sum())
    Tex = -4 * math.pi * a.nu * B                                      # torque of the fluid on the inner cylinder (rho = 1)
    T1 = np.mean([hh["torqueViscous"][0] for hh in sim.history[-200:]])
    print(f"couette n={a.n}, r1={r1}, r2={r2}: profile amplitude / exact = {amp:.4f}   viscous torque on the inner cylinder / exact = {T1 / Tex:.4f}   ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=("shear", "channel", "array", "couette"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--nu", type=float, default=0.0185)
    ap.add_argument("--f", type=float, default=0.03)
    ap.add_argument("--W", type=float, default=0.5)
    ap.add_argument("--R", type=float, default=0.2)
    ap.add_argument("--shape", choices=("disk", "square"), default="disk")
    ap.add_argument("--a", type=float, default=1.0 / 3.0, help="square: nominal side (rounded to whole lattice cells, faces at the rest distance)")
    ap.add_argument("--Kref", type=float, default=None, help="square: the fluid-only reference K of the printed c")
    ap.add_argument("--r1", type=float, default=0.2)
    ap.add_argument("--r2", type=float, default=0.5)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--time", type=float, default=30.0)
    ap.add_argument("--carve", choices=("row0", "cell"), default="cell")        # 'cell': volume-consistent (N dx^2 = the fluid area on average); 'row0' leaves a ~2 % deficit next to a disk that p >= 0 never closes
    ap.add_argument("--maxDt", type=float, default=0.01)
    ap.add_argument("--noClamp", dest="clamp", action="store_false", help="densityClamp=False: the signed pressure (closed domains)")
    ap.add_argument("--set", default="", help="extra DFSPHConfig fields, e.g. \"gradientCorrection='wall', closedDomain=False\"")
    ap.add_argument("--save", default=None, help="array: save the final positions / velocities (npz)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"shear": run_shear, "channel": run_channel, "array": run_array, "couette": run_couette}[a.case](a)


if __name__ == "__main__":
    main()
