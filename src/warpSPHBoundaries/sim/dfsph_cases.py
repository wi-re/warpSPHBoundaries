"""Ready-made DFSPH2D cases on a calibrated rest lattice (omniSPH kernel/packing conventions)."""
import numpy as np

from .dfsph2d import DFSPH2D, DFSPHConfig, PACKING, carve, domain_scene, lattice_calibration
from ..scene.scene import Body, Scene, SurfaceRep, VolumeRep


def hexagon(Rh, kind="surface"):
    """regular hexagon of circumradius Rh around its centre (the body frame origin) as a surface loop or six triangles."""
    if kind == "surface":
        return SurfaceRep.regularPolygon((0, 0), Rh, 6, areaPreserving=False)
    P = np.array([[0.0, 0.0]] + [[Rh * np.cos(k * np.pi / 3), Rh * np.sin(k * np.pi / 3)] for k in range(6)])
    return VolumeRep(P, np.array([[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)]))


def tank_with_obstacle(L=0.6, H=0.45, fill=0.30, Rh=0.06, center=None, omega=0.0, rep="surface", r=0.005, device="cuda:0", cfg=None, domain="surface", fluidWidth=None):
    """a water tank [0, L] x [0, fill] (lattice, free surface above; `fluidWidth`: only the left block of that width is filled, a dam break) with a hexagonal obstacle
    rotating at `omega` (rad/s) about its centre, carved out of the lattice.
    Returns (sim, info); info holds the calibration, the obstacle area and the fluid weight."""
    h = r * np.sqrt(20.0)
    dx = dy = PACKING * h
    cal = lattice_calibration(dx, dy, h)
    nx, ny = int(round((fluidWidth or L) / dx)), int(round(fill / dy))
    X, Y = np.meshgrid(dx * (np.arange(nx) + 0.5), dy * (np.arange(ny) + 0.5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    center = center if center is not None else (0.5 * L, 0.5 * fill - 0.02)
    obst = Body(bodyId=1, center=center, angularVelocity=omega, reps=[hexagon(Rh, rep)])
    pos = carve(pos, [obst], h, cal["lamRow0"], device)
    lo = np.array([pos[:, 0].min() - cal["dwallX"], pos[:, 1].min() - cal["dwallY"]])
    hi = np.array([pos[:, 0].max() + cal["dwallX"] if fluidWidth is None else L, H])
    dom = domain_scene(domain, lo, hi, h, device).bodies[0]
    sc = Scene([dom, obst], device)
    cfg = cfg or DFSPHConfig(wallMass=cal["mu"])
    sim = DFSPH2D(pos, np.zeros_like(pos), cal["V"], np.full(len(pos), h), sc, cfg, device)
    info = dict(cal=cal, h=h, dx=dx, hexArea=1.5 * np.sqrt(3.0) * Rh ** 2, weight=float(sim.V.sum()) * 9.81, lo=lo, hi=hi, buoyancy=9.81 * 1.5 * np.sqrt(3.0) * Rh ** 2)
    return sim, info
