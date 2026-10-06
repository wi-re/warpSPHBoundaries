"""Test cases of the delta+-SPH solver: flat tank (English 2022 s.4.1), English wedge, Marrone 3.1 dam break, SPHERIC test case 10 sloshing.
Each builder returns `(sim, info)`; probes and scoring are in `probes.py` / `validation.py`."""
import math
from typing import Optional

import numpy as np
import torch

from warpSPHCore import KernelFunctions

from ..scene.scene import Body, Scene, SurfaceRep
from .deltasph2d import DeltaSPH2D, DeltaSPHConfig
from .dfsph2d import domain_scene
from .pairs import F64


def hydrostatic_tank(dp=0.02, L=2.4, Htank=1.2, Hwater=0.5, c0Ratio=20.0, domain="surface", device="cuda:0", cfg: Optional[DeltaSPHConfig] = None):
    """English et al. 2022 s.4.1 still water: tank L x Htank (inner faces), water of depth Hwater on the hydrostatic density profile, particles at the lattice mid-points (first row dp/2 from every wall,
    as warpSPH's `alignBoundaryLattice`), h/dp = 2 (support 4 dp), c0 = c0Ratio sqrt(g Hwater)."""
    g = 9.81
    c0 = c0Ratio * math.sqrt(g * Hwater)
    nx, ny = int(round(L / dp)), int(round(Hwater / dp))
    X, Y = np.meshgrid(-L / 2 + dp * (np.arange(nx) + 0.5), -Htank / 2 + dp * (np.arange(ny) + 0.5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    depth = np.clip(-Htank / 2 + Hwater - pos[:, 1], 0.0, None)
    rho = 1.0 * (1.0 + g * depth / c0 ** 2)
    scene = domain_scene(domain, (-L / 2, -Htank / 2), (L / 2, Htank / 2), 4 * dp, device)
    cfg = cfg or DeltaSPHConfig(gravity=(0.0, -g), c0=c0)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), rho, dp, scene, cfg, device)
    return sim, dict(g=g, c0=c0, Hwater=Hwater, bed=-Htank / 2, nx=nx, ny=ny)


WEDGE_APEX = (0.0, 0.24)                    # above the bed
WEDGE_HALF_BASE = 0.27733333333333343       # warpSPH `equilateralBottom` (aspectRatio 2, maxExtent 0.2903): apex 0.24 above the bed, half base 0.2773 (98 deg apex)


def triangle_distance(p, tri):
    """signed distance of points p [M,2] to the triangle tri [3,2] (negative inside); numpy."""
    a, b, c = tri
    d = np.full(len(p), np.inf)
    for u, w in ((a, b), (b, c), (c, a)):
        e = w - u
        t = np.clip(((p - u) @ e) / (e @ e), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(p - (u + t[:, None] * e), axis=1))
    def sgn(p1, p2, p3):
        return (p1[:, 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[:, 1] - p3[1])
    d1, d2, d3 = sgn(p, a, b), sgn(p, b, c), sgn(p, c, a)
    inside = ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
    return np.where(inside, -d, d)


def english_wedge(dp=0.02, L=2.4, Htank=1.2, Hwater=0.5, c0Ratio=20.0, domain="surface", device="cuda:0", cfg: Optional[DeltaSPHConfig] = None, margin=0.5, wedge=True, staircase=False):
    """English et al. 2022 s.4.1 with the sharp wedge on the bed (apex 0.24 m above the bed, `WEDGE_HALF_BASE`, centred): the flat tank of `hydrostatic_tank` plus a `SurfaceRep` triangle body
    with exact corners.  Lattice points closer than `margin * dp` to the wedge are removed (the first fluid row keeps the half-spacing it has at the flat walls)."""
    g = 9.81
    c0 = c0Ratio * math.sqrt(g * Hwater)
    bed = -Htank / 2
    nx, ny = int(round(L / dp)), int(round(Hwater / dp))
    X, Y = np.meshgrid(-L / 2 + dp * (np.arange(nx) + 0.5), bed + dp * (np.arange(ny) + 0.5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    tri = np.array([[-WEDGE_HALF_BASE, bed], [WEDGE_HALF_BASE, bed], [WEDGE_APEX[0], bed + WEDGE_APEX[1]]])
    stair = None
    if wedge and staircase:
        # control: the wall IS the lattice staircase (the union of the removed lattice cells), as for warpSPH's wall particles: every first fluid row is exactly dp/2 from the wall
        rem = triangle_distance(pos, tri) < 0
        cols = np.round((pos[:, 0] - (-L / 2 + 0.5 * dp)) / dp).astype(int)
        top = np.zeros(nx)
        for c in np.unique(cols[rem]):
            top[c] = (rem & (cols == c)).sum() * dp
        pts = []
        for c in range(nx):
            if top[c] > 0:
                xl, xr = -L / 2 + c * dp, -L / 2 + (c + 1) * dp
                pts += [(xl, bed + top[c]), (xr, bed + top[c])]
        first = next(c for c in range(nx) if top[c] > 0)
        last = max(c for c in range(nx) if top[c] > 0)
        poly = [(-L / 2 + first * dp, bed)] + pts + [(-L / 2 + (last + 1) * dp, bed)]
        stair = np.array(poly)
        pos = pos[~rem]
    elif wedge:
        pos = pos[triangle_distance(pos, tri) >= margin * dp]
    depth = np.clip(bed + Hwater - pos[:, 1], 0.0, None)
    rho = 1.0 * (1.0 + g * depth / c0 ** 2)
    dom = domain_scene(domain, (-L / 2, bed), (L / 2, bed + Htank), 4 * dp, device).bodies[0]
    bodies = [dom]
    if wedge:
        bodies.append(Body(bodyId=1, reps=[SurfaceRep.polygon(stair if stair is not None else tri, solid="inside")]))
    cfg = cfg or DeltaSPHConfig(gravity=(0.0, -g), c0=c0)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), rho, dp, Scene(bodies, device), cfg, device)
    return sim, dict(g=g, c0=c0, Hwater=Hwater, bed=bed, nx=nx, ny=ny, tri=tri, L=L)


def marrone_dambreak(nx=67, c0Ratio=40.0, domain="surface", device="cuda:0", cfg: Optional[DeltaSPHConfig] = None, **cfgkw):
    """Marrone et al. 2011 s.3.1 as warpSPH's `dambreak` case builds it (`probe_deltaSPHMarrone.py`): column 2H x H (H = 0.6) in the left corner of a closed tank 5.366 H long, ceiling 0.985 m above the bed
    (L = 1 m, dx = L/nx, walls dx/2 outside the first lattice row on all sides), c0 = c0Ratio sqrt(g H), uniform rho0 at t = 0 (no hydrostatic init), time-centred continuity on.
    x-spacing of warpSPH's lattice is W/round(W/dx) (0.13 % smaller at nx = 67); here dx in both directions."""
    H, L, g = 0.6, 1.0, 9.81
    Wt = 5.366 * H
    dx = L / nx
    c0 = c0Ratio * math.sqrt(g * H)
    xl, xr, yb, yt = -Wt / 2 + dx / 2, Wt / 2 - dx / 2, -L / 2 + dx / 2, L / 2 - dx / 2
    ncol, nrow = int(math.ceil(2 * H / dx - 1e-9)), int(math.floor(H / dx + 1e-9))
    X, Y = np.meshgrid(xl + dx * (np.arange(ncol) + 0.5), yb + dx * (np.arange(nrow) + 0.5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    scene = domain_scene(domain, (xl, yb), (xr, yt), 4 * dx, device)
    cfg = cfg or DeltaSPHConfig(gravity=(0.0, -g), c0=c0, timeCentred=True, **cfgkw)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, scene, cfg, device)
    return sim, dict(g=g, c0=c0, H=H, Wt=Wt, xl=xl, xr=xr, yb=yb, yt=yt, dx=dx, nx=nx)


# ---------------------------------------------------------------------------------------------------------------------------- SPHERIC test case 10 (sloshing tank)
SPHERIC_DIR = "/home/lu26029/dev/warpSPH/examples/sloshingTank/SPHERIC_TestCase10/data_files"


def load_roll(path=None):
    """SPHERIC roll table lateral_water_1x.txt -> (t [s], theta [rad], measured sensor pressure [Pa]); columns t, p [mbar], smoothed roll angle [deg], ..."""
    raw = np.genfromtxt(path or SPHERIC_DIR + "/lateral_water_1x.txt", delimiter="\t", skip_header=1)
    return raw[:, 0], np.radians(raw[:, 2]), raw[:, 1] * 100.0


def sloshing_tank(nx=200, T_unused=None, domain="surface", device="cuda:0", cfg: Optional[DeltaSPHConfig] = None, rollFile=None, **cfgkw):
    """SPHERIC test case 10, lateral water, as warpSPH's `sloshingTank` case (`examples/sloshingTank`): tank 0.9 x 0.508 m, still water 0.093 m (rows of dx = 0.9/nx up to the fill depth), Wendland C4, support 4 dx,
    c0 = 20, constant dt = 1e-4, isothermal EOS, alpha = 0.02, time-centred continuity, free-slip walls.  The tank is NOT moved: it rolls in the tank-fixed frame by rotating gravity,
    g(t) = 9.81 (-sin theta(t), -cos theta(t)), theta from the measured roll table, updated after every step (as warpSPH's `postStep`).  Sensor 1 at (-0.45, 0.093) on the left wall."""
    L, Ht, fill, g = 0.9, 0.508, 0.093, 9.81
    dx = L / nx
    nrow = int(round(fill / dx))
    X, Y = np.meshgrid(-L / 2 + dx * (np.arange(nx) + 0.5), dx * (np.arange(nrow) + 0.5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    scene = domain_scene(domain, (-L / 2, 0.0), (L / 2, Ht), 4 * dx, device)
    cfg = cfg or DeltaSPHConfig(gravity=(0.0, -g), c0=20.0, alpha=0.02, kernel=KernelFunctions.Wendland4, fixedDt=1e-4, timeCentred=True, **cfgkw)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, scene, cfg, device)
    t, th, pexp = load_roll(rollFile)
    sim.gravityFn = lambda tt: (-g * math.sin(float(np.interp(tt, t, th))), -g * math.cos(float(np.interp(tt, t, th))))
    sim.g = torch.tensor(sim.gravityFn(0.0), dtype=F64, device=device)
    return sim, dict(g=g, L=L, Ht=Ht, fill=fill, dx=dx, sensor=(-L / 2, fill), roll=(t, th, pexp), rho0Phys=1000.0)
