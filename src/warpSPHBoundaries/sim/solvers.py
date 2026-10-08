"""One interface to the two fluid solvers of this package, for the example notebooks and the cross-scheme studies.

    delta   `DeltaSPH2D`: weakly compressible delta+-SPH (equation of state, particle shifting)
    dfsph   `DFSPH2D`:    divergence-free SPH (omniSPH-style; the converged compact projection in closed / periodic flows)

`make_solver(scheme, ...)` builds either from the same description of the flow: the lattice spacing `dx` of the (cut) initial lattice, the fluid's kinematic viscosity `nu`, the scene of the walls, the driving
(gravity, body force, periodic box, prescribed-velocity frame) and the initial velocity.  What differs between the schemes is made here and nowhere else: the support (delta: 4 dx; dfsph: h = dx / PACKING on the
calibrated lattice), the viscosity (both the calibrated Morris operator with the no-slip moment closure at the walls; delta sets alpha from nu through nu = alpha c0 H / (8 xi)), the speed of sound (delta only), the
particle volume (dfsph: the lattice calibration; delta: dx^2) and the wall mass.  The returned `Solver` exposes the common part: `step()`, `time`, `x`, `v`, `rho`, `volume`, `loads(body)` (the fluid's force on a
body per step) and the bookkeeping that the examples share (`run(until, every, callback)`).

The lattice the caller generates must have spacing `dx` (for dfsph the lattice of `lattice_calibration(dx, dx, dx / PACKING)`: particles at the cell centres, the first row `dwall` from a wall, see the examples).
"""
import math
import time as _time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from .deltasph2d import DeltaSPH2D, DeltaSPHConfig
from .dfsph2d import DFSPH2D, DFSPHConfig, PACKING, lattice_calibration

SCHEMES = ("delta", "dfsph")
XI = 2.821384729                    # sphKernel_xi(Wendland C2, 2D): nu = alpha c0 H / (8 xi) for the pair viscosity
MORRIS_CAL = 0.985                  # nu_eff / nu of the Morris operator on the Wendland C2 square lattice at H = 4 dx (kernel_viscosity_table.py --morris)


@dataclass
class Solver:
    """a running solver behind the common interface (see the module docstring)."""
    scheme: str
    sim: object
    dx: float
    nu: float
    cal: Optional[dict] = None              # dfsph: the lattice calibration (V, mu, dwall ...)
    series: dict = field(default_factory=dict)

    # ------------------------------------------------------------------------------------------------ state
    @property
    def time(self):
        return float(self.sim.time)

    @property
    def dt(self):
        return float(self.sim.dt)

    @property
    def x(self):
        return self.sim.x.detach().cpu().numpy()

    @property
    def v(self):
        return self.sim.v.detach().cpu().numpy()

    @property
    def rho(self):
        return self.sim.rho.detach().cpu().numpy()

    @property
    def pressure(self):
        """particle pressure: the equation of state of delta (c0^2 (rho - rho0)), the pressure of the solve for dfsph."""
        if self.scheme == "dfsph":
            return self.sim.p.detach().cpu().numpy()
        return (self.sim.cfg.c0 ** 2 * (self.sim.rho - self.sim.cfg.rho0)).detach().cpu().numpy()

    @property
    def volume(self):
        """particle volumes (area in 2D)."""
        if self.scheme == "dfsph":
            return self.sim.V.detach().cpu().numpy()
        return (self.sim.m / self.sim.rho).detach().cpu().numpy()

    @property
    def n(self):
        return len(self.sim.x)

    def step(self):
        self.sim.step()
        return self.time

    # ------------------------------------------------------------------------------------------------ loads
    def loads(self, body=0):
        """(Fx, Fy) of the fluid on the body in the step just taken (pressure + viscous / friction), force per unit depth.  Delta: the wall terms booked by the step; dfsph: the booked pressure, friction and
        viscous force of the step (`recordForces`, on by default)."""
        if self.scheme == "delta":
            return self.sim.wallLoads[:, body, :2].sum(0).detach().cpu().numpy()
        h = self.sim.history[-1]
        return np.asarray(h["pressure"][body]) + np.asarray(h["friction"][body]) + np.asarray(h["viscous"][body])

    def torque(self, body=0):
        """torque (z) of the fluid on the body about its centre in the step just taken; delta: pressure + viscous + impulse terms; dfsph: the booked viscous torque (the pressure torque of the density solve is
        not booked: exact for a disk, whose pressure acts through the centre)."""
        if self.scheme == "delta":
            return float(self.sim.wallLoads[:, body, 2].sum())
        return float(self.sim.history[-1]["torqueViscous"][body])

    # ------------------------------------------------------------------------------------------------ driving
    def run(self, until, every=None, callback: Optional[Callable] = None, report=None, maxSteps=10 ** 9):
        """step to `until` (simulation time); `callback(solver)` after every step, `report(solver)` (returns a string) every `every` seconds of simulation time and at the end.  Stops (returns False) when the speed
        becomes non-finite or exceeds 1e3."""
        t0, nextReport, k = _time.time(), self.time + (every or 0.0), 0
        while self.time < until - 1e-12 and k < maxSteps:
            self.step()
            k += 1
            if callback is not None:
                callback(self)
            if every and report is not None and self.time >= nextReport - 1e-12:
                nextReport += every
                vmax = float(self.sim.v.norm(dim=1).max())
                print(f"  t {self.time:8.3f}  dt {self.dt:.2e}  vmax {vmax:8.3f}  {report(self)}  [{_time.time() - t0:.0f} s]", flush=True)
                if not math.isfinite(vmax) or vmax > 1e3:
                    print("  diverged")
                    return False
        return True


def results_dir():
    """results/examples of the checkout (created on demand); the notebooks store the series of a run there, so the run of the other scheme can be overlaid later."""
    import os
    from .. import paths
    d = paths.results_dir() / "examples"
    os.makedirs(d, exist_ok=True)
    return str(d)


def save_series(case, scheme, **arrays):
    """store arrays of a run as results/examples/<case>_<scheme>.npz."""
    np.savez(f"{results_dir()}/{case}_{scheme}.npz", **{k: np.asarray(v) for k, v in arrays.items()})


def load_series(case, scheme):
    """the arrays stored by `save_series` (a dict), or None when that scheme has not been run."""
    import os
    path = f"{results_dir()}/{case}_{scheme}.npz"
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}


def lattice(lo, hi, dx):
    """cell-centre lattice of spacing dx filling the box [lo, hi] (the extent is rounded to whole cells), [N, 2]."""
    nx, ny = int(round((hi[0] - lo[0]) / dx)), int(round((hi[1] - lo[1]) / dx))
    X, Y = np.meshgrid(lo[0] + dx * (np.arange(nx) + 0.5), lo[1] + dx * (np.arange(ny) + 0.5), indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


def wall_gap(scheme, dx):
    """distance of the first particle row from a wall for the lattice of `scheme`: dx / 2 (delta), the calibrated `dwall` of the rest lattice (dfsph, 0.552 dx)."""
    return 0.5 * dx if scheme == "delta" else lattice_calibration(dx, dx, dx / PACKING)["dwallX"]


def box_lattice(scheme, lo, cells, dx, walls=(False, False)):
    """the lattice of a rectangular fluid block with `cells` = (nx, ny) rows: a wall (True) at an end of an axis puts the first row `wall_gap` from it (the block is then (n - 1) dx + 2 gap long), a free /
    periodic end (False) is a half spacing away (n dx long).  Returns (positions [N, 2], hi) with hi the far corner of the block."""
    g = wall_gap(scheme, dx)
    axes, hi = [], []
    for a in range(2):
        off = g if walls[a] else 0.5 * dx
        axes.append(lo[a] + off + dx * np.arange(cells[a]))
        hi.append(lo[a] + (cells[a] - 1) * dx + 2 * off)
    X, Y = np.meshgrid(*axes, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1), tuple(hi)


def lattice_constants(scheme, dx):
    """the per-scheme constants of a lattice of spacing dx: support, particle volume, and (dfsph) the calibration dict."""
    if scheme == "dfsph":
        h = dx / PACKING
        cal = lattice_calibration(dx, dx, h)
        return dict(h=h, V=cal["V"], cal=cal)
    return dict(h=4.0 * dx, V=dx * dx, cal=None)


def make_solver(scheme, pos, dx, scene=None, *, nu=0.0, vel=None, gravity=(0.0, 0.0), bodyForce=(0.0, 0.0), periodic=None, pinned=None, c0=10.0, rho0=1.0, device="cuda:0", **extra):
    """`Solver` for `scheme` ('delta' | 'dfsph') from the common description.  `pos` [N, 2] the initial lattice (spacing `dx`), `vel` [N, 2] the initial velocity (default rest), `nu` the kinematic viscosity (0:
    inviscid), `c0` the speed of sound of delta (in the units of the flow; the Mach number c0 / U should be >= 10), `rho0` the initial density of delta (1 = the rest density; the pressure level of the equation of state),
    `extra` fields of the scheme's config (e.g. `shifting=False`, `projection='compact'`)."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    pos = np.asarray(pos, float)
    vel = np.zeros_like(pos) if vel is None else np.asarray(vel, float)
    k = lattice_constants(scheme, dx)
    if scheme == "delta":
        H = k["h"]
        if nu > 0:           # viscous flow: the calibrated Morris operator with the no-slip moment closure, pressure-consistent walls
            kw = dict(gravity=tuple(gravity), c0=c0, alpha=nu * 8.0 * XI / (c0 * H), periodic=periodic, pinned=pinned, bodyForce=tuple(bodyForce), fluidViscosity="morris", morrisCalibration=MORRIS_CAL,
                      wallViscosityForm="noslipMoment", pressureConsistent=True, shifting=True, graphStep=True)
        else:                # inviscid flow (free-surface cases): the artificial-viscosity form of the case defaults (alpha = 0.01), free-slip walls, time-centred continuity
            kw = dict(gravity=tuple(gravity), c0=c0, periodic=periodic, pinned=pinned, bodyForce=tuple(bodyForce), timeCentred=True, graphStep=True)
        kw.update(extra)
        sim = DeltaSPH2D(pos, vel, np.full(len(pos), rho0), dx, scene, DeltaSPHConfig(**kw), device, support=H)
        return Solver("delta", sim, dx, nu)
    cal = k["cal"]
    kw = dict(gravity=tuple(gravity), bodyForce=tuple(bodyForce), periodic=periodic, viscosity=nu, boundaryFriction=0.0 if nu > 0 else 5e-3, wallMass=cal["mu"], recordForces=True)
    if pinned is not None:
        kw["pinned"] = pinned
    kw.update(extra)
    sim = DFSPH2D(pos, vel, cal["V"], k["h"], scene, DFSPHConfig(**kw), device)
    return Solver("dfsph", sim, dx, nu, cal=cal)
