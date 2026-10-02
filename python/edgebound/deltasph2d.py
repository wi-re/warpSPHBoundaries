"""delta+-SPH (weakly compressible, Sun et al. 2017) in 2D on the exact-integral scene boundary layer.

Follows warpSPH's `sun2017DeltaSPH` / `deltaSPH` scheme term by term (`schemes/deltaSPH.py::_deltaSPH_rhs`; the map to the warpSPH modules is in docs/deltasph-porting-notes.md), with the
boundary-particle population (mDBC ghost nodes, free-slip mirror, no-penetration shift) replaced by integrals over the scene:

    isothermal EOS      P = c0^2 (rho - rho0)
    continuity          drho/dt = -rho_i sum_j V_j (v_j - v_i).grad_i W_ij   + wall:  2 rho_i sum_b u_n,b |mu grad lambda_b|        (free-slip mirror: (v_b - v_i) = -2 u_n n)
    pressure force      a_i = -(1/rho_i) [ sum_j V_j (P_j + s_i P_i) grad_i W_ij + wall ]        s_i = +1 if P_i >= 0 or i is a free-surface particle (Antuono switch, Sun 2018 Eq. 9), else -1
                        wall: sum_b [ (p_i^+ + s_i P_i) mu grad lambda_b + mu int (a1.y) grad W ],  p_b(x') = p_i^+ + rho0 (g - a_w).(x' - x_i), clamped >= 0 (see below)
    density diffusion   fourtakas2019, FLUID-TO-FLUID only (as warpSPH / DualSPHysics: no wall term):
                        d(rho)/dt += delta H c0 / xi  sum_j V_j psi_ij . grad_i W_ij ,   psi_ij = -2 [ (rho_j - rho_i) + rho0 g.x_ij / c0^2 ] x_ij / r_ij^2
    artificial visc.    dv_i = alpha c0 H / xi  sum_j V_j mu_ij grad_i W_ij / mean(rho_i, rho_j),   mu_ij = (v_i - v_j).x_ij / (r_ij^2 + 1e-14 H^2)     [wall term: not yet]
    free surface        Barecasco: cover vector C = sum_j unit(x_i - x_j); surface iff no neighbour within pi/6 of C/|C|.  The wall counts as the continuum of wall particles
                        (number density mu / dx^2): the solid is sampled on a polar grid around the particle with `Scene.inside`
    time integration    DualSPHysics symplectic Euler: k0 at t^n; half step; k1 at the half state; v^{n+1} = v^n + dt a_1, x^{n+1} = x^n + dt (v^n + v^{n+1}) / 2, rho^{n+1} = rho^n + dt drho_1
    time step           Sun 2017 Eq. (5): min(viscous, acoustic CFL, acceleration), growth <= 1.1

Units as warpSPH (rest density 1, P* = P / (rho0 g H)); Wendland C2, support H = 4 dx (h/dx = 2), mass m = rho0 dx^2, V_j = m / rho_j.

Wall pressure: `P_b >= 0` (`clampWallPressure`): the hydrostatic extrapolation is a suction at a ceiling (dfsph-validation.md s.7); warpSPH's mDBC carries the ghost's own (possibly negative) pressure there.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation

from .dfsph2d import F64, dwendland2, neighbor_pairs
from .scene import BodyField, Scene, buildCellList, sceneOperation  # noqa: F401  (buildCellList re-exported for callers)

XI = 2.8213846683502197                 # warpSPHCore sphKernel_xi(Wendland2, 2D) = packing * kernelScale
KSCALE = 1.897367                       # warpSPHCore sphKernelScale(Wendland2, 2D): support / smoothing length


@dataclass
class DeltaSPHConfig:
    gravity: tuple = (0.0, -9.81)
    rho0: float = 1.0
    c0: float = 44.29
    delta: float = 0.1                  # density diffusion coefficient
    alpha: float = 0.01                 # artificial viscosity (inviscid path)
    cfl: float = 0.3
    maxDt: float = 1e-2
    minDt: float = 1e-8
    growth: float = 1.1
    viscosity: bool = True
    ddt: bool = True
    wallMass: float = 1.0               # mass per area of the wall continuum in units of rho0 (a wall made of the same lattice: 1)
    clampWallPressure: bool = True
    wallContinuity: bool = True         # free-slip mirror term in the continuity equation (ablation switch)
    barecascoThreshold: float = math.pi / 3
    surfaceSamples: tuple = (8, 48)     # radial x angular samples of the solid around a particle (wall part of the free-surface detector)
    kernel: KernelFunctions = KernelFunctions.Wendland2


class DeltaSPH2D:
    def __init__(self, positions, velocities, densities, dx, scene: Optional[Scene], cfg: Optional[DeltaSPHConfig] = None, device="cuda:0", support=None):
        self.cfg = cfg or DeltaSPHConfig()
        self.dev = device
        t = lambda a: torch.as_tensor(a, dtype=F64, device=device)
        self.x, self.v, self.rho = t(positions).clone(), t(velocities).clone(), t(densities).clone()
        n = len(self.x)
        self.dx = float(dx)
        self.H = float(support if support is not None else 4.0 * dx)
        self.m = self.cfg.rho0 * self.dx ** 2
        self.Hvec = torch.full((n,), self.H, dtype=F64, device=device)
        self.scene = scene
        self.nb = len(scene.bodies) if scene is not None else 0
        self.kinds = torch.zeros(n, dtype=torch.int32, device=device)
        self.g = torch.tensor(self.cfg.gravity, dtype=F64, device=device)
        self.time = 0.0
        self.dt = self.cfg.cfl * self.H / (self.cfg.c0 * KSCALE)
        self.wallForce = torch.zeros((self.nb, 2), dtype=F64, device=device)      # force of the fluid on every body (pressure part), last RHS call
        self.surface = torch.zeros(n, dtype=torch.bool, device=device)
        self.history = []

    # ---------------------------------------------------------------------------------------------------------------- helpers
    def _sum(self, vals, i):
        out = torch.zeros((len(self.x),) + vals.shape[1:], dtype=vals.dtype, device=self.dev)
        return out.index_add_(0, i, vals)

    def _props(self, op, mode=GradientScheme.Naive):
        return OperationProperties(kernel=self.cfg.kernel, operation=op, gradientMode=mode, operationMode=OperationDirection.BoundaryToFluid)

    def _wall_op(self, ps, adj, op, flds, mode=GradientScheme.Naive):
        flds = flds if isinstance(flds, list) else [flds] * self.nb
        return self.cfg.wallMass * sceneOperation(ps, self._props(op, mode), self.scene, adj, None, flds, perBody=True)

    def _wall_data(self, x, rho):
        """per-body wall integrals at positions x: lam [B,N], G = mu grad lambda [B,N,2], A = mu int (a1.y) grad W [B,N,2] with a1 = rho0 (g - a_wall)."""
        ps = ParticleState(positions=x, supports=self.Hvec, masses=torch.full_like(rho, self.m), kinds=self.kinds, densities=rho)
        adj = self.scene.buildAdjacency(ps, self._props(WarpOperation.Density))
        one = BodyField(torch.tensor(1.0, dtype=F64, device=self.dev))
        lam = self._wall_op(ps, adj, WarpOperation.Density, BodyField(rho=1.0))
        G = self._wall_op(ps, adj, WarpOperation.Gradient, one)
        flds = []
        for b in self.scene.bodies:
            a1 = self.cfg.rho0 * (self.g[None] - b.accelerationAt(x))
            flds.append(BodyField(torch.zeros(len(x), dtype=F64, device=self.dev), a1, rho=1.0, perQuery=True))
        A = self._wall_op(ps, adj, WarpOperation.Gradient, flds)
        return lam, G, A

    def _wall_excess(self, A, G, pp):
        """p_b >= 0 (dfsph-validation.md s.7): the hydrostatic term mu int (a1.y) grad W is an effective wall pressure offset q (times mu grad lambda); the effective wall pressure p_i + q is clamped
        at 0 by removing (1 - theta) q G, theta = clip(p_i^+ / (-q), 0, 1) for q < 0 (only the normal offset, so round-off cannot flip the force)."""
        eps = 1e-5 * self.cfg.wallMass / self.H
        q = (A * G).sum(2) / (G * G).sum(2).clamp(min=eps * eps)
        theta = torch.where(q < 0, (pp[None] / (-q).clamp(min=1e-300)).clamp(0.0, 1.0), torch.ones_like(q))
        return (1.0 - theta) * q

    # ---------------------------------------------------------------------------------------------------------------- free-surface detector (Barecasco)
    def _detect_surface(self, x, i, j, r, lam):
        """warpSPH `detectFreeSurfaceBarecasco`: C = sum unit(x_i - x_j) over neighbours (W > 0), c = C/|C|; surface iff no neighbour (j != i) has angle(x_j - x_i, c) <= threshold/2
        (all neighbours count when C = 0).  Wall: the continuum of wall particles (density mu/dx^2), sampled on a polar grid with `Scene.inside`."""
        cfg, N = self.cfg, len(x)
        nz = i != j
        ii, jj, rr = i[nz], j[nz], r[nz]
        d = x[ii] - x[jj]
        unit = d / rr.clamp(min=1e-300)[:, None]
        C = self._sum(unit, ii)
        nw = cfg.wallMass / self.dx ** 2
        sample = None
        if self.scene is not None:
            near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
            if len(near):
                nr, nphi = cfg.surfaceSamples
                rk = (torch.arange(nr, dtype=F64, device=self.dev) + 0.5) * self.H / nr
                ph = (torch.arange(nphi, dtype=F64, device=self.dev) + 0.5) * 2 * math.pi / nphi
                u = torch.stack([torch.cos(ph), torch.sin(ph)], 1)                                   # [P,2]
                w = (rk * (self.H / nr) * (2 * math.pi / nphi))[:, None].expand(nr, nphi)            # area weights [R,P]
                pts = x[near][:, None, None, :] + rk[None, :, None, None] * u[None, None, :, :]      # [Q,R,P,2]
                ins = self.scene.inside(pts.reshape(-1, 2)).reshape(len(near), nr, nphi)
                wt = ins.to(F64) * w[None]                                                           # solid area per sample
                Cw = -nw * (wt[..., None] * u[None, None]).sum((1, 2))                               # sum unit(x_i - p) = -u
                C = C.index_add(0, near, Cw)
                sample = (near, u, wt)
        norm = C.norm(dim=1)
        c = C / norm.clamp(min=1e-300)[:, None]
        cosang = -((unit) * c[ii]).sum(1)                                                            # -n_ij . c
        inCone = torch.acos(cosang.clamp(-1.0, 1.0)) <= cfg.barecascoThreshold / 2
        count = self._sum(inCone.to(F64), ii)
        if sample is not None:
            near, u, wt = sample
            cn = (u[None] * c[near][:, None, :]).sum(2)                                              # [Q,P]
            cone = (torch.acos(cn.clamp(-1.0, 1.0)) <= cfg.barecascoThreshold / 2).to(F64)
            count = count.index_add(0, near, nw * (wt * cone[:, None, :]).sum((1, 2)))
        allcount = self._sum(torch.ones_like(rr), ii)
        if sample is not None:                                                                       # C = 0: all neighbours count (wall particles too)
            near, u, wt = sample
            allcount = allcount.index_add(0, near, nw * wt.sum((1, 2)))
        count = torch.where(norm > 1e-12, count, allcount)
        return count < 0.5

    # ---------------------------------------------------------------------------------------------------------------- right-hand side
    def rhs(self, x, v, rho, want_forces=False):
        cfg = self.cfg
        H, V = self.H, self.m / rho
        i, j, r = neighbor_pairs(x, self.Hvec)
        nz = i != j
        d = x[i] - x[j]
        gW = torch.where(nz[:, None], dwendland2(r, H)[:, None] * d / r.clamp(min=1e-300)[:, None], torch.zeros_like(d))      # grad_i W_ij
        P = cfg.c0 ** 2 * (rho - cfg.rho0)
        if self.scene is not None:
            lam, G, A = self._wall_data(x, rho)
        else:
            lam = torch.zeros((0, len(x)), dtype=F64, device=self.dev)
            G = torch.zeros((0, len(x), 2), dtype=F64, device=self.dev)
            A = G
        self.surface = self._detect_surface(x, i, j, r, lam)
        sw = (P >= 0) | self.surface
        s = torch.where(sw, torch.ones_like(P), -torch.ones_like(P))

        # continuity: fluid pairs, wall (free-slip mirror)
        drho = -rho * self._sum(V[j] * ((v[j] - v[i]) * gW).sum(1), i)
        if self.nb and cfg.wallContinuity:
            for bi, b in enumerate(self.scene.bodies):
                gm = G[bi].norm(dim=1)
                nb_ = G[bi] / gm.clamp(min=1e-300)[:, None]
                un = ((v - b.velocityAt(x)) * nb_).sum(1)
                drho = drho + 2.0 * rho * un * gm
        # density diffusion (fourtakas2019, fluid only)
        if cfg.ddt:
            xij = d
            tot = (rho[j] - rho[i]) + cfg.rho0 * (xij @ self.g) / cfg.c0 ** 2
            psi = -2.0 * tot[:, None] * xij / (r * (r + 1e-14 * H)).clamp(min=1e-300)[:, None]
            drho = drho + cfg.delta * H * cfg.c0 / XI * self._sum(torch.where(nz, V[j] * (psi * gW).sum(1), torch.zeros_like(r)), i)
        # pressure force
        acc = -self._sum((V[j] * (P[j] + s[i] * P[i]))[:, None] * gW, i) / rho[:, None]
        pp = P.clamp(min=0)
        forces = None
        if self.nb:
            Aeff = A - self._wall_excess(A, G, pp)[:, :, None] * G if cfg.clampWallPressure else A
            wall = (pp + s * P)[None, :, None] * G + Aeff                                            # [B,N,2]
            accw = -wall / rho[None, :, None]
            acc = acc + accw.sum(0)
            if want_forces:
                forces = -(self.m * accw).sum(1)                                                     # force of the fluid on each body
        # artificial viscosity (fluid only)
        if cfg.viscosity:
            vij = v[i] - v[j]
            mu = (vij * d).sum(1) / (r * r + 1e-14 * H * H)
            fac = cfg.alpha * cfg.c0 * H / XI
            acc = acc + fac * self._sum(torch.where(nz, V[j] / (0.5 * (rho[i] + rho[j])) * mu, torch.zeros_like(r))[:, None] * gW, i)
        acc = acc + self.g[None]
        return acc, drho, forces

    # ---------------------------------------------------------------------------------------------------------------- body-fitted packing
    def residual(self, x=None):
        """static wall consistency residual S_i = sum_j V_j grad_i W_ij + sum_b mu grad lambda_b: a uniform pressure exerts the force -2 P S_i / rho_i on particle i, so S_i = 0 is the condition for
        a particle layout that fits the wall (zero on the interior of a lattice and on a flat wall at dp/2; not zero where a regular lattice meets a smooth sloped wall)."""
        x = self.x if x is None else x
        i, j, r = neighbor_pairs(x, self.Hvec)
        nz = i != j
        d = x[i] - x[j]
        gW = torch.where(nz[:, None], dwendland2(r, self.H)[:, None] * d / r.clamp(min=1e-300)[:, None], torch.zeros_like(d))
        S = self._sum((self.m / self.rho)[j][:, None] * gW, i)
        if self.scene is not None:
            S = S + self._wall_data(x, self.rho)[1].sum(0)
        return S

    def pack(self, movable, iters=300, beta=0.05, maxStep=0.05, verbose=False):
        """relax the particles selected by `movable` (bool [N]) along -S_i until S_i = 0 (see `residual`), at most `maxStep * dx` per iteration; the others stay and still act as neighbours.
        This is the body-fitted layout a smooth analytic wall needs: a regular lattice carved by a sloped face leaves the first layer at irregular distances (0.4-1.8 dx) and rings the pool afterwards."""
        mv = torch.as_tensor(movable, device=self.dev)
        for it in range(iters):
            S = self.residual()
            step = -beta * self.dx ** 2 * S
            n = step.norm(dim=1, keepdim=True)
            step = torch.where(n > maxStep * self.dx, step * (maxStep * self.dx / n.clamp(min=1e-300)), step)
            self.x = self.x + torch.where(mv[:, None], step, torch.zeros_like(step))
            if verbose and it % 50 == 0:
                print(f"  pack {it}: max |S| of the movable {float(S[mv].norm(dim=1).max()):.2f}, mean {float(S[mv].norm(dim=1).mean()):.2f}", flush=True)
        return self

    def settle(self, T, gamma=30.0, verbose=False):
        """relax the initial layout against the exact walls by running with velocity damping v <- v exp(-gamma dt) for T seconds, then reset the clock: a regular lattice carved by a smooth sloped wall
        is not an equilibrium of the scheme (the first layer sits at irregular distances), the damped run finds the discrete equilibrium the wall forces actually define."""
        t0 = self.time
        k = 0
        while self.time - t0 < T:
            self.step()
            self.v = self.v * math.exp(-gamma * self.dt)
            k += 1
            if verbose and k % 500 == 0:
                print(f"  settle t={self.time - t0:.3f} KE={self.kinetic():.2e}", flush=True)
        self.time = t0
        return self

    # ---------------------------------------------------------------------------------------------------------------- time stepping
    def _next_dt(self, acc):
        cfg = self.cfg
        nu = cfg.alpha * cfg.c0 * self.H / (2 * 4)
        dtv = 0.125 * self.H ** 2 / nu / KSCALE if cfg.viscosity and cfg.alpha > 0 else cfg.maxDt
        dtc = cfg.cfl * self.H / cfg.c0 / KSCALE
        dta = 0.25 * math.sqrt(self.H / (float(acc.norm(dim=1).max()) + 1e-7)) / KSCALE
        new = min(dtv, dta, dtc, cfg.maxDt)
        new = max(new, cfg.minDt)
        if new > self.dt:
            new = min(new, cfg.growth * self.dt)
        return new

    def step(self):
        dt = self.dt
        a0, d0, _ = self.rhs(self.x, self.v, self.rho)
        xh, vh, rh = self.x + 0.5 * dt * self.v, self.v + 0.5 * dt * a0, self.rho + 0.5 * dt * d0
        if self.scene is not None:
            for b in self.scene.bodies:
                b.move(0.5 * dt)
        a1, d1, forces = self.rhs(xh, vh, rh, want_forces=True)
        vn = self.v + dt * a1
        self.x = self.x + 0.5 * dt * (self.v + vn)
        self.v = vn
        self.rho = self.rho + dt * d1
        if self.scene is not None:
            for b in self.scene.bodies:
                b.move(0.5 * dt)
        if forces is not None:
            self.wallForce = forces
        self.time += dt
        self.dt = self._next_dt(a1)
        return self.time

    # ---------------------------------------------------------------------------------------------------------------- diagnostics
    def pressure(self):
        return self.cfg.c0 ** 2 * (self.rho - self.cfg.rho0)

    def kinetic(self):
        return float(0.5 * self.m * (self.v * self.v).sum())


# ---------------------------------------------------------------------------------------------------------------------------- cases
def hydrostatic_tank(dp=0.02, L=2.4, Htank=1.2, Hwater=0.5, c0Ratio=20.0, domain="surface", device="cuda:0", cfg: Optional[DeltaSPHConfig] = None):
    """English et al. 2022 s.4.1 still water: tank L x Htank (inner faces), water of depth Hwater on the hydrostatic density profile, particles at the lattice mid-points (first row dp/2 from every wall,
    as warpSPH's `alignBoundaryLattice`), h/dp = 2 (support 4 dp), c0 = c0Ratio sqrt(g Hwater)."""
    from .dfsph2d import domain_scene
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
    from .dfsph2d import domain_scene
    from .scene import Body, Scene, SurfaceRep
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
