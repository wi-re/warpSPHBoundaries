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

from .cone_area import cone_area_scene
from .cover import cover_vector_scene
from .dfsph2d import F64, dwendland2, neighbor_pairs, wendland2
from .scene import BodyField, Scene, buildCellList, sceneOperation  # noqa: F401  (buildCellList re-exported for callers)
from .tensile import tensile_vector_scene

XI = 2.8213846683502197                 # warpSPHCore sphKernel_xi(Wendland2, 2D) = packing * kernelScale
KSCALE = 1.897367                       # warpSPHCore sphKernelScale(Wendland2, 2D): support / smoothing length


def wendland4(r, h):
    """Wendland C4 in 2D, support radius h: 9/(pi h^2) (1 - q)^6 (1 + 6 q + 35 q^2 / 3)."""
    q = r / h
    return torch.where(q < 1, 9.0 / (math.pi * h * h) * (1 - q) ** 6 * (1 + 6 * q + 35.0 / 3.0 * q * q), torch.zeros_like(q))


def dwendland4(r, h):
    q = r / h
    return torch.where(q < 1, -9.0 / (math.pi * h ** 3) * (56.0 / 3.0) * q * (1 + 5 * q) * (1 - q) ** 5, torch.zeros_like(q))


# kernel -> (W, dW/dr, xi = warpSPHCore sphKernel_xi, kernel scale = support / smoothing length)
KERNELS = {KernelFunctions.Wendland2: (wendland2, dwendland2, 2.8213846683502197, 1.897367),
           KernelFunctions.Wendland4: (wendland4, dwendland4, 3.56734561920166, 2.171239)}


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
    dilateSurface: bool = True          # the Antuono switch uses the free-surface mask dilated once (any neighbour within the support flagged), as warpSPH's `detectFreeSurface` returns it
    shifting: bool = False              # delta+ particle shifting (Sun et al. 2017 Eq. 7, surface treatment of Sun et al. 2019 Eq. 14, 20, 21) once per step after the integration
    shiftR: float = 0.2                 # tensile-control R of Eq. 7 (warpSPH `sun2017Eq7Shift`: R = 0.2, prefactor 16 h^2)
    shiftCFL: float = 0.3
    shiftThreshold: float = 0.5         # per-component clamp (x dx)
    shiftCapFraction: float = 0.5       # |shift| <= fraction * Umax * dt  (Sun 2019 Eq. 14)
    shiftLambda: float = 0.4            # lambda gate in the surface set F
    shiftCurvatureAngle: float = 15.0   # degrees
    noPen: str = "off"                  # 'impulse': once per step, closing particles within dp/4 of a wall get v_n <- v_n (1 - f(d)), f = 3 - 4 clip(1/2 + d/dp, 1/4, 1) (warpSPH mDBC no-penetration, 'impulse' placement)
    wallViscosity: bool = True          # wall term of the alpha-viscosity (free-slip mirror): (alpha c0 H/xi) 2 u_n (M2 n) mu / rho,  M2 = int W'(r) dr int yhat (x) yhat 1[solid] dphi (polar quadrature)
    timeCentred: bool = False           # warpSPH `timeCentredContinuity`: the kinematic part of drho/dt is advanced with the mean velocity (v^n + v^{n+1})/2 at the half-step positions
    wallContinuity: bool = True         # free-slip mirror term in the continuity equation (ablation switch)
    barecascoThreshold: float = math.pi / 3
    surfaceSamples: tuple = (24, 96)    # radial x angular samples of the solid around a particle (wall part of the free-surface detector)
    coverExact: bool = False            # wall part of the Barecasco cover vector from the exact edge reduction (cover.cover_vector_scene) instead of the polar quadrature; the cone count still uses the samples (Q3b)
    coneExact: bool = False               # wall part of the Barecasco cone count (and of the all-neighbour count) from the closed-form area (cone_area.cone_area_scene) instead of the polar samples
    tensileExact: bool = False            # wall part of the delta+ tensile term from the exact edge reduction (tensile.tensile_vector_scene, Wendland C2 and C4) instead of the polar quadrature
    kernel: KernelFunctions = KernelFunctions.Wendland2
    fixedDt: float = 0.0                # > 0: constant time step (the sloshing case pins dt = 1e-4)


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
        self.W, self.dW, self.xi, self.ks = KERNELS[self.cfg.kernel]
        self.dt = self.cfg.fixedDt if self.cfg.fixedDt else self.cfg.cfl * self.H / (self.cfg.c0 * self.ks)
        self.gravityFn = None                                  # t -> (gx, gy): a rolling tank as rotating gravity (SPHERIC test case 10), evaluated after every step
        self.wallForce = torch.zeros((self.nb, 2), dtype=F64, device=device)      # force of the fluid on every body (pressure part), last RHS call
        self.surface = torch.zeros(n, dtype=torch.bool, device=device)
        self.surfaceDilated = self.surface
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
    def _solid_samples(self, x, near):
        """the solid around the particles `near` on a polar grid: (inside [B,Q,R,P], unit directions u [P,2], radii rk [R], radial step dr, angular step dphi)."""
        nr, nphi = self.cfg.surfaceSamples
        dr, dphi = self.H / nr, 2 * math.pi / nphi
        rk = (torch.arange(nr, dtype=F64, device=self.dev) + 0.5) * dr
        ph = (torch.arange(nphi, dtype=F64, device=self.dev) + 0.5) * dphi
        u = torch.stack([torch.cos(ph), torch.sin(ph)], 1)
        pts = (x[near][:, None, None, :] + rk[None, :, None, None] * u[None, None, :, :]).reshape(-1, 2)
        ins = torch.stack([self.scene.inside(pts, body=bi).reshape(len(near), nr, nphi) for bi in range(self.nb)])
        return ins, u, rk, dr, dphi

    def _detect_surface(self, x, i, j, r, lam, samples=None):
        """warpSPH `detectFreeSurfaceBarecasco`: C = sum unit(x_i - x_j) over neighbours (W > 0), c = C/|C|; surface iff no neighbour (j != i) has angle(x_j - x_i, c) <= threshold/2
        (all neighbours count when C = 0).  Wall: the continuum of wall particles (density mu/dx^2), the solid sampled on a polar grid with `Scene.inside` (`samples` from `_solid_samples`)."""
        cfg, N = self.cfg, len(x)
        nz = i != j
        ii, jj, rr = i[nz], j[nz], r[nz]
        d = x[ii] - x[jj]
        unit = d / rr.clamp(min=1e-300)[:, None]
        C = self._sum(unit, ii)
        nw = cfg.wallMass / self.dx ** 2
        sample = None
        if samples is not None:
            near, (ins, u, rk, dr, dphi) = samples
            area = (rk * dr * dphi)[:, None]                                                        # [R,1]
            wt = ins.any(0).to(F64) * area[None]                                                    # solid area per sample [Q,R,P]
            if cfg.coverExact:  Cw = nw * cover_vector_scene(self.scene, x[near], self.H)        # grad int K = int unit(x - x'), no minus
            else:               Cw = -nw * (wt[..., None] * u[None, None]).sum((1, 2))           # sum unit(x_i - p) = -u
            C = C.index_add(0, near, Cw)
            sample = (near, u, wt)
        norm = C.norm(dim=1)
        c = C / norm.clamp(min=1e-300)[:, None]
        cosang = -((unit) * c[ii]).sum(1)                                                            # -n_ij . c
        inCone = torch.acos(cosang.clamp(-1.0, 1.0)) <= cfg.barecascoThreshold / 2
        count = self._sum(inCone.to(F64), ii)
        if sample is not None:
            near, u, wt = sample
            if cfg.coneExact:
                count = count.index_add(0, near, nw * cone_area_scene(self.scene, x[near], c[near], cfg.barecascoThreshold / 2, self.H))
            else:
                cn = (u[None] * c[near][:, None, :]).sum(2)                                          # [Q,P]
                cone = (torch.acos(cn.clamp(-1.0, 1.0)) <= cfg.barecascoThreshold / 2).to(F64)
                count = count.index_add(0, near, nw * (wt * cone[:, None, :]).sum((1, 2)))
        allcount = self._sum(torch.ones_like(rr), ii)
        if sample is not None:                                                                       # C = 0: all neighbours count (wall particles too)
            near, u, wt = sample
            if cfg.coneExact:
                allcount = allcount.index_add(0, near, nw * cone_area_scene(self.scene, x[near], c[near], math.pi, self.H))
            else:
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
        gW = torch.where(nz[:, None], self.dW(r, H)[:, None] * d / r.clamp(min=1e-300)[:, None], torch.zeros_like(d))      # grad_i W_ij
        P = cfg.c0 ** 2 * (rho - cfg.rho0)
        if self.scene is not None:
            lam, G, A = self._wall_data(x, rho)
        else:
            lam = torch.zeros((0, len(x)), dtype=F64, device=self.dev)
            G = torch.zeros((0, len(x), 2), dtype=F64, device=self.dev)
            A = G
        samples = None
        if self.scene is not None:
            near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
            if len(near):
                samples = (near, self._solid_samples(x, near))
        self.surface = self._detect_surface(x, i, j, r, lam, samples)
        if cfg.dilateSurface:
            self.surfaceDilated = self._sum(self.surface[j].to(F64), i) > 0.5            # pairs include i = j
        else:
            self.surfaceDilated = self.surface
        sw = (P >= 0) | self.surfaceDilated
        s = torch.where(sw, torch.ones_like(P), -torch.ones_like(P))

        # continuity: fluid pairs, wall (free-slip mirror); the kinematic rate is a function of the velocity (time-centred continuity re-evaluates it with the mean velocity)
        def kinematic(vel):
            out = -rho * self._sum(V[j] * ((vel[j] - vel[i]) * gW).sum(1), i)
            if self.nb and cfg.wallContinuity:
                for bi, b in enumerate(self.scene.bodies):
                    gm = G[bi].norm(dim=1)
                    nb_ = G[bi] / gm.clamp(min=1e-300)[:, None]
                    out = out + 2.0 * rho * ((vel - b.velocityAt(x)) * nb_).sum(1) * gm
            return out
        self._kinematic = kinematic
        kin = kinematic(v)
        drho = kin
        # density diffusion (fourtakas2019, fluid only)
        if cfg.ddt:
            xij = d
            tot = (rho[j] - rho[i]) + cfg.rho0 * (xij @ self.g) / cfg.c0 ** 2
            psi = -2.0 * tot[:, None] * xij / (r * (r + 1e-14 * H)).clamp(min=1e-300)[:, None]
            drho = drho + cfg.delta * H * cfg.c0 / self.xi * self._sum(torch.where(nz, V[j] * (psi * gW).sum(1), torch.zeros_like(r)), i)
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
            fac = cfg.alpha * cfg.c0 * H / self.xi
            acc = acc + fac * self._sum(torch.where(nz, V[j] / (0.5 * (rho[i] + rho[j])) * mu, torch.zeros_like(r))[:, None] * gW, i)
        if cfg.viscosity and cfg.wallViscosity and samples is not None:
            near, (ins, u, rk, dr, dphi) = samples
            wprime = self.dW(rk, H) * dr                                                          # W'(r) dr  [R] (negative)
            fac = cfg.alpha * cfg.c0 * H / self.xi
            for bi, b in enumerate(self.scene.bodies):
                gm = G[bi].norm(dim=1)
                nb_ = G[bi] / gm.clamp(min=1e-300)[:, None]
                un = ((v - b.velocityAt(x)) * nb_).sum(1)
                yy = u[:, :, None] * u[:, None, :]                                                   # [P,2,2]
                M2 = torch.einsum("qrp,r,pab->qab", ins[bi].to(F64), wprime, yy) * dphi              # int W' dr int yhat (x) yhat 1[solid] dphi
                accv = (fac * cfg.wallMass * 2.0 * un[near] / rho[near])[:, None] * torch.einsum("qab,qb->qa", M2, nb_[near])
                acc = acc.index_add(0, near, accv)
        acc = acc + self.g[None]
        return acc, drho, forces

    # ---------------------------------------------------------------------------------------------------------------- particle shifting (delta+) and no-penetration
    def _surface_state(self, x, rho):
        """pair data, detector and renormalisation matrices at positions x: dict(i, j, r, d, gW, V, lam [B,N], G, samples, surface, F (dilated), Mf [N,2,2], Mt [N,2,2])."""
        H = self.H
        i, j, r = neighbor_pairs(x, self.Hvec)
        nz = i != j
        d = x[i] - x[j]
        gW = torch.where(nz[:, None], self.dW(r, H)[:, None] * d / r.clamp(min=1e-300)[:, None], torch.zeros_like(d))
        V = self.m / rho
        samples, lam, G, Mw = None, torch.zeros((0, len(x)), dtype=F64, device=self.dev), None, None
        if self.scene is not None:
            ps = ParticleState(positions=x, supports=self.Hvec, masses=torch.full_like(rho, self.m), kinds=self.kinds, densities=rho)
            adj = self.scene.buildAdjacency(ps, self._props(WarpOperation.Density))
            lam = self._wall_op(ps, adj, WarpOperation.Density, BodyField(rho=1.0))
            G = self._wall_op(ps, adj, WarpOperation.Gradient, BodyField(torch.tensor(1.0, dtype=F64, device=self.dev)))
            Mw = self._wall_op(ps, adj, WarpOperation.Covariance, BodyField(torch.tensor(1.0, dtype=F64, device=self.dev))).reshape(self.nb, len(x), 2, 2)
            near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
            if len(near):
                samples = (near, self._solid_samples(x, near))
        surface = self._detect_surface(x, i, j, r, lam, samples)
        F = self._sum(surface[j].to(F64), i) > 0.5
        Mf = torch.zeros((len(x), 2, 2), dtype=F64, device=self.dev).index_add_(0, i, V[j][:, None, None] * (-d)[:, :, None] * gW[:, None, :])
        Mt = Mf + (Mw.sum(0) if Mw is not None else 0.0)
        return dict(i=i, j=j, r=r, d=d, gW=gW, V=V, lam=lam, G=G, samples=samples, surface=surface, F=F, Mf=Mf, Mt=Mt)

    def shift(self, dt):
        """delta+ particle shift (warpSPH `solveShifting`, projection 'surfaceNormal', one iteration): raw shift
        dr_i = -CFL Ma 16 h^2 sum_j  m_j / (2 (rho_i + rho_j)) [1 + R (W_ij / W(dx))^4] grad_i W_ij   (wall: the continuum of wall particles, W^4 term by polar quadrature of the solid),
        then in the dilated surface set F: a shift pointing into the surface keeps its tangential part (zero where a neighbour's normal differs by > 15 deg), every shift in F is zero where the fluid-only
        lambda_min of the renormalisation matrix is < 0.4, capped at 0.5 Umax dt and clamped to 0.5 dx per component.  Normals n = -grad(lambda_min)/|grad(lambda_min)|, lambda_min of the fluid + wall matrix."""
        cfg, x, v, rho, H = self.cfg, self.x, self.v, self.rho, self.H
        st = self._surface_state(x, rho)
        i, j, r, gW, V, F = st["i"], st["j"], st["r"], st["gW"], st["V"], st["F"]
        n_ = len(x)
        w0 = self.W(torch.tensor([self.dx], dtype=F64, device=self.dev), H)[0]
        Wij = self.W(r, H)
        coef = 0.5 * self.m / (rho[i] + rho[j]) * (1.0 + cfg.shiftR * (Wij / w0) ** 4)
        raw = self._sum(coef[:, None] * gW, i)
        if self.scene is not None:
            # wall particles: sum_b m_b/(2 (rho_i + rho_b)) [..] grad W = (rho0 / (4 rho_i)) mu int [1 + R (W/W0)^4] grad_i W dA   (rho_b ~ rho_i, wall particle density mu / dx^2 of mass m)
            #   mu int grad_i W dA = G exactly;  T = int W^4 grad_i W dA by polar quadrature of the solid: grad_i W = -W'(r) yhat, dA = r dr dphi
            wall = st["G"].sum(0)
            if st["samples"] is not None:
                near, (ins, u, rk, dr, dphi) = st["samples"]
                if cfg.tensileExact:
                    family = {KernelFunctions.Wendland2: "w2", KernelFunctions.Wendland4: "w4"}.get(cfg.kernel)
                    if family is None:  raise NotImplementedError("tensileExact: Wendland C2 and C4 only")
                    T = tensile_vector_scene(self.scene, x[near], H, family)
                else:
                    Fr = self.W(rk, H) ** 4 * self.dW(rk, H) * rk * dr                                # W^4 W' r dr  [R]
                    T = -torch.einsum("bqrp,r,pa->qa", ins.to(F64), Fr, u) * dphi
                wall = wall.index_add(0, near, cfg.wallMass * cfg.shiftR / w0 ** 4 * T)
            raw = raw + (cfg.rho0 / (4.0 * rho))[:, None] * wall
        vmax = float(v.norm(dim=1).max())
        Ma = vmax / cfg.c0
        Ma = Ma if Ma >= 1e-6 else 0.1
        hs = H / self.ks
        upd = raw * (-cfg.shiftCFL * Ma * 16.0 * hs ** 2)
        # ---- surface treatment
        ev_t = torch.linalg.eigvalsh(st["Mt"])
        lam_t = ev_t.abs().min(dim=1).values
        ev_f = torch.linalg.eigvalsh(st["Mf"])
        lam_f = ev_f.abs().min(dim=1).values
        L = torch.linalg.pinv(st["Mt"])
        dl = lam_t[j] - lam_t[i]
        gl_ = torch.einsum("nab,nb->na", L, self._sum((V[j] * dl)[:, None] * gW, i))
        nrm = -gl_ / gl_.norm(dim=1, keepdim=True).clamp(min=1e-300)
        nz = i != j
        both = F[i] & F[j] & nz
        dots = torch.where(both, (nrm[i] * nrm[j]).sum(1), torch.full_like(r, float("inf")))
        minDot = torch.full((n_,), float("inf"), dtype=F64, device=self.dev).scatter_reduce(0, i, dots, reduce="amin", include_self=False)
        kappa = (minDot >= math.cos(math.radians(cfg.shiftCurvatureAngle))).to(F64)
        outward = (upd * nrm).sum(1)
        tang = upd - outward[:, None] * nrm
        upd = torch.where((F & (outward >= 0))[:, None], kappa[:, None] * tang, upd)
        upd = torch.where(F[:, None], upd * (lam_f >= cfg.shiftLambda).to(F64)[:, None], upd)
        # ---- caps
        cap = cfg.shiftCapFraction * vmax * dt
        mag = upd.norm(dim=1, keepdim=True)
        if cap > 0:
            upd = upd * (cap / mag.clamp(min=1e-30)).clamp(max=1.0)
        upd = upd.clamp(-cfg.shiftThreshold * self.dx, cfg.shiftThreshold * self.dx)
        self.surface, self.surfaceDilated = st["surface"], F
        return upd

    def no_penetration(self):
        """warpSPH mDBC no-penetration, 'impulse' placement, for an analytic wall: a fluid particle at signed distance d < dp/4 from a wall that is closing on it (v_rel . n < 0, n into the fluid) gets
        v += -f vn n with f = 3 - 4 clip(1/2 + d/dp, 1/4, 1) (the ghost / boundary-particle geometry of a flat wall dp/2 inside the solid); f = 1 at the face (inelastic), 2 for a particle 1/4 dp inside (reflection)."""
        if self.scene is None or self.cfg.noPen != "impulse":
            return 0
        lam = self._wall_data(self.x, self.rho)[0]
        near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
        if not len(near):
            return 0
        d, n, hit = self.scene.signed_distance(self.x[near])
        wallv = torch.zeros(len(near), 2, dtype=F64, device=self.dev)                              # static walls (moving bodies: velocityAt of the nearest body, not needed yet)
        vn = ((self.v[near] - wallv) * n).sum(1)
        f = 3.0 - 4.0 * (0.5 + d / self.dx).clamp(0.25, 1.0)
        act = hit & (d < 0.25 * self.dx) & (vn < 0)
        corr = torch.where(act[:, None], (-f * vn)[:, None] * n, torch.zeros_like(n))
        self.v = self.v.index_add(0, near, corr)
        return int(act.sum())

    # ---------------------------------------------------------------------------------------------------------------- body-fitted packing
    def residual(self, x=None):
        """static wall consistency residual S_i = sum_j V_j grad_i W_ij + sum_b mu grad lambda_b: a uniform pressure exerts the force -2 P S_i / rho_i on particle i, so S_i = 0 is the condition for
        a particle layout that fits the wall (zero on the interior of a lattice and on a flat wall at dp/2; not zero where a regular lattice meets a smooth sloped wall)."""
        x = self.x if x is None else x
        i, j, r = neighbor_pairs(x, self.Hvec)
        nz = i != j
        d = x[i] - x[j]
        gW = torch.where(nz[:, None], self.dW(r, self.H)[:, None] * d / r.clamp(min=1e-300)[:, None], torch.zeros_like(d))
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
        if cfg.fixedDt:
            return cfg.fixedDt
        nu = cfg.alpha * cfg.c0 * self.H / (2 * 4)
        dtv = 0.125 * self.H ** 2 / nu / self.ks if cfg.viscosity and cfg.alpha > 0 else cfg.maxDt
        dtc = cfg.cfl * self.H / cfg.c0 / self.ks
        dta = 0.25 * math.sqrt(self.H / (float(acc.norm(dim=1).max()) + 1e-7)) / self.ks
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
        rho_new = self.rho + dt * d1
        if self.cfg.timeCentred:
            # warpSPHIntegrators `symplecticEuler` drift field: rho^{n+1} = rho^n + dt (k1 - kin(k1)) + dt K(x_h, vbar),  vbar = (v^n + v^{n+1}) / 2, K the kinematic rate (same positions and densities as k1)
            K = self._kinematic(0.5 * (self.v + vn))
            rho_new = rho_new + dt * (K - self._kinematic(vh))
        self.v = vn
        self.rho = rho_new
        if self.scene is not None:
            for b in self.scene.bodies:
                b.move(0.5 * dt)
        if self.cfg.shifting:
            self.x = self.x + self.shift(dt)
        self.nopen_count = self.no_penetration()
        if forces is not None:
            self.wallForce = forces
        self.time += dt
        if self.gravityFn is not None:
            self.g = torch.tensor(self.gravityFn(self.time), dtype=F64, device=self.dev)
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


# ---------------------------------------------------------------------------------------------------------------------------- Marrone 3.1 dam break
_GAUSS7_NODES = (-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585)
_GAUSS7_WEIGHTS = (0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694, 0.3818300505051189, 0.2797053914892766, 0.1294849661688697)
_DISC_CHORD_WEIGHTS = tuple(w * max(0.0, 1.0 - x * x) ** 0.5 for x, w in zip(_GAUSS7_NODES, _GAUSS7_WEIGHTS))


def marrone_dambreak(nx=67, c0Ratio=40.0, domain="surface", device="cuda:0", cfg: Optional[DeltaSPHConfig] = None, **cfgkw):
    """Marrone et al. 2011 s.3.1 as warpSPH's `dambreak` case builds it (`probe_deltaSPHMarrone.py`): column 2H x H (H = 0.6) in the left corner of a closed tank 5.366 H long, ceiling 0.985 m above the bed
    (L = 1 m, dx = L/nx, walls dx/2 outside the first lattice row on all sides), c0 = c0Ratio sqrt(g H), uniform rho0 at t = 0 (no hydrostatic init), time-centred continuity on.
    x-spacing of warpSPH's lattice is W/round(W/dx) (0.13 % smaller at nx = 67); here dx in both directions."""
    from .dfsph2d import domain_scene
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


def mls_pressure(sim, q, neighbor_threshold=4):
    """first-order MLS (Liu-Liu) fit of the fluid pressure at query points q [M,2] (warpSPH `_mlsPressureDevice`): weights V_j W, a + b.(x_j - q), Shepard fallback where the 3x3 system is ill-conditioned
    or has fewer than `neighbor_threshold` neighbours, clamped >= 0.  Returns (value [M], neighbours [M])."""
    x, H = sim.x, sim.H
    P = sim.pressure()
    V = sim.m / sim.rho
    M = len(q)
    d = x[None, :, :] - q[:, None, :]                                           # [M,N,2]
    r = d.norm(dim=2)
    inside = r < H
    w = torch.where(inside, V[None] * sim.W(r, H), torch.zeros_like(r))     # [M,N]
    nn = inside.sum(1)
    y = d / H
    B = torch.stack([torch.ones_like(r), y[..., 0], y[..., 1]], 2)              # [M,N,3]
    A = torch.einsum("mn,mni,mnj->mij", w, B, B)
    b = torch.einsum("mn,mni,n->mi", w, B, P)
    ev = torch.linalg.eigvalsh(A)
    wc = (nn >= neighbor_threshold) & (ev[:, 0] > 1e-6 * ev[:, 2].clamp(min=1e-300))
    sol = torch.linalg.solve(A + torch.where(wc, 0.0, 1.0)[:, None, None] * torch.eye(3, dtype=F64, device=x.device)[None], b[:, :, None])[:, :, 0]
    shep = torch.where(A[:, 0, 0] > 0, b[:, 0] / A[:, 0, 0].clamp(min=1e-300), torch.zeros_like(r[:, 0]))
    val = torch.where(wc, sol[:, 0], shep).clamp(min=0.0)
    return val, nn


def wall_probes(sim, info, heights=(0.16, 0.584, 1.0), disc=0.045, inset=0.0):
    """P* = P / (rho0 g H) of the three impact-wall probes (disc-averaged with the 7-point chord quadrature, samples with <= 1 neighbour dry) at the wall and one dx into the fluid."""
    xw = info["xr"] - inset
    out = []
    for xq in (xw, xw - info["dx"]):
        q = torch.tensor([[xq, info["yb"] + z + disc * s] for z in heights for s in _GAUSS7_NODES], dtype=F64, device=sim.dev)
        val, nn = mls_pressure(sim, q)
        val, nn = val.reshape(len(heights), 7), nn.reshape(len(heights), 7)
        w = torch.tensor(_DISC_CHORD_WEIGHTS, dtype=F64, device=sim.dev)[None] * (nn > 1).to(F64)
        ws = w.sum(1)
        out.append(torch.where(ws > 0, (val * w).sum(1) / ws.clamp(min=1e-12), torch.zeros_like(ws)) / (sim.cfg.rho0 * info["g"] * info["H"]))
    return out[0].cpu().numpy(), out[1].cpu().numpy()


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
    from .dfsph2d import domain_scene
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


def sloshing_probes(sim, info, radius=0.02):
    """Sensor-1 pressure in Pa: (a) warpSPH's `sensorPressureProbe`: Gaussian Shepard average (exp(-(r / (radius / 2))^2), fluid particles within `radius`) of the Tait pressure rho0 c0^2 / 7 ((rho / rho0)^7 - 1) x rho0Phys;
    (b) first-order MLS of the (linear) EOS pressure at the wall point of the sensor.  NaN where fewer than 3 neighbours."""
    c0, rho0, rp = sim.cfg.c0, sim.cfg.rho0, info["rho0Phys"]
    q = torch.tensor(info["sensor"], dtype=F64, device=sim.dev)
    r = (sim.x - q[None]).norm(dim=1)
    near = r < radius
    if int(near.sum()) >= 3:
        w = torch.exp(-(r[near] / (0.5 * radius)) ** 2)
        tait = rp * rho0 * c0 ** 2 / 7.0 * ((sim.rho[near] / rho0) ** 7 - 1.0)
        pg = float((w * tait).sum() / w.sum())
    else:
        pg = float("nan")
    val, nn = mls_pressure(sim, q[None])
    return pg, float(val[0]) * rp
