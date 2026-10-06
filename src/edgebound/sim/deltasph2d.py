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
import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

import torch

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation

from ..scene.cone_area import cone_area_scene
from ..scene.fixedadj import fixed_adjacency
from .fluidwarp import FluidWarp
from ..scene.boundaryOps import kernelName
from ..scene.cover import cover_vector_scene
from ..scene.fused import FusedWall, WallOutput
from ..edge.warpfused import FusedGroup
from ..scene.scene import BodyField, Scene, sceneOperation
from ..scene.tensile import tensile_factor, tensile_vector_scene
from ..scene.viscosity import lap_factor, lap_lambda_scene
from .pairs import F64, dwendland2, dwendland4, neighbor_pairs, wendland2, wendland4

XI = 2.8213846683502197                 # warpSPHCore sphKernel_xi(Wendland2, 2D) = packing * kernelScale
KSCALE = 1.897367                       # warpSPHCore sphKernelScale(Wendland2, 2D): support / smoothing length


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
    wallViscosity: bool = True          # wall term of the alpha-viscosity (free-slip mirror), form: wallViscosityForm
    wallViscosityForm: str = "laplacian"   # "laplacian": exact wall Laplacian (viscosity.lap_lambda_scene, free-slip mirror, nu_eff = alpha c0 H/(8 xi), Wendland C2 and C4) | "pairwise": the warpSPH pairwise (Monaghan) form with the free-slip mirror, polar quadrature of the solid (cfg.surfaceSamples) | "noslip": no-slip wall (v_w = body velocity): Chiron-style flux term -2 nu_eff (v - v_w) |G| / (rho d), nu_eff = alpha c0 H/(8 xi), d = max(distance to the wall, 0.25 dx)
    fusedWall: bool = True              # step 3 of docs/plan-wall-evaluation.md: lam, G, Cov, A, the cover vector, the wall Laplacian and the tensile term of one position set from ONE fused kernel launch family (scene/fused.py); surface-loop walls with one support only, else the sceneOperation path is used
    fixedAdjacency: bool = True         # step 5 of the plan: the wall adjacency of the fused path in ONE Warp launch per (body, rep) with fixed shapes and no host sync (scene/fixedadj.py); False = Scene.adjacency (torch.nonzero / cell-list pair search), the oracle
    graphStep: bool = True              # step 5b of the plan: replay the whole step as a CUDA graph (graphstep.py; needs fluidWarp, fusedWall, static surface-loop walls, else eager)
    fluidWarp: bool = True              # phase 3 of the plan: continuity, density diffusion, Antuono pressure force and the alpha viscosity of the fluid pairs from the warpSPH modules on a warpSPHCore Verlet adjacency (sim/fluidwarp.py); False = the torch pair sums, the oracle
    timeCentred: bool = False           # warpSPH `timeCentredContinuity`: the kinematic part of drho/dt is advanced with the mean velocity (v^n + v^{n+1})/2 at the half-step positions
    wallContinuity: bool = True         # free-slip mirror term in the continuity equation (ablation switch)
    barecascoThreshold: float = math.pi / 3
    surfaceSamples: tuple = (24, 96)    # radial x angular samples of the solid around a particle: used ONLY by wallViscosityForm = "pairwise"
    kernel: KernelFunctions = KernelFunctions.Wendland2
    fixedDt: float = 0.0                # > 0: constant time step (the sloshing case pins dt = 1e-4)


def _sym2_lam_pinv(M, pinv=True):
    """(min |eigenvalue| [N], pseudo-inverse [N,2,2]) of 2 x 2 matrices M [N,2,2] that are symmetric to round-off, in closed form (the batched LAPACK-style `eigvalsh` / `pinv` of tiny matrices were 1.1 ms of GPU
    time per step, 8 % of the dam break).  `eigvalsh`'s convention is kept: the lower triangle defines the matrix.  The pseudo-inverse drops eigenvalues below 2 eps max|lambda| (torch's default rtol)."""
    a, b, d = M[:, 0, 0], M[:, 1, 0], M[:, 1, 1]
    mid, disc = 0.5 * (a + d), torch.sqrt((0.5 * (a - d)) ** 2 + b * b)
    l1, l2 = mid + disc, mid - disc
    lam = torch.minimum(l1.abs(), l2.abs())
    if not pinv:
        return lam, None
    # unit eigenvector of l1: the larger of (b, l1 - a) and (l1 - d, b); isotropic matrices (disc = 0) take (1, 0)
    ua, ub = torch.stack([b, l1 - a], 1), torch.stack([l1 - d, b], 1)
    u = torch.where((ua * ua).sum(1, keepdim=True) >= (ub * ub).sum(1, keepdim=True), ua, ub)
    nrm = u.norm(dim=1, keepdim=True)
    ex = torch.zeros_like(u); ex[:, 0] = 1.0
    u = torch.where(nrm > 1e-300, u / nrm.clamp(min=1e-300), ex)
    w = torch.stack([-u[:, 1], u[:, 0]], 1)
    cut = 2.0 * torch.finfo(M.dtype).eps * torch.maximum(l1.abs(), l2.abs())
    c1 = torch.where(l1.abs() > cut, 1.0 / torch.where(l1 == 0, torch.ones_like(l1), l1), torch.zeros_like(l1))
    c2 = torch.where(l2.abs() > cut, 1.0 / torch.where(l2 == 0, torch.ones_like(l2), l2), torch.zeros_like(l2))
    L = c1[:, None, None] * u[:, :, None] * u[:, None, :] + c2[:, None, None] * w[:, :, None] * w[:, None, :]
    return lam, L


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
        self.dt_t = torch.tensor(float(self.dt), dtype=F64, device=device)     # the time step on the device (the host copy `dt` follows after every step)
        self.gravityFn = None                                  # t -> (gx, gy): a rolling tank as rotating gravity (SPHERIC test case 10), evaluated after every step
        self.wallForce = torch.zeros((self.nb, 2), dtype=F64, device=device)      # force of the fluid on every body (pressure part), last RHS call
        self.surface = torch.zeros(n, dtype=torch.bool, device=device)
        self.surfaceDilated = self.surface
        self.history = []
        self._w0 = float(self.W(torch.tensor([self.dx], dtype=F64, device=device), self.H)[0])          # W(dx): the tensile reference of the shift
        self._fluidwarp = None                                 # FluidWarp of cfg.fluidWarp (lazy)
        self._graphCfg = None
        self._graphed = None                                   # GraphedStep (or False when the configuration does not allow it), lazy
        self._graphMode = False                                # True while the step is captured / replayed as a CUDA graph: no host-side caches or reads (graphstep.py)
        self._hc = None                                        # (Hvec tensor, all supports == H) cache of _constSupport
        self._fgroups = None                                   # (FusedGroup tuple, {output role: group index}, config key) of the fused wall evaluation
        self._wallCache = None                                 # (x, body poses, adjacency, pair moments, lam, G) of the last _wall_data call: the next step's first RHS is evaluated at the positions the no-penetration law just used

    # ---------------------------------------------------------------------------------------------------------------- helpers
    def _sum(self, vals, i):
        out = torch.zeros((len(self.x),) + vals.shape[1:], dtype=vals.dtype, device=self.dev)
        return out.index_add_(0, i, vals)

    def _props(self, op, mode=GradientScheme.Naive):
        return OperationProperties(kernel=self.cfg.kernel, operation=op, gradientMode=mode, operationMode=OperationDirection.BoundaryToFluid)

    def _wall_op(self, ps, pm, op, flds, mode=GradientScheme.Naive):
        flds = flds if isinstance(flds, list) else [flds] * self.nb
        return self.cfg.wallMass * sceneOperation(ps, self._props(op, mode), self.scene, pm, None, flds, perBody=True)

    def _wall_data(self, x, rho):
        """per-body wall integrals at positions x: lam [B,N], G = mu grad lambda [B,N,2], A = mu int (a1.y) grad W [B,N,2] with a1 = rho0 (g - a_wall)."""
        return self._wall_state(x, rho)[:3]

    def _wall_state(self, x, rho):
        """`_wall_data` and the wall adjacency at x (the near-wall consumers -- cover, Laplacian, tensile -- take its `restrict` instead of searching again)."""
        ps = ParticleState(positions=x, supports=self.Hvec, masses=torch.full_like(rho, self.m), kinds=self.kinds, densities=rho)
        poses = [(b.center.clone(), float(b.angle)) for b in self.scene.bodies]
        c = self._wallCache
        fused = self.cfg.fusedWall and FusedWall.supported(self.scene)
        if not self._graphMode and c is not None and c[0].shape == x.shape and len(c[1]) == len(poses) and torch.equal(c[0], x) and torch.equal(c[6], self.Hvec) and torch.equal(c[7], self.kinds) and all(float(p[1]) == q[1] and torch.equal(p[0], q[0]) for p, q in zip(c[1], poses)) and isinstance(c[2], FusedWall) == fused and (not fused or c[2].key == self._fused_key()):
            adj, pm, lam, G = c[2], c[3], c[4], c[5]            # same positions, same poses: lam and G do not depend on the densities or the gravity
        elif fused and self._constSupport():
            adj = self._fused_state(ps)
            pm, lam, G = None, self.cfg.wallMass * adj.out["lam"], self.cfg.wallMass * adj.out["G"]
            self._wallCache = None if self._graphMode else (x.clone(), poses, adj, pm, lam, G, self.Hvec.clone(), self.kinds.clone())
        else:
            adj = self.scene.adjacency(ps, self._props(WarpOperation.Density))
            pm = self.scene.precompute(adj, self._props(WarpOperation.Density))
            lam = self._wall_op(ps, pm, WarpOperation.Density, BodyField(rho=1.0))
            G = self._wall_op(ps, pm, WarpOperation.Gradient, BodyField(torch.tensor(1.0, dtype=F64, device=self.dev)))
            self._wallCache = None if self._graphMode else (x.clone(), poses, adj, pm, lam, G, self.Hvec.clone(), self.kinds.clone())
        if isinstance(adj, FusedWall):
            a1 = torch.stack([self.cfg.rho0 * (self.g[None] - b.accelerationAt(x)) for b in self.scene.bodies])
            return lam, G, self.cfg.wallMass * adj.evaluate((WallOutput("A", 0, "a1g1"),), a1=a1)["A"], adj
        flds = []
        for b in self.scene.bodies:
            a1 = self.cfg.rho0 * (self.g[None] - b.accelerationAt(x))
            flds.append(BodyField(torch.zeros(len(x), dtype=F64, device=self.dev), a1, rho=1.0, perQuery=True))
        A = self._wall_op(ps, pm, WarpOperation.Gradient, flds)
        return lam, G, A, adj

    def _constSupport(self):
        """every support equals H (the fused wall path serves one support only); the device comparison is made once per `Hvec` tensor object, not per call (a host sync)."""
        if self._hc is None or self._hc[0] is not self.Hvec:
            self._hc = (self.Hvec, bool((self.Hvec == self.H).all()))
        return self._hc[1]

    def _fused_key(self):
        return (self.cfg.kernel, bool(self.cfg.viscosity and self.cfg.wallViscosity and self.cfg.wallViscosityForm == "laplacian"), bool(self.cfg.fixedAdjacency))

    def _fused_state(self, ps):
        """the fused wall evaluation at the positions of `ps`: adjacency, one stage-1 launch family for the kernels the step needs, and every static output of this position set in `.out`
        (lam, G, Cov of the kernel, `cover` = g0 of the degree-1 cone kernel per body, `lap` = sum(2 lam - tr g1) of lw, `tens` = g0 of wp5, per body, raw: factors applied by the consumers)."""
        key = self._fused_key()
        if self._fgroups is None or self._fgroups[2] != key:          # the layout follows the config flags (tests and sweeps change them on a live solver)
            name = kernelName(self.cfg.kernel)
            groups, idx = [FusedGroup(name), FusedGroup("cone", (3, 4))], {"w": 0, "cone": 1}
            if self.cfg.viscosity and self.cfg.wallViscosity and self.cfg.wallViscosityForm == "laplacian":
                fam = self._family(); lap_factor(1.0, fam)                       # registers the kernel
                idx["lap"] = len(groups); groups.append(FusedGroup("l" + fam))
            fam = self._family(); tensile_factor(1.0, fam)                       # `shift` may be called whatever cfg.shifting says: the tensile group (4 terms) is always there
            idx["tens"] = len(groups); groups.append(FusedGroup(fam + "p5", (3, 4)))
            self._fgroups = (tuple(groups), idx, key)
        groups, idx, _ = self._fgroups
        adj = fixed_adjacency(self.scene, ps, self._props(WarpOperation.Density), self.H) if self.cfg.fixedAdjacency else self.scene.adjacency(ps, self._props(WarpOperation.Density))
        fw = FusedWall(self.scene, adj, groups)
        outs = [WallOutput("lam", 0, "lam"), WallOutput("G", 0, "g0"), WallOutput("Cov", 0, "cov"), WallOutput("cover", idx["cone"], "g0")]
        if "lap" in idx: outs.append(WallOutput("lap", idx["lap"], "lap"))
        if "tens" in idx: outs.append(WallOutput("tens", idx["tens"], "g0"))
        fw.out = fw.evaluate(outs)
        fw.key = key
        return fw

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

    def _family(self):
        """Wendland family name of the exact wall operations (tensile, wall Laplacian): C2 and C4 only."""
        fam = {KernelFunctions.Wendland2: "w2", KernelFunctions.Wendland4: "w4"}.get(self.cfg.kernel)
        if fam is None:  raise NotImplementedError("exact wall operations: Wendland C2 and C4 only")
        return fam

    def _detect_surface(self, x, i, j, r, lam, adj=None, fk=None):
        """warpSPH `detectFreeSurfaceBarecasco`: C = sum unit(x_i - x_j) over neighbours (W > 0), c = C/|C|; surface iff no neighbour (j != i) has angle(x_j - x_i, c) <= threshold/2
        (all neighbours count when C = 0).  Wall: the continuum of wall particles (density mu/dx^2), the exact edge reduction (cover) and closed-form area (cone).  No scene: lam of shape [0, N], near empty, nothing added.
        `adj`: the wall adjacency at x (the cover vector takes its restriction to the near-wall particles).  `fk`: the `FluidKernels` of the fluid adjacency (cfg.fluidWarp; then i, j, r are unused)."""
        cfg = self.cfg
        nw = cfg.wallMass / self.dx ** 2
        if fk is not None:
            p1 = fk.pass1()
            C, allcount = p1["C"], p1["nAll"]
        else:
            nz = i != j
            ii, jj, rr = i[nz], j[nz], r[nz]
            d = x[ii] - x[jj]
            unit = d / rr.clamp(min=1e-300)[:, None]
            C = self._sum(unit, ii)
        fused = isinstance(adj, FusedWall)
        if fused:                                                                                    # full-length outputs, masked to the near-wall particles (no host sync)
            nearm = (lam.sum(0) > 1e-9).to(F64)
            C = C - nw * (math.pi * self.H ** 3 / 3) * adj.out["cover"].sum(0) * nearm[:, None]      # grad int K over the solid = -(pi H^3 / 3) g0 of the cone kernel (cover.cover_vector_scene); grad int K = int unit(x - x'), no minus
        else:
            near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
            if len(near):
                cover = cover_vector_scene(self.scene, x[near], self.H, None if adj is None else adj.restrict(near))
                C = C.index_add(0, near, nw * cover)
        norm = C.norm(dim=1)
        c = C / norm.clamp(min=1e-300)[:, None]
        if fk is not None:
            count = fk.cone_count(c, cfg.barecascoThreshold / 2)
        else:
            cosang = -((unit) * c[ii]).sum(1)                                                        # -n_ij . c
            inCone = torch.acos(cosang.clamp(-1.0, 1.0)) <= cfg.barecascoThreshold / 2
            count = self._sum(inCone.to(F64), ii)
            allcount = self._sum(torch.ones_like(rr), ii)                                            # C = 0: all neighbours count (wall particles too)
        if fused:
            areas = adj.cone_area(c, cfg.barecascoThreshold / 2) * nearm                             # [2, N]: cone, full disk (the wall continuum)
            count, allcount = count + nw * areas[0], allcount + nw * areas[1]
        elif len(near):
            count = count.index_add(0, near, nw * cone_area_scene(self.scene, x[near], c[near], cfg.barecascoThreshold / 2, self.H))
            allcount = allcount.index_add(0, near, nw * cone_area_scene(self.scene, x[near], c[near], math.pi, self.H))
        count = torch.where(norm > 1e-12, count, allcount)
        return count < 0.5

    # ---------------------------------------------------------------------------------------------------------------- right-hand side
    def rhs(self, x, v, rho, want_forces=False):
        cfg = self.cfg
        H, V = self.H, self.m / rho
        fw = None
        if cfg.fluidWarp:
            if self._fluidwarp is None:
                self._fluidwarp = FluidWarp(self)
            fw = self._fluidwarp
            fps, fadj = fw.state(x, v, rho)
            fk = fw.kernels(fadj, x, rho)
        else:
            i, j, r = neighbor_pairs(x, self.Hvec)
            nz = i != j
            d = x[i] - x[j]
            gW = torch.where(nz[:, None], self.dW(r, H)[:, None] * d / r.clamp(min=1e-300)[:, None], torch.zeros_like(d))      # grad_i W_ij
            fk = None
        P = cfg.c0 ** 2 * (rho - cfg.rho0)
        adj = None
        if self.scene is not None:
            lam, G, A, adj = self._wall_state(x, rho)
        else:
            lam = torch.zeros((0, len(x)), dtype=F64, device=self.dev)
            G = torch.zeros((0, len(x), 2), dtype=F64, device=self.dev)
            A = G
        near = torch.zeros(0, dtype=torch.long, device=self.dev)
        if self.scene is not None and (not isinstance(adj, FusedWall) or not self._graphMode):         # the index list is only needed by the non-fused and the 'pairwise' paths (and diagnostics); a graph capture has none
            near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
        if cfg.viscosity and cfg.wallViscosity and cfg.wallViscosityForm not in ("laplacian", "pairwise", "noslip"):
            raise ValueError("wallViscosityForm must be 'laplacian', 'pairwise' or 'noslip', got %r" % (cfg.wallViscosityForm,))
        self.surface = self._detect_surface(x, None if fk else i, None if fk else j, None if fk else r, lam, adj, fk)
        if cfg.dilateSurface:
            self.surfaceDilated = fk.dilate(self.surface) if fk else self._sum(self.surface[j].to(F64), i) > 0.5            # pairs include i = j
        else:
            self.surfaceDilated = self.surface
        sw = (P >= 0) | self.surfaceDilated
        s = torch.where(sw, torch.ones_like(P), -torch.ones_like(P))

        # continuity: fluid pairs, wall (free-slip mirror); the kinematic rate is a function of the velocity (time-centred continuity re-evaluates it with the mean velocity)
        def kinematic(vel):
            out = fw.continuity(fps, fadj, vel) if fw is not None else -rho * self._sum(V[j] * ((vel[j] - vel[i]) * gW).sum(1), i)
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
        if cfg.ddt and fw is not None:
            drho = drho + fw.density_diffusion(fps, fadj)
        elif cfg.ddt:
            xij = d
            tot = (rho[j] - rho[i]) + cfg.rho0 * (xij[:, 0] * self.g[0] + xij[:, 1] * self.g[1]) / cfg.c0 ** 2
            psi = -2.0 * tot[:, None] * xij / (r * (r + 1e-14 * H)).clamp(min=1e-300)[:, None]
            drho = drho + cfg.delta * H * cfg.c0 / self.xi * self._sum(torch.where(nz, V[j] * (psi * gW).sum(1), torch.zeros_like(r)), i)
        # pressure force
        acc = fw.pressure(fps, fadj, P, self.surfaceDilated) if fw is not None else -self._sum((V[j] * (P[j] + s[i] * P[i]))[:, None] * gW, i) / rho[:, None]
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
            if fw is not None:
                acc = acc + fw.viscosity(fps, fadj, v)
            else:
                vij = v[i] - v[j]
                mu = (vij * d).sum(1) / (r * r + 1e-14 * H * H)
                fac = cfg.alpha * cfg.c0 * H / self.xi
                acc = acc + fac * self._sum(torch.where(nz, V[j] / (0.5 * (rho[i] + rho[j])) * mu, torch.zeros_like(r))[:, None] * gW, i)
        if cfg.viscosity and cfg.wallViscosity and isinstance(adj, FusedWall) and cfg.wallViscosityForm in ("laplacian", "noslip"):
            fac = cfg.alpha * cfg.c0 * H / self.xi                                                  # full-length form: the near-wall rows are a mask, not an index list (no host sync)
            nearm = (lam.sum(0) > 1e-9).to(F64)
            for bi, b in enumerate(self.scene.bodies):
                gm = G[bi].norm(dim=1)
                nb_ = G[bi] / gm.clamp(min=1e-300)[:, None]
                if cfg.wallViscosityForm == "laplacian":
                    un = ((v - b.velocityAt(x)) * nb_).sum(1)
                    dl = lap_factor(H, self._family()) * adj.out["lap"][bi]
                    acc = acc + (-2.0 * (fac / 8.0) * cfg.wallMass * un / rho * dl * nearm)[:, None] * nb_
                else:
                    dsd, _, hit = self.scene.signed_distance(x, body=bi, supportMax=self.H)
                    dd = dsd.clamp(min=0.25 * self.dx)
                    acc = acc + torch.where((hit & (nearm > 0))[:, None], -2.0 * (fac / 8.0) * (v - b.velocityAt(x)) * (gm / (rho * dd))[:, None], torch.zeros_like(v))
        elif cfg.viscosity and cfg.wallViscosity and len(near):
            fac = cfg.alpha * cfg.c0 * H / self.xi
            if cfg.wallViscosityForm == "laplacian":
                dl = lap_factor(H, self._family()) * adj.out["lap"][:, near] if isinstance(adj, FusedWall) else lap_lambda_scene(self.scene, x[near], H, self._family(), adj.restrict(near))             # [B, Q]  int_solid lap W dA', per body
            elif cfg.wallViscosityForm == "pairwise":
                ins, u, rk, dr, dphi = self._solid_samples(x, near)                                         # the polar grid exists only for this form
                wprime = self.dW(rk, H) * dr                                                          # W'(r) dr  [R] (negative)
            elif cfg.wallViscosityForm == "noslip":
                pass                                                                          # per body below
            else:
                raise ValueError("wallViscosityForm must be 'laplacian', 'pairwise' or 'noslip', got %r" % (cfg.wallViscosityForm,))
            for bi, b in enumerate(self.scene.bodies):
                gm = G[bi].norm(dim=1)
                nb_ = G[bi] / gm.clamp(min=1e-300)[:, None]
                un = ((v - b.velocityAt(x)) * nb_).sum(1)
                if cfg.wallViscosityForm == "noslip":
                    d, _, hit = self.scene.signed_distance(x[near], body=bi)                           # d_signed > 0 in the fluid; hit False only for volume representations
                    dd = d.clamp(min=0.25 * self.dx)                                                   # the 1/d floor (the no-penetration law keeps particles at d >= ~0.25 dx)
                    acc = acc.index_add(0, near, torch.where(hit[:, None], -2.0 * (fac / 8.0) * (v - b.velocityAt(x))[near] * (gm[near] / (rho[near] * dd))[:, None], torch.zeros_like(v[near])))   # fac/8 = nu_eff; gm = |G_b|; all-components relative velocity (no-slip)
                    continue
                if cfg.wallViscosityForm == "laplacian":
                    acc = acc.index_add(0, near, (-2.0 * (fac / 8.0) * cfg.wallMass * un[near] / rho[near] * dl[bi])[:, None] * nb_[near])   # fac/8 = fac/(2(d+2)), d = 2: moment identity, viscosity.py
                else:
                    yy = u[:, :, None] * u[:, None, :]                                                   # [P,2,2]
                    M2 = torch.einsum("qrp,r,pab->qab", ins[bi].to(F64), wprime, yy) * dphi              # int W' dr int yhat (x) yhat 1[solid] dphi
                    accv = (fac * cfg.wallMass * 2.0 * un[near] / rho[near])[:, None] * torch.einsum("qab,qb->qa", M2, nb_[near])
                    acc = acc.index_add(0, near, accv)
        acc = acc + self.g[None]
        return acc, drho, forces

    # ---------------------------------------------------------------------------------------------------------------- particle shifting (delta+) and no-penetration
    def _surface_state(self, x, rho):
        """pair data, detector and renormalisation matrices at positions x: dict(i, j, r, d, gW, V, lam [B,N], G, adj (the wall adjacency), near, surface, F (dilated), Mf [N,2,2], Mt [N,2,2])."""
        H = self.H
        fk = None
        if self.cfg.fluidWarp:
            if self._fluidwarp is None:
                self._fluidwarp = FluidWarp(self)
            fk = self._fluidwarp.kernels(self._fluidwarp.state(x, self.v, rho)[1], x, rho)
            i = j = r = d = gW = None
        else:
            i, j, r = neighbor_pairs(x, self.Hvec)
            nz = i != j
            d = x[i] - x[j]
            gW = torch.where(nz[:, None], self.dW(r, H)[:, None] * d / r.clamp(min=1e-300)[:, None], torch.zeros_like(d))
        V = self.m / rho
        lam, G, Mw, adj = torch.zeros((0, len(x)), dtype=F64, device=self.dev), None, None, None
        near = torch.zeros(0, dtype=torch.long, device=self.dev)
        if self.scene is not None:
            ps = ParticleState(positions=x, supports=self.Hvec, masses=torch.full_like(rho, self.m), kinds=self.kinds, densities=rho)
            if self.cfg.fusedWall and FusedWall.supported(self.scene) and self._constSupport():
                adj = self._fused_state(ps)
                lam, G, Mw = self.cfg.wallMass * adj.out["lam"], self.cfg.wallMass * adj.out["G"], self.cfg.wallMass * adj.out["Cov"]
            else:
                adj = self.scene.adjacency(ps, self._props(WarpOperation.Density))
                pm = self.scene.precompute(adj, self._props(WarpOperation.Density))
                lam = self._wall_op(ps, pm, WarpOperation.Density, BodyField(rho=1.0))
                G = self._wall_op(ps, pm, WarpOperation.Gradient, BodyField(torch.tensor(1.0, dtype=F64, device=self.dev)))
                Mw = self._wall_op(ps, pm, WarpOperation.Covariance, BodyField(torch.tensor(1.0, dtype=F64, device=self.dev))).reshape(self.nb, len(x), 2, 2)
            if not isinstance(adj, FusedWall) or not self._graphMode:
                near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
        surface = self._detect_surface(x, i, j, r, lam, adj, fk)
        if fk is not None:
            F = fk.dilate(surface)
            Mf = fk.pass1()["Mf"]
        else:
            F = self._sum(surface[j].to(F64), i) > 0.5
            Mf = torch.zeros((len(x), 2, 2), dtype=F64, device=self.dev).index_add_(0, i, V[j][:, None, None] * (-d)[:, :, None] * gW[:, None, :])
        Mt = Mf + (Mw.sum(0) if Mw is not None else 0.0)
        return dict(i=i, j=j, r=r, d=d, gW=gW, V=V, lam=lam, G=G, adj=adj, near=near, surface=surface, F=F, Mf=Mf, Mt=Mt, fk=fk)

    def shift(self, dt):
        """delta+ particle shift (warpSPH `solveShifting`, projection 'surfaceNormal', one iteration): raw shift
        dr_i = -CFL Ma 16 h^2 sum_j  m_j / (2 (rho_i + rho_j)) [1 + R (W_ij / W(dx))^4] grad_i W_ij   (wall: the continuum of wall particles, W^4 term by the exact edge reduction),
        then in the dilated surface set F: a shift pointing into the surface keeps its tangential part (zero where a neighbour's normal differs by > 15 deg), every shift in F is zero where the fluid-only
        lambda_min of the renormalisation matrix is < 0.4, capped at 0.5 Umax dt and clamped to 0.5 dx per component.  Normals n = -grad(lambda_min)/|grad(lambda_min)|, lambda_min of the fluid + wall matrix."""
        cfg, x, v, rho, H = self.cfg, self.x, self.v, self.rho, self.H
        st = self._surface_state(x, rho)
        i, j, r, gW, V, F, fk = st["i"], st["j"], st["r"], st["gW"], st["V"], st["F"], st["fk"]
        n_ = len(x)
        w0 = self._w0
        if fk is not None:
            raw = fk.pass1()["raw"]
        else:
            Wij = self.W(r, H)
            coef = 0.5 * self.m / (rho[i] + rho[j]) * (1.0 + cfg.shiftR * (Wij / w0) ** 4)
            raw = self._sum(coef[:, None] * gW, i)
        if self.scene is not None:
            # wall particles: sum_b m_b/(2 (rho_i + rho_b)) [..] grad W = (rho0 / (4 rho_i)) mu int [1 + R (W/W0)^4] grad_i W dA   (rho_b ~ rho_i, wall particle density mu / dx^2 of mass m)
            #   mu int grad_i W dA = G exactly;  T = int W^4 grad_i W dA by the exact edge reduction (tensile.tensile_vector_scene)
            wall = st["G"].sum(0)
            near = st["near"]
            if isinstance(st["adj"], FusedWall):
                nearm = (st["lam"].sum(0) > 1e-9).to(F64)
                wall = wall + (cfg.wallMass * cfg.shiftR / w0 ** 4 * tensile_factor(H, self._family()) * st["adj"].out["tens"].sum(0)) * nearm[:, None]
            elif len(near):
                T = tensile_vector_scene(self.scene, x[near], H, self._family(), st["adj"].restrict(near))
                wall = wall.index_add(0, near, cfg.wallMass * cfg.shiftR / w0 ** 4 * T)
            raw = raw + (cfg.rho0 / (4.0 * rho))[:, None] * wall
        vmax = v.norm(dim=1).max()                                                                  # device scalars: no host sync
        Ma = vmax / cfg.c0
        Ma = torch.where(Ma >= 1e-6, Ma, torch.full_like(Ma, 0.1))
        hs = H / self.ks
        upd = raw * (-cfg.shiftCFL * Ma * 16.0 * hs ** 2)
        # ---- surface treatment
        lam_t, L = _sym2_lam_pinv(st["Mt"])
        lam_f, _ = _sym2_lam_pinv(st["Mf"], pinv=False)
        if fk is not None:
            gl_ = torch.einsum("nab,nb->na", L, fk.lam_gradient(lam_t))
            nrm = -gl_ / gl_.norm(dim=1, keepdim=True).clamp(min=1e-300)
            minDot = fk.min_dot(F, nrm)
        else:
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
        upd = upd * torch.where(cap > 0, (cap / mag.clamp(min=1e-30)).clamp(max=1.0), torch.ones_like(mag))      # no cap while the flow is at rest (cap = 0)
        upd = upd.clamp(-cfg.shiftThreshold * self.dx, cfg.shiftThreshold * self.dx)
        self.surface, self.surfaceDilated = st["surface"], F
        return upd

    def no_penetration(self):
        """warpSPH mDBC no-penetration, 'impulse' placement, for an analytic wall: a fluid particle at signed distance d < dp/4 from a wall that is closing on it (v_rel . n < 0, n into the fluid) gets
        v += -f vn n with f = 3 - 4 clip(1/2 + d/dp, 1/4, 1) (the ghost / boundary-particle geometry of a flat wall dp/2 inside the solid); f = 1 at the face (inelastic), 2 for a particle 1/4 dp inside (reflection)."""
        if self.scene is None or self.cfg.noPen != "impulse":
            return 0
        ws = self._wall_state(self.x, self.rho)
        lam = ws[0]
        if isinstance(ws[3], FusedWall):                                                           # full-length form: the near-wall rows are a mask (no host sync); a device count of the corrections
            d, n, hit = self.scene.signed_distance(self.x, supportMax=self.H)
            vn = (self.v * n).sum(1)                                                               # static walls (moving bodies: velocityAt of the nearest body, not needed yet)
            f = 3.0 - 4.0 * (0.5 + d / self.dx).clamp(0.25, 1.0)
            act = (lam.sum(0) > 1e-9) & hit & (d < 0.25 * self.dx) & (vn < 0)
            self.v = self.v + torch.where(act[:, None], (-f * vn)[:, None] * n, torch.zeros_like(n))
            return act.sum()
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
        """Sun 2017 time step as a device scalar (no host read): min of the viscous, acceleration and acoustic limits, within [minDt, maxDt], growth-limited."""
        cfg = self.cfg
        if cfg.fixedDt:
            return self.dt_t
        nu = cfg.alpha * cfg.c0 * self.H / (2 * 4)
        dtv = 0.125 * self.H ** 2 / nu / self.ks if cfg.viscosity and cfg.alpha > 0 else cfg.maxDt
        dtc = cfg.cfl * self.H / cfg.c0 / self.ks
        dta = 0.25 * torch.sqrt(self.H / (acc.norm(dim=1).max() + 1e-7)) / self.ks
        new = torch.clamp(dta, max=min(dtv, dtc, cfg.maxDt)).clamp(min=cfg.minDt)
        return torch.where(new > self.dt_t, torch.clamp(new, max=cfg.growth * self.dt_t), new)

    def step(self):
        """one step: eager, or the replay of the captured CUDA graph when cfg.graphStep is set and the configuration allows it (graphstep.py)."""
        if self.cfg.graphStep:
            ck = tuple(getattr(self.cfg, f.name) for f in dataclasses.fields(self.cfg))
            if self._graphed is not None and ck != self._graphCfg:                    # a test or a sweep changed cfg on the live solver: decide (and capture) again
                self._graphed = None
            self._graphCfg = ck
            if self._graphed is None:
                from .graphstep import GraphedStep, graphable
                self._graphed = GraphedStep(self) if graphable(self) else False
            if self._graphed:
                return self._graphed.step()
        return self._step_eager()

    def _step_eager(self):
        """one step: the device part (`_step_core`, no host reads, capturable as a CUDA graph) and the host bookkeeping (time, the rolling gravity, the host copy of dt)."""
        self.dt_t.fill_(self.dt)                                  # the device scalar follows the host value (a test or a caller may set `sim.dt`)
        dt = self.dt
        forces, nopen = self._step_core()
        if forces is not None:
            self.wallForce = forces
        self.nopen_count = nopen
        self.time += dt
        if self.gravityFn is not None:
            self.g = torch.tensor(self.gravityFn(self.time), dtype=F64, device=self.dev)
        self.dt = float(self.dt_t)
        return self.time

    def _step_core(self):
        """DualSPHysics symplectic Euler with time-centred continuity, shifting and the no-penetration impulse on the device state (x, v, rho, g, dt_t); returns (forces, no-penetration count)."""
        dt = self.dt_t
        a0, d0, _ = self.rhs(self.x, self.v, self.rho)
        xh, vh, rh = self.x + 0.5 * dt * self.v, self.v + 0.5 * dt * a0, self.rho + 0.5 * dt * d0
        if self.scene is not None and not self._graphMode:
            for b in self.scene.bodies:
                b.move(0.5 * self.dt)
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
        if self.scene is not None and not self._graphMode:
            for b in self.scene.bodies:
                b.move(0.5 * self.dt)
        if self.cfg.shifting:
            self.x = self.x + self.shift(dt)
        nopen = self.no_penetration()
        self.dt_t.copy_(self._next_dt(a1))                         # in place: the device scalar is a persistent buffer
        return forces, nopen

    # ---------------------------------------------------------------------------------------------------------------- diagnostics
    def pressure(self):
        return self.cfg.c0 ** 2 * (self.rho - self.cfg.rho0)

    def kinetic(self):
        return float(0.5 * self.m * (self.v * self.v).sum())


# ---------------------------------------------------------------------------------------------------------------------------- cases
