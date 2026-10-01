"""omniSPH-style DFSPH in 2D (divergence solve + density solve, XSPH, boundary friction) on the exact-integral scene boundary layer.

The solver follows `omniSPH/simulation/fluidMechanics.cpp` term by term (units of omniSPH: rest density 1, `V` = particle area, Wendland C2,
support h = sqrt(20 V / pi), packing 0.3992 h), but every boundary term is an exact kernel integral over the boundary representation instead of a sum over
triangles hit by a closest-point heuristic:

    density           rho_i   = sum_j V_j W_ij   + lambda_i                         lambda = int_B W            (Density)
    alpha, source     sum_j ~V_j grad W + grad lambda,   - dt v_i . grad lambda     grad lambda = int_B grad W (Gradient)
    pressure update   dt^2 a_i . grad lambda
    boundary accel    a_b,i = -( p_i^+ / rho_i^2 grad lambda + int_B p_b grad W )    p_b from a local linear reconstruction (below)
    friction          v_i -= min(mu lambda, 1) * tangential(v_i),  normal = -grad lambda / |grad lambda|

Wall pressure: p_b(x') = p_i^+ + G_i . (x' - x_i), G_i the kernel-weighted least-squares gradient of the neighbours' pressure (omniSPH extrapolates the
fluid pressure to the wall by an MLS fit; this is the same idea but gives a field over the boundary, so it enters as `BodyField(perQuery=True)` with a per-query
gradient and works for every representation, including SDF and primitives).  For the hydrostatic pressure field p = rho g (H - y) the reconstruction is exact.

All fluid-fluid sums are plain torch pair sums (cell list neighbours, `edgebound.scene.buildCellList`) and are checked against `warpSPHCore.warpOperation`.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation

from .implicitBodies import HalfPlaneBody
from .scene import Body, BodyField, ImplicitRep, Scene, SdfRep, SurfaceRep, VolumeRep, buildCellList, sceneOperation

F64 = torch.float64
PACKING = 0.399200743165053487            # omniSPH packing_2D (spacing / h)
TARGET_NEIGHBORS = 20


# ----------------------------------------------------------------------------------------------------------------------------- kernel / pairs
def wendland2(r, h):
    q = r / h
    return torch.where(q < 1, 7.0 / (math.pi * h * h) * (1 - q) ** 4 * (1 + 4 * q), torch.zeros_like(q))


def dwendland2(r, h):
    q = r / h
    return torch.where(q < 1, -7.0 / (math.pi * h ** 3) * 20.0 * q * (1 - q) ** 3, torch.zeros_like(q))


def neighbor_pairs(pos, h):
    """(i, j, r) for all ordered pairs |x_i - x_j| <= (h_i + h_j) / 2, including i = j (as omniSPH's neighbour lists do)."""
    dev = pos.device
    cell = float(h.max())
    cl = buildCellList(pos, pos, cell)
    nx, ny = cl.dims
    c = torch.floor((pos - cl.lo) / cell).long()
    I, J = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx, cy = c[:, 0] + dx, c[:, 1] + dy
            ok = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
            cid = torch.where(ok, cx * ny + cy, torch.zeros_like(cx))
            cnt = torch.where(ok, cl.start[cid + 1] - cl.start[cid], torch.zeros_like(cid))
            q = torch.repeat_interleave(torch.arange(len(pos), device=dev), cnt)
            off = torch.arange(int(cnt.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
            I.append(q)
            J.append(cl.items[cl.start[cid[q]] + off])
    i, j = torch.cat(I), torch.cat(J)
    r = (pos[i] - pos[j]).norm(dim=1)
    keep = r <= 0.5 * (h[i] + h[j])
    return i[keep], j[keep], r[keep]


# ----------------------------------------------------------------------------------------------------------------------------- lattice calibration
def lattice_calibration(dx, dy, h, kernel="w2"):
    """Make a regular particle lattice an exact rest state of the SPH discretisation:
        S    = sum over the infinite lattice of W * dx dy                       (the discrete kernel sum, = 1 + O(1e-2) for h / dx ~ 2.5)
        V'   = dx dy / S      -> sum_j V' W = 1 in the bulk
        mu   = 1 / S          mass per area of the wall continuum (a wall filled with the same lattice)
        d_x, d_y             distance of the first lattice row from the wall face for which the first row also has density exactly 1.
    omniSPH uses V = pi r^2 and a face one spacing outside the block; with these numbers the initial density is 1 +- few %, which DFSPH removes in ONE step
    (an impulse); calibrated, the lattice starts at rest (density error ~1e-3)."""
    from .implicitBodies import Tier3
    W = lambda r: np.where(r < h, 7.0 / (np.pi * h * h) * (1 - np.minimum(r / h, 1)) ** 4 * (1 + 4 * np.minimum(r / h, 1)), 0.0)
    N = int(np.ceil(h / min(dx, dy))) + 2
    n, m = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1), indexing="ij")
    S = float((W(np.hypot(n * dx, m * dy)) * dx * dy).sum())
    Vp, mu = dx * dy / S, 1.0 / S
    t3 = Tier3(kernel, "cpu")
    lam = lambda d: float(t3.lam(torch.tensor([d / h], dtype=F64), torch.zeros(1, dtype=F64))[0])

    def rho_row0(d, spacing_normal, spacing_tan):
        k = np.arange(0, N + 2)
        nn = np.arange(-N, N + 1)
        X, Y = np.meshgrid(nn * spacing_tan, k * spacing_normal, indexing="ij")
        return Vp * W(np.hypot(X, Y)).sum() + mu * lam(d)

    def solve(sn, st):
        a, b = 0.2 * sn, 1.5 * sn
        fa, fb = rho_row0(a, sn, st) - 1.0, rho_row0(b, sn, st) - 1.0
        for _ in range(60):
            c = 0.5 * (a + b)
            fc = rho_row0(c, sn, st) - 1.0
            if fa * fc <= 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return 0.5 * (a + b)
    return dict(S=S, V=Vp, mu=mu, dwallX=solve(dx, dy), dwallY=solve(dy, dx))


# ----------------------------------------------------------------------------------------------------------------------------- domains
def domain_scene(kind: str, lo, hi, h: float, device, thickness: Optional[float] = None, volumeMode: str = "moments"):
    """closed box of fluid [lo, hi] (the INNER wall faces) as one body (`volume`: omniSPH's thick triangle slabs, `surface`: one clockwise loop in an infinite solid,
    `sdf`: sampled box SDF with the loop as fallback near the corners, `halfplanes`: four half planes (negative control: double counts the corner regions))."""
    (x0, y0), (x1, y1) = lo, hi
    t = thickness if thickness is not None else 2.5 * h
    if kind == "surface":
        return Scene([Body(reps=[SurfaceRep.box((x0, y0), (x1, y1), solid="outside")])], device)
    if kind == "volume":
        V = np.array([[x0 - t, y0 - t], [x0, y0 - t], [x1, y0 - t], [x1 + t, y0 - t],
                      [x0 - t, y0], [x0, y0], [x1, y0], [x1 + t, y0],
                      [x0 - t, y1], [x0, y1], [x1, y1], [x1 + t, y1],
                      [x0 - t, y1 + t], [x0, y1 + t], [x1, y1 + t], [x1 + t, y1 + t]], dtype=float)
        E = []
        for r in range(3):
            for c in range(4 - 1):
                if (r, c) == (1, 1):                 # the fluid cell
                    continue
                a, b, cc, d = 4 * r + c, 4 * r + c + 1, 4 * (r + 1) + c, 4 * (r + 1) + c + 1
                E += [[a, b, d], [a, d, cc]]
        return Scene([Body(reps=[VolumeRep(V, np.array(E))])], device, volumeMode=volumeMode)
    if kind == "sdf":
        c = torch.tensor([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=F64)
        hw = torch.tensor([0.5 * (x1 - x0), 0.5 * (y1 - y0)], dtype=F64)

        def boxsdf(p):                              # positive inside the fluid box
            q = (p - c).abs() - hw
            return -(q.clamp(min=0).norm(dim=1) + q.max(dim=1).values.clamp(max=0))
        m = 2 * h
        sdf = SdfRep.fromFunction(boxsdf, (x0 - m, y0 - m), (x1 + m, y1 + m), h / 16, fallback=SurfaceRep.box((x0, y0), (x1, y1), solid="outside"))
        return Scene([Body(reps=[sdf])], device)
    if kind == "halfplanes":
        planes = [((x0, 0.0), (1.0, 0.0)), ((x1, 0.0), (-1.0, 0.0)), ((0.0, y0), (0.0, 1.0)), ((0.0, y1), (0.0, -1.0))]
        return Scene([Body(bodyId=i, reps=[ImplicitRep(HalfPlaneBody(p, n))]) for i, (p, n) in enumerate(planes)], device)
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------------------------------------- solver
@dataclass
class DFSPHConfig:
    gravity: tuple = (0.0, -9.81)
    maxDt: float = 1e-3
    minDt: float = 1e-4
    cfl: float = 0.4
    divergenceSolve: bool = True
    densityEta: float = 1e-3
    divergenceEta: float = 1e-3
    maxIterations: int = 256
    divergenceMaxIterations: int = 3         # omniSPH: the divergence loop always runs exactly 4 iterations
    minIterations: int = 4
    boundaryInDivergence: bool = False       # omniSPH's divergence solve ignores the wall; True adds the wall flux, alpha and wall pressure acceleration (no clamp)
    omega: float = 0.5
    xsph: float = 1e-4
    boundaryFriction: float = 5e-3
    kernel: KernelFunctions = KernelFunctions.Wendland2
    wallPressure: str = "hydrostatic"       # 'hydrostatic' (p_b = p_i + rho g.(x'-x_i): dp/dn = rho (g - a_wall).n), 'linear' (MLS gradient of the neighbours' pressure), 'mirror' (p_b = p_i)
    wallMass: float = 1.0                   # mass per area of the wall continuum (1 = omniSPH; 1/S for a calibrated lattice)


class DFSPH2D:
    def __init__(self, positions, velocities, V, h, scene: Optional[Scene], cfg: Optional[DFSPHConfig] = None, device="cuda:0"):
        self.cfg = cfg or DFSPHConfig()
        self.dev = device
        t = lambda a: torch.as_tensor(a, dtype=F64, device=device)
        self.x, self.v = t(positions).clone(), t(velocities).clone()
        n = len(self.x)
        self.V = t(V) if np.ndim(V) else torch.full((n,), float(V), dtype=F64, device=device)
        self.h = t(h) if np.ndim(h) else torch.full((n,), float(h), dtype=F64, device=device)
        self.scene = scene
        self.p = torch.zeros(n, dtype=F64, device=device)               # fluidPriorPressure
        self.rho = torch.ones(n, dtype=F64, device=device)
        self.dt = self.cfg.maxDt
        self.time = 0.0
        self.iters = (0, 0)
        self.wallForce = torch.zeros(2, dtype=F64, device=device)
        self.stats = {}
        self.kinds = torch.zeros(n, dtype=torch.int32, device=device)

    # ---- geometry-dependent data, valid for the current positions
    def _prepare(self):
        x, h = self.x, self.h
        i, j, r = neighbor_pairs(x, h)
        hij = 0.5 * (h[i] + h[j])
        self.pi, self.pj = i, j
        self.W = wendland2(r, hij)
        d = x[i] - x[j]
        self.gW = torch.where((r > 1e-14 * hij)[:, None], dwendland2(r, hij)[:, None] * d / r.clamp(min=1e-300)[:, None], torch.zeros_like(d))
        self.ps = self._particleState(torch.ones_like(self.V))
        if self.scene is not None:
            self.adj = self.scene.buildAdjacency(self.ps, self._props(WarpOperation.Density))
            self.lam = self.cfg.wallMass * self._boundary(WarpOperation.Density, BodyField(rho=1.0))
            self.gk = self.cfg.wallMass * self._boundary(WarpOperation.Gradient, BodyField(torch.tensor(1.0, dtype=F64, device=self.dev)))
        else:
            self.lam = torch.zeros_like(self.V)
            self.gk = torch.zeros_like(self.x)
        # moment matrix of the pressure reconstruction
        w = self.V[j] * self.W
        y = x[j] - x[i]
        M = torch.zeros((len(x), 2, 2), dtype=F64, device=self.dev).index_add_(0, i, (w[:, None, None] * y[:, :, None] * y[:, None, :]))
        ev, U = torch.linalg.eigh(M)
        inv = torch.where(ev > 1e-3 * ev.amax(1, keepdim=True).clamp(min=1e-300), 1.0 / ev.clamp(min=1e-300), torch.zeros_like(ev))
        self.Minv = U @ torch.diag_embed(inv) @ U.transpose(1, 2)
        self.wy = w[:, None] * y

    def _particleState(self, rho):
        return ParticleState(positions=self.x, supports=self.h, masses=self.V.clone(), kinds=self.kinds, densities=rho)

    def _props(self, op, mode=GradientScheme.Naive):
        return OperationProperties(kernel=self.cfg.kernel, operation=op, gradientMode=mode, operationMode=OperationDirection.BoundaryToFluid)

    def _boundary(self, op, fld, mode=GradientScheme.Naive, queryValues=None, rho=None):
        ps = self.ps if rho is None else self._particleState(rho)
        return sceneOperation(ps, self._props(op, mode), self.scene, self.adj, queryValues, [fld] * len(self.scene.bodies))

    # ---- sums over fluid neighbours
    def _sum(self, vals):
        out = torch.zeros((len(self.x),) + vals.shape[1:], dtype=vals.dtype, device=self.dev)
        return out.index_add_(0, self.pi, vals)

    def _fluid_accel(self, p):
        """-sum_j V_j (p_i / rho_i^2 + p_j / rho_j^2) grad W_ij"""
        i, j = self.pi, self.pj
        f = p / self.rho ** 2
        return self._sum(-(self.V[j] * (f[i] + f[j]))[:, None] * self.gW)

    def _boundary_accel(self, p, clamp=True):
        """-( p_i^+ / rho_i^2 grad lambda + int p_b grad W ),  p_b = p_i^+ + G_i . (x' - x_i)"""
        if self.scene is None:
            return torch.zeros_like(self.x)
        pp = p.clamp(min=0) if clamp else p
        a1 = None
        if self.cfg.wallPressure == "hydrostatic":                       # dp/dn = rho (g - a_wall) . n: the wall pressure continues the hydrostatic gradient
            a1 = self.rho[:, None] * torch.tensor(self.cfg.gravity, dtype=F64, device=self.dev)[None]
        elif self.cfg.wallPressure == "linear":
            dp = pp[self.pj] - pp[self.pi]
            rhs = self._sum(dp[:, None] * self.wy)
            a1 = torch.einsum("nij,nj->ni", self.Minv, rhs)
        fld = BodyField(pp, a1, rho=1.0, perQuery=True)
        out = self._boundary(WarpOperation.Gradient, fld, GradientScheme.Symmetric, queryValues=pp, rho=self.rho)
        return -self.cfg.wallMass * out / self.rho[:, None]

    # ---- DFSPH
    def _alpha(self, dt, Vt, density):
        j = self.pj
        ks1 = self._sum(Vt[j][:, None] * self.gW)
        if density:
            ks1 = ks1 + self.gk
        ks2 = self._sum((Vt[j] ** 2 / self.V[j]) * (self.gW * self.gW).sum(1))
        return -dt * dt * Vt / self.V * (ks1 * ks1).sum(1) - dt * dt * Vt * ks2

    def _source(self, dt, Vt, vp, density, wall):
        i, j = self.pi, self.pj
        s = self._sum(-dt * Vt[j] * ((vp[i] - vp[j]) * self.gW).sum(1))
        if density:
            s = s + 1.0 - self.rho
        if wall:
            s = s - dt * (vp * self.gk).sum(1)
        return s

    def _solve(self, acc, density):
        cfg, dt = self.cfg, self.dt
        i, j = self.pi, self.pj
        Vt = self.V / self.rho
        vp = self.v + dt * acc
        wall = density or cfg.boundaryInDivergence
        alpha = self._alpha(dt, Vt, wall)
        src = self._source(dt, Vt, vp, density, wall)
        p2 = 0.5 * self.p if density else torch.zeros_like(self.V)
        p1 = p2.clone()
        eta = cfg.densityEta if density else cfg.divergenceEta
        maxit = cfg.maxIterations if density else cfg.divergenceMaxIterations
        counter = 0
        while True:
            pred = (self._boundary_accel(p2, density) if wall else torch.zeros_like(self.x)) + self._fluid_accel(p2)
            p1 = p2
            ks = dt * dt * self._sum(Vt[j] * ((pred[i] - pred[j]) * self.gW).sum(1))
            if wall:
                ks = ks + dt * dt * (pred * self.gk).sum(1)
            pn = p1 + cfg.omega / alpha * (src - ks)
            if density:
                pn = pn.clamp(min=0)
            bad = (alpha.abs() < 1e-25) | ~torch.isfinite(pn) | (pn > 1e25)
            p2 = torch.where(bad, torch.zeros_like(pn), pn)
            res = torch.where(bad, torch.zeros_like(ks), ks - src)
            err = torch.maximum(res, torch.full_like(res, -0.001)).mean()                      # omniSPH: mean(max(residual, -0.001))
            counter += 1
            if counter >= cfg.minIterations and not (float(err) > eta and counter < maxit):
                break
        pred = (self._boundary_accel(p2, density) if wall else torch.zeros_like(self.x)) + self._fluid_accel(p2)
        if density:
            self.wallForce = -(self.V[:, None] * self._boundary_accel(p2)).sum(0) if self.scene is not None else self.wallForce
            self.p = p2
        self.err = float(err)
        return acc + pred, counter

    def step(self):
        cfg, dt = self.cfg, self.dt
        self._prepare()
        i, j = self.pi, self.pj
        self.rho = self._sum(self.V[j] * self.W) + self.lam
        acc = torch.tensor(cfg.gravity, dtype=F64, device=self.dev).expand(len(self.x), 2).clone()
        nd = 0
        if cfg.divergenceSolve:
            acc, nd = self._solve(acc, False)
        acc, nq = self._solve(acc, True)
        self.iters = (nd, nq)
        # XSPH
        w = 2.0 * self.V[j] / (self.rho[i] + self.rho[j]) * self.W
        self.v = self.v + cfg.xsph * self._sum(w[:, None] * (self.v[j] - self.v[i]))
        # integrate
        self.v = self.v + dt * acc
        self.x = self.x + dt * self.v
        self.time += dt
        # boundary friction at the new positions
        if self.scene is not None and cfg.boundaryFriction > 0:
            self.ps = self._particleState(torch.ones_like(self.V))
            adj = self.scene.buildAdjacency(self.ps, self._props(WarpOperation.Density))
            lam = cfg.wallMass * sceneOperation(self.ps, self._props(WarpOperation.Density), self.scene, adj, None, [BodyField(rho=1.0)] * len(self.scene.bodies))
            gk = cfg.wallMass * sceneOperation(self.ps, self._props(WarpOperation.Gradient), self.scene, adj, None,
                                               [BodyField(torch.tensor(1.0, dtype=F64, device=self.dev))] * len(self.scene.bodies))
            nrm = gk.norm(dim=1, keepdim=True)
            n = -gk / nrm.clamp(min=1e-300)
            tang = self.v - (self.v * n).sum(1, keepdim=True) * n
            fac = (cfg.boundaryFriction * lam).clamp(max=1.0)
            self.v = torch.where((nrm[:, 0] > 0)[:, None], self.v - fac[:, None] * tang, self.v)
        vmax = float(self.v.norm(dim=1).max())
        self.dt = float(np.clip(cfg.cfl * float(self.h.min()) / max(vmax, 1e-12), cfg.minDt, cfg.maxDt))
        return self.time


# ----------------------------------------------------------------------------------------------------------------------------- diagnostics
def diagnostics(sim: DFSPH2D, g=9.81):
    x, v, V = sim.x, sim.v, sim.V
    return dict(t=sim.time, ke=float(0.5 * (V * (v * v).sum(1)).sum()), pe=float((V * g * x[:, 1]).sum()), xmax=float(x[:, 0].max()),
                ymax=float(x[:, 1].max()), vmax=float(v.norm(dim=1).max()), rhoMax=float(sim.rho.max()), rhoMean=float(sim.rho.mean()),
                com=(float(x[:, 0].mean()), float(x[:, 1].mean())), wallForce=tuple(float(a) for a in sim.wallForce), iters=sim.iters)
