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

All fluid-fluid sums are plain torch pair sums (cell list neighbours, `warpSPHBoundaries.scene.buildCellList`) and are checked against `warpSPHCore.warpOperation`.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation

from ..scene.implicitBodies import HalfPlaneBody
from ..scene.scene import Body, BodyField, BoxRep, ImplicitRep, Scene, SdfRep, SurfaceRep, VolumeRep, sceneOperation
from ..scene.fused import WallOutput
from .pairs import F64, dwendland2, neighbor_pairs, wendland2

PACKING = 0.399200743165053487            # omniSPH packing_2D (spacing / h)
TARGET_NEIGHBORS = 20


# ----------------------------------------------------------------------------------------------------------------------------- lattice calibration
def lattice_calibration(dx, dy, h, kernel="w2"):
    """Make a regular particle lattice an exact rest state of the SPH discretisation:
        S    = sum over the infinite lattice of W * dx dy                       (the discrete kernel sum, = 1 + O(1e-2) for h / dx ~ 2.5)
        V'   = dx dy / S      -> sum_j V' W = 1 in the bulk
        mu   = 1 / S          mass per area of the wall continuum (a wall filled with the same lattice)
        d_x, d_y             distance of the first lattice row from the wall face for which the first row also has density exactly 1.
    omniSPH uses V = pi r^2 and a face one spacing outside the block; with these numbers the initial density is 1 +- few %, which DFSPH removes in ONE step
    (an impulse); calibrated, the lattice starts at rest (density error ~1e-3)."""
    from ..scene.implicitBodies import Tier3
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
    dwx, dwy = solve(dx, dy), solve(dy, dx)
    return dict(S=S, V=Vp, mu=mu, dwallX=dwx, dwallY=dwy, lamRow0=lam(dwy))


def carve(positions, bodies, h, lamMax, device, kernel=KernelFunctions.Wendland2):
    """remove the lattice particles that overlap the given bodies: those with a body kernel integral lambda > lamMax (`lattice_calibration(...)['lamRow0']`: the first row
    of a flat wall; so a carved surface keeps the same first-row distance as the flat walls).  Returns the kept positions."""
    pos = torch.as_tensor(positions, dtype=F64, device=device)
    n = len(pos)
    ps = ParticleState(positions=pos, supports=torch.full((n,), float(h), dtype=F64, device=device), masses=torch.ones(n, dtype=F64, device=device),
                       kinds=torch.zeros(n, dtype=torch.int32, device=device), densities=torch.ones(n, dtype=F64, device=device))
    sc = Scene(bodies, device)
    props = OperationProperties(kernel=kernel, operation=WarpOperation.Density, operationMode=OperationDirection.BoundaryToFluid)
    lam = sceneOperation(ps, props, sc, bodyFields=[BodyField(rho=1.0)] * len(bodies))
    return positions[(lam <= lamMax).cpu().numpy()]


# ----------------------------------------------------------------------------------------------------------------------------- domains
def domain_scene(kind: str, lo, hi, h: float, device, thickness: Optional[float] = None, volumeMode: str = "moments"):
    """closed box of fluid [lo, hi] (the INNER wall faces) as one body (`volume`: omniSPH's thick triangle slabs, `surface`: one clockwise loop in an infinite solid,
    `box`: the same loop as a `BoxRep`, `sdf`: sampled box SDF with the loop as fallback near the corners, `halfplanes`: four half planes (negative control: double counts the corner regions))."""
    (x0, y0), (x1, y1) = lo, hi
    t = thickness if thickness is not None else 2.5 * h
    if kind == "surface":
        return Scene([Body(reps=[SurfaceRep.box((x0, y0), (x1, y1), solid="outside")])], device)
    if kind == "box":                                 # the same domain as a BoxRep: corner tables, no edge pairs (docs/box-domain-primitive.md)
        return Scene([Body(reps=[BoxRep((x0, y0), (x1, y1), solid="outside")])], device)
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
    divergenceClamp: bool = False            # clamp the divergence pressure of ALL particles at >= 0 as the density pressure is (omniSPH does not: its negative divergence pressure is the cohesion that keeps
                                             # the fluid together after a splash; clamping it expands the fluid by ~30 %).  Needed only for many divergence iterations with a moving wall
    wallDivergenceClamp: Optional[bool] = None   # clamp the divergence pressure only where it enters the WALL acceleration (fluid-fluid terms keep the signed pressure).  None: iff the wall is in the divergence solve.
                                             # An unclamped negative pressure there is a wall suction: with a moving body (wall in the divergence solve) a particle leaving a wall is pulled back and
                                             # fluid sticks to every wall, ceiling included (docs/dfsph-validation.md s.8)
    boundaryInDivergence: Optional[bool] = None   # omniSPH's divergence solve ignores the wall; True adds the wall flux, alpha and wall pressure acceleration (no clamp);
                                                  # None: True iff a body moves (a moving wall has a normal velocity the divergence solve must see)
    recordForces: bool = True
    gradientCorrection: str = "none"         # 'wall': zeroth-order consistent wall closure (docs/dfsph-validation.md s.7): the wall gradient of the p_i terms is scaled so that a uniform pressure
                                             #   exerts no net force on a wall-contact particle, s_i = clip(-(sum_j V_j grad W_ij).n / |grad lambda|, 1 - kappa, 1 + kappa)
    closureLimit: float = 0.2                # kappa
    clampWallPressure: bool = True          # p_b >= 0 (omniSPH clamps the extrapolated wall pressure in the density solve): the hydrostatic wall term can never be a suction, e.g. at a ceiling
    omega: float = 0.5
    xsph: float = 1e-4
    boundaryFriction: float = 5e-3
    kernel: KernelFunctions = KernelFunctions.Wendland2
    wallPressure: str = "hydrostatic"       # 'hydrostatic' (p_b = p_i + rho g.(x'-x_i): dp/dn = rho (g - a_wall).n), 'linear' (MLS gradient of the neighbours' pressure), 'mirror' (p_b = p_i)
    wallMass: float = 1.0                   # mass per area of the wall continuum (1 = omniSPH; 1/S for a calibrated lattice)
    wallBackend: str = "auto"               # 'fused': the wall terms from the boundary provider (AnalyticBoundary / FusedWall: lam, grad lam, the first-moment tensor int y (x) grad W, m1 = int y W, per body; one
                                            # evaluation per position set, the iterates contract the stored tensor); 'scene': the oracle `sceneOperation`; 'auto': fused when every body is supported (surface, box,
                                            # disks, implicit / SDF lowered), else scene (VolumeRep: the omniSPH slabs)


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
        nb = len(scene.bodies) if scene is not None else 0
        self.nb = nb
        self.forcePressure = torch.zeros((nb, 2), dtype=F64, device=device)    # force of the fluid on each body, pressure part (this step)
        self.forceFriction = torch.zeros((nb, 2), dtype=F64, device=device)    # friction (boundary viscosity) part
        self.history = []                                                       # per step: dict(t, dt, pressure [B,2], friction [B,2], balance)
        self.balance = 0.0
        self._pmNext = None
        self._fwNext = None
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
        if self.scene is not None and self._fused():
            fw = self._fwNext if self._fwNext is not None else self._aggregate(self.ps)
            self._fwNext = None
            wm = self.cfg.wallMass
            self.lam = wm * fw.out["lam"].sum(0)
            self.gkb = wm * fw.out["G"]                                                          # [B,N,2] mu int grad W per body
            self.covb = wm * fw.out["Cov"]                                                       # [B,N,2,2] mu int y_d d_j W per body: the wall pressure term and the rigid wall divergence
            self.wallDiv = self._wall_div() if self._moving() else None
            self.gk = self.gkb.sum(0)
            self.sClose = self._closure()
            self.gk = self.sClose[:, None] * self.gk
            self._out1 = None
            self._theta_q = None
        elif self.scene is not None:
            self.pm = self._pmNext if self._pmNext is not None else self.scene.pairMoments(self.ps, self._props(WarpOperation.Density))
            self._pmNext = None
            self.wallDiv = self._wall_div() if self._moving() else None
            self.lam = self.cfg.wallMass * self._boundary(WarpOperation.Density, BodyField(rho=1.0))
            self.gkb = self.cfg.wallMass * self._boundary(WarpOperation.Gradient, BodyField(torch.tensor(1.0, dtype=F64, device=self.dev)), perBody=True)    # [B,N,2] mu int grad W per body
            self.gk = self.gkb.sum(0)
            self.sClose = self._closure()
            self.gk = self.sClose[:, None] * self.gk
            self._out1 = None
            self._theta_q = None
        else:
            self.lam = torch.zeros_like(self.V)
            self.gk = torch.zeros_like(self.x)
            self.gkb = torch.zeros((0,) + self.x.shape, dtype=F64, device=self.dev)
            self.sClose = torch.ones_like(self.V)
            self.wallDiv = None
        # moment matrix of the pressure reconstruction
        w = self.V[j] * self.W
        y = x[j] - x[i]
        M = torch.zeros((len(x), 2, 2), dtype=F64, device=self.dev).index_add_(0, i, (w[:, None, None] * y[:, :, None] * y[:, None, :]))
        ev, U = torch.linalg.eigh(M)
        inv = torch.where(ev > 1e-3 * ev.amax(1, keepdim=True).clamp(min=1e-300), 1.0 / ev.clamp(min=1e-300), torch.zeros_like(ev))
        self.Minv = U @ torch.diag_embed(inv) @ U.transpose(1, 2)
        self.wy = w[:, None] * y

    def _closure(self):
        """zeroth-order consistent wall closure: a uniform pressure must exert no net force on a particle that touches a wall.  The fluid neighbours give sum_j V_j grad W_ij (the discrete
        incompleteness of the kernel sum); the wall has to supply minus that.  The exact wall gradient `gk` fixes the direction (the geometry), the closure only rescales its magnitude,
        s_i = -(sum_j V_j grad W_ij) . n / |gk|  clipped to [1 - kappa, 1 + kappa] (n = gk / |gk|): bounded where the incompleteness is a free surface rather than the wall."""
        if self.cfg.gradientCorrection != "wall":
            return torch.ones_like(self.V)
        gsum = -self._sum(self.V[self.pj][:, None] * self.gW)
        mag = self.gk.norm(dim=1)
        n = self.gk / mag.clamp(min=1e-300)[:, None]
        s = (gsum * n).sum(1) / mag.clamp(min=1e-300)
        k = self.cfg.closureLimit
        return torch.where(mag > 1e-12, s.clamp(1 - k, 1 + k), torch.ones_like(s))

    def _fused(self):
        """the wall terms come from the boundary provider (cfg.wallBackend)."""
        mode = self.cfg.wallBackend
        if mode == "scene":
            return False
        if getattr(self, "_provider", None) is None:
            from ..scene.provider import AnalyticBoundary
            self._provider = AnalyticBoundary(self.scene)
        if mode == "fused":
            return True
        if mode != "auto":
            raise ValueError("wallBackend must be 'auto', 'fused' or 'scene', got %r" % (mode,))
        if getattr(self, "_fusedOK", None) is None:
            self._fusedOK = bool(self._provider.supported(fixedAdjacency=True))
        return self._fusedOK

    def _aggregate(self, ps):
        """the provider's wall aggregate at the positions of `ps` (lam, G, Cov in `.out`)."""
        return self._provider.aggregate(ps, float(self.h.max()), kernel=self.cfg.kernel)

    def _particleState(self, rho):
        return ParticleState(positions=self.x, supports=self.h, masses=self.V.clone(), kinds=self.kinds, densities=rho)

    def _props(self, op, mode=GradientScheme.Naive):
        return OperationProperties(kernel=self.cfg.kernel, operation=op, gradientMode=mode, operationMode=OperationDirection.BoundaryToFluid)

    def _boundary(self, op, fld, mode=GradientScheme.Naive, queryValues=None, rho=None, perBody=False, pm=None, ps=None):
        ps = ps if ps is not None else (self.ps if rho is None else self._particleState(rho))
        flds = fld if isinstance(fld, list) else [fld] * len(self.scene.bodies)
        return sceneOperation(ps, self._props(op, mode), self.scene, pm or self.pm, queryValues, flds, perBody=perBody)

    def _moving(self):
        return self.scene is not None and any(float(b.angularVelocity) != 0.0 or float(b.linearVelocity.norm()) != 0.0 for b in self.scene.bodies)

    def _wall_div(self):
        """sum over bodies of  int v_b . grad W  (the wall velocity enters the source term as (v_p,i - v_b) . grad W), 0 for fixed walls."""
        out = torch.zeros(len(self.x), dtype=F64, device=self.dev)
        if self._fused():                                                                       # int (v_b(x) + omega J y) . grad W = v_b(x) . G + omega (C_01 - C_10), C_dj = int y_d d_j W
            for bi, b in enumerate(self.scene.bodies):
                if float(b.angularVelocity) == 0.0 and float(b.linearVelocity.norm()) == 0.0:
                    continue
                C = self.covb[bi]
                out = out + (b.velocityAt(self.x) * self.gkb[bi]).sum(1) + float(b.angularVelocity) * (C[:, 0, 1] - C[:, 1, 0])
            return out                                                                          # covb / gkb already carry the wall mass
        for bi, b in enumerate(self.scene.bodies):
            if float(b.angularVelocity) == 0.0 and float(b.linearVelocity.norm()) == 0.0:
                continue
            f = [BodyField(torch.zeros(2, dtype=F64, device=self.dev))] * len(self.scene.bodies)
            f[bi] = BodyField.rigid(b)
            out = out + self._boundary(WarpOperation.Divergence, f)
        return self.cfg.wallMass * out

    # ---- sums over fluid neighbours
    def _sum(self, vals):
        out = torch.zeros((len(self.x),) + vals.shape[1:], dtype=vals.dtype, device=self.dev)
        return out.index_add_(0, self.pi, vals)

    def _fluid_accel(self, p):
        """-sum_j V_j (p_i / rho_i^2 + p_j / rho_j^2) grad W_ij"""
        i, j = self.pi, self.pj
        f = p / self.rho ** 2
        return self._sum(-(self.V[j] * (f[i] + f[j]))[:, None] * self.gW)

    def _boundary_accel(self, p, clamp=True, perBody=False):
        """acceleration of the fluid by the walls, omniSPH form  -mu ( p_i^+/rho_i^2 int grad W + int p_b grad W ),  p_b(x') = p_i^+ + a1_i . (x' - x_i):

            a_b = -(p_i^+/rho_i^2 + p_i^+) s_i mu grad lambda_b  -  mu int (a1_i . y) grad W          (s_i = wall closure factor, 1 without correction)

        a1_i = rho_i (g - a_wall,b(x_i)) (`hydrostatic`: dp/dn = rho (g - a_wall) . n, the O(omega^2 h^2) curvature of a rotating wall neglected) | MLS gradient (`linear`) | 0 (`mirror`).
        The first part is a per-body gradient of lambda fixed within the step, the second does not depend on p for `hydrostatic` / `mirror` and is cached: no scene operation per iteration.
        `perBody`: [B, N, 2] with the contribution of every body."""
        if self.scene is None:
            return torch.zeros((self.nb, len(self.x), 2) if perBody else self.x.shape, dtype=F64, device=self.dev)
        pp = p.clamp(min=0) if clamp else p
        pfac = (pp / self.rho ** 2 + pp) * self.sClose
        a1 = self._a1_part(pp)
        if self.cfg.clampWallPressure and self.cfg.wallPressure == "hydrostatic":
            a1 = a1 - self._wall_excess(pp)[:, :, None] * self.gkb
        out = -pfac[None, :, None] * self.gkb - a1
        return out if perBody else out.sum(0)

    def _wall_excess(self, pp):
        """p_b >= 0.  To leading order the hydrostatic term mu int (a1.y) grad W is an effective wall pressure offset q (per body) times the wall gradient, q = (term . n) / |mu grad lambda|, n = grad lambda / |grad lambda|
        (q < 0: the wall is above the particle, p_b = p_i + rho g.(x' - x_i) < p_i).  The effective wall pressure p_i + q is clamped at 0: with theta = clip(p_i / (-q), 0, 1) for q < 0 (theta = 1 where the wall is below or beside
        the particle, or the particle's own pressure carries the offset) only the normal pressure offset is reduced, the term loses (1 - theta) q grad lambda (-> 0 continuously for q -> 0, so round-off cannot flip it; the tangential part is untouched)."""
        if self._theta_q is None:
            a1 = self._a1_part(pp)
            eps = 1e-5 * self.cfg.wallMass / float(self.h.min())          # |mu grad lambda| ~ mu / h in contact: below 1e-5 of that the wall force is negligible and the ratio is round-off
            self._theta_q = (a1 * self.gkb).sum(2) / (self.gkb * self.gkb).sum(2).clamp(min=eps * eps)
        q = self._theta_q
        theta = torch.where(q < 0, (pp[None, :] / (-q).clamp(min=1e-300)).clamp(0.0, 1.0), torch.ones_like(q))
        return (1.0 - theta) * q

    def _a1_part(self, pp):
        """mu int (a1_i . y) grad W per body [B,N,2]."""
        mode = self.cfg.wallPressure
        if mode == "mirror":
            return torch.zeros_like(self.gkb)
        if mode == "hydrostatic" and self._out1 is not None:
            return self._out1
        g = torch.tensor(self.cfg.gravity, dtype=F64, device=self.dev)[None]
        flds = []
        for b in self.scene.bodies:
            if mode == "hydrostatic":
                a1 = self.rho[:, None] * (g - b.accelerationAt(self.x))
            else:
                dp = pp[self.pj] - pp[self.pi]
                a1 = torch.einsum("nij,nj->ni", self.Minv, self._sum(dp[:, None] * self.wy))
            flds.append(BodyField(torch.zeros_like(pp), a1, rho=1.0, perQuery=True))
        if self._fused():                                                                       # mu int (a1 . y) grad W = a1_d C_dj (the stored tensor: no launch per iterate)
            out = torch.stack([torch.einsum("nd,ndj->nj", f.a1, self.covb[bi]) for bi, f in enumerate(flds)])
        else:
            out = self.cfg.wallMass * self._boundary(WarpOperation.Gradient, flds, GradientScheme.Naive, perBody=True)
        if mode == "hydrostatic":
            self._out1 = out
        return out

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
            if self.wallDiv is not None:
                s = s + dt * self.wallDiv
        return s

    def _solve(self, acc, density):
        cfg, dt = self.cfg, self.dt
        i, j = self.pi, self.pj
        Vt = self.V / self.rho
        vp = self.v + dt * acc
        wall = density or self._bdiv
        alpha = self._alpha(dt, Vt, wall)
        src = self._source(dt, Vt, vp, density, wall)
        p2 = 0.5 * self.p if density else torch.zeros_like(self.V)
        p1 = p2.clone()
        eta = cfg.densityEta if density else cfg.divergenceEta
        maxit = cfg.maxIterations if density else cfg.divergenceMaxIterations
        counter = 0
        while True:
            clampP = density or cfg.divergenceClamp
            pred = (self._boundary_accel(p2, density or clampP or self._clampWallDiv) if wall else torch.zeros_like(self.x)) + self._fluid_accel(p2)
            p1 = p2
            ks = dt * dt * self._sum(Vt[j] * ((pred[i] - pred[j]) * self.gW).sum(1))
            if wall:
                ks = ks + dt * dt * (pred * self.gk).sum(1)
            pn = p1 + cfg.omega / alpha * (src - ks)
            if clampP:
                pn = pn.clamp(min=0)
            bad = (alpha.abs() < 1e-25) | ~torch.isfinite(pn) | (pn > 1e25)
            p2 = torch.where(bad, torch.zeros_like(pn), pn)
            res = torch.where(bad, torch.zeros_like(ks), ks - src)
            err = torch.maximum(res, torch.full_like(res, -0.001)).mean()                      # omniSPH: mean(max(residual, -0.001))
            counter += 1
            if counter >= cfg.minIterations and not (float(err) > eta and counter < maxit):
                break
        pred = self._fluid_accel(p2)
        if wall and self.scene is not None:
            ab = self._boundary_accel(p2, density or cfg.divergenceClamp or self._clampWallDiv, perBody=True)                         # [B,N,2]: acceleration of the fluid by each body
            pred = pred + ab.sum(0)
            self.forcePressure = self.forcePressure - (self.V[None, :, None] * ab).sum(1)    # force of the fluid on each body (m_i = V_i, rest density 1)
        if density:
            self.wallForce = self.forcePressure.sum(0) if self.scene is not None else self.wallForce
            self.p = p2
        self.err = float(err)
        return acc + pred, counter

    def step(self):
        cfg, dt = self.cfg, self.dt
        self._bdiv = cfg.boundaryInDivergence if cfg.boundaryInDivergence is not None else self._moving()
        self._clampWallDiv = cfg.wallDivergenceClamp if cfg.wallDivergenceClamp is not None else self._bdiv
        self.forcePressure = torch.zeros((self.nb, 2), dtype=F64, device=self.dev)
        self.forceFriction = torch.zeros((self.nb, 2), dtype=F64, device=self.dev)
        self._prepare()
        i, j = self.pi, self.pj
        v0 = self.v.clone()
        self.rho = self._sum(self.V[j] * self.W) + self.lam
        g = torch.tensor(cfg.gravity, dtype=F64, device=self.dev)
        acc = g.expand(len(self.x), 2).clone()
        nd = 0
        if cfg.divergenceSolve:
            acc, nd = self._solve(acc, False)
        acc, nq = self._solve(acc, True)
        self.iters = (nd, nq)
        # XSPH (pairwise antisymmetric: conserves momentum)
        w = 2.0 * self.V[j] / (self.rho[i] + self.rho[j]) * self.W
        self.v = self.v + cfg.xsph * self._sum(w[:, None] * (self.v[j] - self.v[i]))
        # integrate; prescribed bodies move with the fluid
        self.v = self.v + dt * acc
        self.x = self.x + dt * self.v
        self.time += dt
        if self.scene is not None:
            for b in self.scene.bodies:
                b.move(dt)
        # boundary friction at the new positions, relative to the wall velocity (per body)
        if self.scene is not None and cfg.boundaryFriction > 0:
            self.ps = self._particleState(torch.ones_like(self.V))
            fused = self._fused()
            if fused:
                fw = self._aggregate(self.ps)
                lam, gk = cfg.wallMass * fw.out["lam"], cfg.wallMass * fw.out["G"]
                m1 = None
            else:
                pm = self.scene.pairMoments(self.ps, self._props(WarpOperation.Density))
                lam = cfg.wallMass * self._boundary(WarpOperation.Density, BodyField(rho=1.0), perBody=True, pm=pm, ps=self.ps)           # [B,N]
                gk = cfg.wallMass * self._boundary(WarpOperation.Gradient, BodyField(torch.tensor(1.0, dtype=F64, device=self.dev)), perBody=True, pm=pm, ps=self.ps)
            dv = torch.zeros_like(self.v)
            for bi, b in enumerate(self.scene.bodies):
                moving = float(b.angularVelocity) != 0.0 or float(b.linearVelocity.norm()) != 0.0
                vw = torch.zeros_like(self.v)
                if moving:
                    if fused:                                                          # int v_b W = v_b(x) lam + omega J m1 (m1 = int y W, the provider's first moment)
                        if m1 is None:
                            m1 = fw.evaluate((WallOutput("m1", 0, "m1"),))["m1"]
                        lb = lam[bi] / cfg.wallMass
                        num = b.velocityAt(self.x) * lb[:, None] + float(b.angularVelocity) * torch.stack([-m1[bi][:, 1], m1[bi][:, 0]], 1)
                    else:
                        f = [BodyField(torch.zeros(2, dtype=F64, device=self.dev))] * len(self.scene.bodies)
                        f[bi] = BodyField.rigid(b)
                        num = self._boundary(WarpOperation.Interpolate, f, perBody=True, pm=pm, ps=self.ps)[bi]
                    ok = lam[bi] > 1e-8 * cfg.wallMass                                 # the ratio is meaningless where lambda is round-off
                    vw = torch.where(ok[:, None], num * cfg.wallMass / lam[bi].clamp(min=1e-300)[:, None], torch.zeros_like(num))
                nrm = gk[bi].norm(dim=1, keepdim=True)
                n = -gk[bi] / nrm.clamp(min=1e-300)
                vr = self.v - vw
                tang = vr - (vr * n).sum(1, keepdim=True) * n
                fac = (cfg.boundaryFriction * lam[bi]).clamp(max=1.0)
                d = torch.where((nrm[:, 0] > 0)[:, None], -fac[:, None] * tang, torch.zeros_like(tang))
                dv = dv + d
                self.forceFriction[bi] = -(self.V[:, None] * d).sum(0) / dt            # force of the fluid on body b (omniSPH: + m fac tang / dt)
            self.v = self.v + dv
            if fused:
                self._fwNext = fw                                                      # same positions and poses: the next step's wall aggregate
            else:
                self._pmNext = pm
        # momentum bookkeeping: d(sum m v) = dt (m g - sum F_pressure - sum F_friction), exact up to round-off
        mtot = self.V.sum()
        expected = dt * (mtot * g - self.forcePressure.sum(0) - self.forceFriction.sum(0))
        self.balance = float(((self.V[:, None] * (self.v - v0)).sum(0) - expected).norm())
        if cfg.recordForces:
            self.history.append(dict(t=self.time, dt=dt, pressure=self.forcePressure.clone().cpu().numpy(), friction=self.forceFriction.clone().cpu().numpy(), balance=self.balance))
        vmax = float(self.v.norm(dim=1).max())
        self.dt = float(np.clip(cfg.cfl * float(self.h.min()) / max(vmax, 1e-12), cfg.minDt, cfg.maxDt))
        return self.time


def force_coefficients(force, rho, U, L):
    """(c_x, c_y) = F / (1/2 rho U^2 L): drag along x and lift along y of a body of reference length L (2D) in a stream U (use the rest density rho)."""
    q = 0.5 * rho * U * U * L
    return float(force[0]) / q, float(force[1]) / q


# ----------------------------------------------------------------------------------------------------------------------------- diagnostics
def diagnostics(sim: DFSPH2D, g=9.81):
    x, v, V = sim.x, sim.v, sim.V
    return dict(t=sim.time, ke=float(0.5 * (V * (v * v).sum(1)).sum()), pe=float((V * g * x[:, 1]).sum()), xmax=float(x[:, 0].max()),
                ymax=float(x[:, 1].max()), vmax=float(v.norm(dim=1).max()), rhoMax=float(sim.rho.max()), rhoMean=float(sim.rho.mean()),
                com=(float(x[:, 0].mean()), float(x[:, 1].mean())), wallForce=tuple(float(a) for a in sim.wallForce), iters=sim.iters)
