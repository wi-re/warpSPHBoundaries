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
import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation

from ..scene.implicitBodies import HalfPlaneBody
from ..scene.scene import Body, BodyField, BoxRep, ImplicitRep, Scene, SdfRep, SurfaceRep, VolumeRep, sceneOperation
from ..scene.fused import WallOutput
from .pairs import F64, dwendland2, neighbor_pairs, pair_delta, wendland2

PACKING = 0.399200743165053487            # omniSPH packing_2D (spacing / h)
TARGET_NEIGHBORS = 20


# ----------------------------------------------------------------------------------------------------------------------------- lattice calibration
def lattice_calibration(dx, dy, h, kernel="w2"):
    """Make a regular particle lattice an exact rest state of the SPH discretisation:
        S    = sum over the infinite lattice of W * dx dy                       (the discrete kernel sum, = 1 + O(1e-2) for h / dx ~ 2.5)
        V'   = dx dy / S      -> sum_j V' W = 1 in the bulk
        mu   = 1 / S          mass per area of the wall continuum (a wall filled with the same lattice)
        d_x, d_y             distance of the first lattice row from the wall face for which the first row also has density exactly 1.
        lamRow0 = lambda(d_y)          the carve threshold that puts a flat wall's first row at its rest distance (a curved or staircased surface keeps only particles at >= d_y: gaps, a volume deficit)
        lamCell = lambda(d_y - dy / 2) the volume-consistent threshold: a particle is kept iff its lattice cell starts beyond the rest state's fluid edge d_y - dy / 2, so N dx dy = the fluid area on average
                                       (the particles inside d_y are over-dense and pushed out by the positive pressure, which the p >= 0 density solve can do; gaps it can never close)
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
    return dict(S=S, V=Vp, mu=mu, dwallX=dwx, dwallY=dwy, lamRow0=lam(dwy), lamCell=lam(max(dwy - 0.5 * dy, 0.0)))


def carve(positions, bodies, h, lamMax, device, kernel=KernelFunctions.Wendland2):
    """remove the lattice particles that overlap the given bodies: those with a body kernel integral lambda > lamMax (`lattice_calibration(...)['lamRow0']`: the first row
    of a flat wall; so a carved surface keeps the same first-row distance as the flat walls).  Returns the kept positions."""
    pos = torch.as_tensor(positions, dtype=F64, device=device)
    n = len(pos)
    ps = ParticleState(positions=pos, supports=torch.full((n,), float(h), dtype=F64, device=device), masses=torch.ones(n, dtype=F64, device=device),
                       kinds=torch.zeros(n, dtype=torch.int32, device=device), densities=torch.ones(n, dtype=F64, device=device))
    sc = Scene(bodies, device)
    from ..scene.provider import AnalyticBoundary
    prov = AnalyticBoundary(sc)
    if prov.supported(fixedAdjacency=True):                                                     # the provider (the disk element exists only there: the oracle `sceneOperation` returns lambda = 0 for DiskArrayRep)
        lam = prov.aggregate(ps, float(h), kernel=kernel, lean=True).out["lam"].sum(0)
    else:
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
    densityClamp: bool = True                # p >= 0 in the density solve (omniSPH; a free surface cannot carry tension).  False: the signed pressure, for closed / periodic domains without a free surface, where
                                             # the pressure is only defined up to a constant and a flow past an obstacle needs pressures below the mean (the clamp leaves the lee side unsupported: voids)
    closedDomain: Optional[bool] = None      # no free surface (a fully periodic box, or one closed by walls): the density source's mean (1 - rho averaged) is in the null space of the pressure operator (a uniform pressure moves
                                             # nothing), the Jacobi can only pump the pressure up / down uniformly (measured: 0.3-0.4 against a Stokes pressure ~5e-3 in a periodic array), and a large uniform pressure is not
                                             # force-free next to a wall.  True: the source's mean is removed (the compatibility projection; warpSPH's closed-box finding for velocity-impulse solvers).  None: True iff fully periodic.
    closedPreset: Optional[bool] = None      # the converged-projection setup for closed / periodic flows without a free surface (docs/plan-next-steps.md "DFSPH in closed periodic domains": Stokes arrays, TGV, channel, Couette ~ 1.00
                                             # against 1.2-1.45 for the density-solve path): every field below still at its library default is replaced by CLOSED_PRESET (projection='compact', densitySolve=False,
                                             # shifting='fixed' shiftA 0.5, divergenceGauge='min'); explicitly set fields win.  None: on iff fully periodic (like closedDomain); True: force (walled closed domains, e.g.
                                             # Taylor-Couette); False: the omniSPH-style path.  Free surfaces keep the omniSPH-style path.
    divergenceWarmStart: float = 0.0         # the divergence solve starts from this fraction of the previous step's divergence pressure (omniSPH: 0, from zero; the Jacobi then cannot build a long-range pressure in 4 iterations)
    pressureGauge: str = "auto"              # closed domains: the density pressure is defined up to a constant, and the Jacobi lets that constant float up (a 1e-4 density excess -> p ~ 0.3 with dt^2 scaling); a large uniform
                                             # pressure is not force-free next to walls (measured: the slot between periodic squares stagnates as p_min lifts off 0).  'min': after the density solve p -= min(p) (the smallest
                                             # admissible background, p >= 0 kept); 'none'; 'auto': 'min' iff closedDomain.
    divergenceGauge: str = "none"            # the same for the divergence solve's pressure (the physical one under densityShift): 'none' | 'min' (p -= min p) | 'mean' (p -= mean p)
    wallPressureFactor: float = 1.0          # scales the wall's p_i term (p_i / rho_i^2 + p_i) mu grad lambda: 1 = omniSPH (pressure mirrored, 2 p at rho = 1); 0.5 = SPlisHSPlasH's (p_i / rho_i^2) grad rho_b (Akinci / density / volume maps)
    projection: str = "dfsph"                # the divergence solve: 'dfsph' (omniSPH: Jacobi on the composed operator div(grad p), a fixed iteration count; converged it is an exact discrete projection that also removes the
                                             # divergence noise of the moving particles and damps the resolved flow: TGV decay 1.02 / 1.38 x analytic at 4 / 100 iterations) | 'compact' (approximate projection, Cummins & Rudman 1999:
                                             # L p = the DFSPH divergence source / dt^2 with the compact Brookshaw / Morris Laplacian (lattice-calibrated like the Morris viscosity, mirror = Neumann walls), solved by
                                             # Jacobi-preconditioned CG to `projectionTol`; the velocity is corrected with this scheme's pressure gradient)
    projectionTol: float = 1e-8              # 'compact': relative residual of the CG
    projectionMaxIterations: int = 2000
    projectionWarmStart: bool = True         # 'compact': start the CG from the previous step's pressure (the true relative residual decides convergence, so a stale start cannot stop it early)
    projectionBlock: int = 8                 # 'compact' on the graph path: CG iterations per captured graph; the residual is read back only at the check schedule's marks (0.50 / 0.75 / 0.85 of the previous solve's
                                             # first-converged count, then every block; every block on the first solve), iterations past convergence are no-ops on the device
    densitySolve: bool = True                # False: the divergence-free projection is the only pressure solve (isolates it from the density correction, whose pressure is a position-correction noise far above a Stokes
                                             # pressure); the wall then has to be in the divergence solve (boundaryInDivergence defaults to True) and the distribution is kept by `shifting`
    densityMode: str = "summation"           # 'summation' (rho = sum V W + mu lambda each step) | 'continuity' (integrated: d rho / dt = sum V_j (v_i - v_j) . grad W_ij + (v_i - v_b) . mu grad lambda; drift-free
                                             # for a solenoidal field, decoupled from the particle arrangement)
    shifting: str = "none"                   # 'fickian': delta x = -D grad C after the advection, grad C = sum V_j grad W_ij + mu grad lambda (the wall completes the concentration, so particles relax to the equilibrium
                                             # distance from it), D = shiftA h_s |v_i| dt (Lind et al. 2012, h_s = support / 2) or 'fixed' D = shiftA h_s^2 per step; a position move only (no momentum)
    freeSurface: Optional[bool] = None       # 'compact' projection / shifting: free-surface treatment (surface particles: rho = sum V W + mu lambda < surfaceRho): P = 0 there (Dirichlet), and the shift keeps only its tangential
                                             # part in the surface layer (the surface particles and their neighbours, Lind et al. 2012).  None: on unless the domain is fully periodic
    surfaceRho: float = 0.85
    projectionGradient: str = "symmetric"     # 'compact': the velocity correction.  'symmetric': this scheme's pressure force (fluid pairs p_i / rho_i^2 + p_j / rho_j^2, the wall (p_i / rho_i^2 + p_i) mu grad lambda) with the
                                             # wall row (Robin) in the operator; 'difference': -(1 / rho_i) sum_j V_j (p_j - p_i) grad W_ij (exact for a uniform pressure; the wall acts through the Neumann
                                             # condition: the flux source and its hydrostatic extrapolation, no p_i-dependent wall force, no wall row); the body loads are the symmetric wall integral either way
    projectionWall: str = "robin"            # 'compact': the wall of the projection, 'robin' | 'mirror' (the consistent Neumann wall; free surfaces), see `_solve_compact`
    renormalisationMinDet: float = 0.1       # the renormalised gradient: det of the moment matrix below this keeps the plain sum
    projectionWallFlux: float = 1.0          # 'compact': the factor on the wall flux of the source (-dt v* . mu grad lambda): summed over the near-wall rows it captures lambda(0) ~ 1/2 of the flux into the wall; 2 = the mirror
                                             # (delta+'s continuity carries the same factor)
    projectionDensity: float = 0.0           # 'compact': + beta max(rho - 1, 0) (an over-compression, per step) in the projection source: a weak density-invariant term against the projection's volume drift (free surfaces only;
                                             # the surface fixes the pressure level, so no closed-domain drift)
    shiftA: float = 2.0
    shiftCap: float = 0.25                   # |delta x| <= shiftCap dx per step
    densityShift: bool = False               # VD+PS (Cornelis et al.): the density solve's correction moves the particles (x += dt^2 a_density) instead of changing their velocity; the velocity carries only the divergence-free
                                             # projection.  Its wall loads are then not momentum exchange and are left out of forcePressure (kept in forcePressureShift)
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
    periodic: Optional[object] = None       # `Periodic` box (sim/pairs.py): the fluid pairs take the minimum image of the raw positions (never wrapped), bodies via `Scene.setPeriodic`
    bodyForce: tuple = (0.0, 0.0)           # a uniform acceleration of the momentum equation (a periodic pressure-gradient driver); the hydrostatic wall pressure carries it like gravity
    viscosity: float = 0.0                  # kinematic viscosity nu (0: inviscid, omniSPH): the Morris operator on the fluid pairs (nu / morrisCalibration) and the shared no-slip wall closure (sim/wallclosure.py,
                                            # complement moments); the wall viscous force and torque per body are booked (incl. the Morris traction correction -2 mu Omega A); set boundaryFriction = 0 with it
    morrisCalibration: Optional[float] = None  # nu_eff / nu of the discrete Morris operator on this lattice (None: the long-wave lattice sum of the rest lattice, `morris_calibration`)
    viscousDt: float = 0.1                  # dt <= viscousDt dx^2 / nu, dx = PACKING h the spacing (explicit viscosity; the no-slip wall term is as stiff as nu / d^2 with d ~ dx / 2: the support h would be unstable)
    graphIterations: bool = True            # replay each pressure iterate as a CUDA graph (persistent per-step buffers, recaptured when a buffer changes shape, i.e. the Verlet list is rebuilt); the host keeps
                                            # omniSPH's convergence test (one scalar read per iterate); needs fluidPairs = 'verlet' and CUDA
    fluidPairs: str = "verlet"              # 'verlet': warpSPHCore's Verlet list (reused while no particle has moved beyond the margin; pairs beyond the support carry W = grad W = 0); 'cells': a fresh torch cell list every step
    verletScale: float = 1.2
    wallBackend: str = "auto"               # 'fused': the wall terms from the boundary provider (AnalyticBoundary / FusedWall: lam, grad lam, the first-moment tensor int y (x) grad W, m1 = int y W, per body; one
                                            # evaluation per position set, the iterates contract the stored tensor); 'scene': the oracle `sceneOperation`; 'auto': fused when every body is supported (surface, box,
                                            # disks, implicit / SDF lowered), else scene (VolumeRep: the omniSPH slabs)


def morris_calibration(h, dx, V, kernel="w2", eta2=0.0025):
    """nu_eff / nu of the Morris operator a_i = sum_j V_j nu (rho_i + rho_j) / rho_i K_ij (v_i - v_j), K = (x_ij . grad W) / (r^2 + eta2 h^2), on a square lattice of spacing dx with particle volume V at rest density 1:
    for a shear field the long-wave rate is -nu (sum_j 2 V K_ij y_j^2) / 2 per unit u'', so the factor is -sum_j V K_ij y_j^2 (the continuum value with eta -> 0 is 1)."""
    m = int(math.ceil(h / dx)) + 1
    i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    r = np.hypot(i * dx, j * dx).ravel()
    y = (j * dx).ravel()
    keep = (r > 1e-14) & (r < h)
    r, y = r[keep], y[keep]
    dw = dwendland2(torch.as_tensor(r), h).numpy()
    K = dw * r / (r * r + eta2 * h * h)
    return float(-np.sum(V * K * y * y))


CLOSED_PRESET = dict(projection="compact", densitySolve=False, shifting="fixed", shiftA=0.5, divergenceGauge="min")


def resolve_closed_preset(cfg):
    """`DFSPHConfig.closedPreset`: the fields of CLOSED_PRESET that are still at their library default take the preset's value (an explicit value, even the default one set deliberately to
    differ from the preset, is indistinguishable from the default: use closedPreset=False then)."""
    on = cfg.closedPreset
    if on is None:
        on = cfg.periodic is not None and all(cfg.periodic.flags)
    if not on:
        return cfg
    dflt = DFSPHConfig()
    return replace(cfg, **{k: v for k, v in CLOSED_PRESET.items() if getattr(cfg, k) == getattr(dflt, k)})


class DFSPH2D:
    def __init__(self, positions, velocities, V, h, scene: Optional[Scene], cfg: Optional[DFSPHConfig] = None, device="cuda:0"):
        self.cfg = resolve_closed_preset(cfg or DFSPHConfig())
        self.dev = device
        t = lambda a: torch.as_tensor(a, dtype=F64, device=device)
        self.x, self.v = t(positions).clone(), t(velocities).clone()
        n = len(self.x)
        self.V = t(V) if np.ndim(V) else torch.full((n,), float(V), dtype=F64, device=device)
        self.h = t(h) if np.ndim(h) else torch.full((n,), float(h), dtype=F64, device=device)
        self.scene = scene
        self.p = torch.zeros(n, dtype=F64, device=device)               # fluidPriorPressure
        self.pDiv = torch.zeros(n, dtype=F64, device=device)            # the divergence solve's pressure of the last step
        self.rho = torch.ones(n, dtype=F64, device=device)
        self.dt = self.cfg.maxDt
        self.time = 0.0
        self.iters = (0, 0)
        self.wallForce = torch.zeros(2, dtype=F64, device=device)
        nb = len(scene.bodies) if scene is not None else 0
        self.nb = nb
        if scene is not None and self.cfg.periodic is not None:
            scene.setPeriodic(self.cfg.periodic, float(self.h.max()))
        self.forcePressure = torch.zeros((nb, 2), dtype=F64, device=device)    # force of the fluid on each body, pressure part (this step)
        self.forceFriction = torch.zeros((nb, 2), dtype=F64, device=device)    # friction (boundary viscosity) part
        self.history = []                                                       # per step: dict(t, dt, pressure [B,2], friction [B,2], balance)
        self.balance = 0.0
        self._pmNext = None
        self._fwNext = None
        self.stats = {}
        self.kinds = torch.zeros(n, dtype=torch.int32, device=device)

    # ---- geometry-dependent data, valid for the current positions
    def _delta(self, x, i, j):
        """x_i - x_j of the pairs, the minimum image on the periodic axes (the positions are raw, never wrapped)."""
        return pair_delta(x, i, j, self.cfg.periodic)

    def _pairs(self):
        """(i, j, r) of the fluid pairs at the current positions: the Verlet list (fixed arrays while it stays valid) or a fresh cell list."""
        x, h = self.x, self.h
        if self.cfg.fluidPairs == "cells":
            return neighbor_pairs(x, h, self.cfg.periodic)
        import warp as wp
        from warpSPHCore import DomainDescription, SupportScheme, buildVerletList
        wp.init()                                                                               # idempotent; without walls nothing else initialises Warp before the Verlet search
        if getattr(self, "_domain", None) is None:
            pts = [x]
            for b in (self.scene.bodies if self.scene is not None else []):
                lo, hi = b.obb()
                pts.append(b.pose.toWorld(torch.stack([lo, hi])))
            P = torch.cat(pts)
            H = float(h.max())
            lo, hi = P.amin(0) - 20.0 * H, P.amax(0) + 20.0 * H
            flags = torch.zeros(2, dtype=torch.bool, device=self.dev)
            per = self.cfg.periodic
            if per is not None:
                per.checkSupport(H)
                for a, f in enumerate(per.flags):
                    if f:
                        lo[a], hi[a], flags[a] = per.lo[a], per.hi[a], True
            self._domain = DomainDescription(lo, hi, flags, 2)
            self._prior = None
        ps = ParticleState(positions=x, supports=h, masses=self.V, kinds=self.kinds, densities=torch.ones_like(self.V))
        adj = buildVerletList(ps, self._domain, verletScale=self.cfg.verletScale, supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=self._prior, verbose=False)
        self._prior = adj
        self._csr = (adj.edgeOffsets, adj.numNeighbors)                                         # CSR rows of the pairs (i sorted): the fused iterate kernels loop over them
        i, j = adj.i.long(), adj.j.long()
        r = self._delta(x, i, j).norm(dim=1)
        return i, j, r

    def _prepare(self):
        x, h = self.x, self.h
        i, j, r = self._pairs()
        hij = 0.5 * (h[i] + h[j])
        self.pi, self.pj = i, j
        self.W = wendland2(r, hij)
        d = self._delta(x, i, j)
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
        # moment matrix of the pressure reconstruction (only the MLS wall pressure uses it)
        if self.cfg.wallPressure != "linear":
            self.Minv = self.wy = None
            return
        w = self.V[j] * self.W
        y = -self._delta(x, i, j)
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
        return self._provider.aggregate(ps, float(self.h.max()), kernel=self.cfg.kernel, lean=True)

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
        pfac = (pp / self.rho ** 2 + pp) * self.sClose * self.cfg.wallPressureFactor
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
        if getattr(self, "_gT", None) is None:
            self._gT = torch.tensor([self.cfg.gravity[0] + self.cfg.bodyForce[0], self.cfg.gravity[1] + self.cfg.bodyForce[1]], dtype=F64, device=self.dev)   # g + f (dp/dn = rho (g + f - a_wall) . n); made once (graph capture)
        g = self._gT[None]
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
        if density and self._closed():
            s = s - (self.V * s).sum() / self.V.sum()
        return s

    def _solve_compact(self, acc):
        """the approximate projection (sim/projection.py): sum_j w_ij (p_i - p_j) [+ D_i p_i] = Vt_i s_i, w the compact Laplacian weights times dt^2 (the left side ~ Vt dt^2 lap p, as the DFSPH operator), s the DFSPH
        divergence source of v* = v + dt acc; the shared CG to projectionTol; then v -= dt grad p / rho.  The wall (`projectionWall`):

          'robin'   v* carries the pressure-independent wall part (the hydrostatic extrapolation), the source the wall flux (x projectionWallFlux), the operator the wall row D (the wall force of the particle's own
                    pressure moves its own flux), the correction is this scheme's pressure force (`projectionGradient`).  Closed domains (the periodic Stokes arrays).
          'mirror'  the consistent Neumann wall of a mirror (ghost) continuation: the velocity mirrored (odd normal part) -> the wall flux of the source twice the integral term, the pressure mirrored (even) -> the
                    walls add nothing to the compact Laplacian (Neumann), the correction is the renormalised fluid gradient (exact for a linear p with a one-sided support; the mirrored p adds nothing to it),
                    no pressure push in v* (the hydrostatic pressure is produced by the Neumann condition from v* = v + dt g).  Free surfaces: P = 0 on the surface particles (Dirichlet).
        The body loads are the wall integral of the solved pressure."""
        from .projection import CompactProjection, compact_weights
        from .wallmoments import MORRIS_ETA2
        cfg, dt = self.cfg, self.dt
        mirror = cfg.projectionWall == "mirror"
        walls = self.scene is not None and self.nb > 0
        Vt = self.V / self.rho
        ab0 = None
        if walls and not mirror:                                                                    # the pressure-independent wall part (the hydrostatic extrapolation rho (g - a_w) . y): known before the solve, so it goes into v*
            ab0 = self._boundary_accel(torch.zeros_like(self.V), False, perBody=True)
            acc = acc + ab0.sum(0)
        vp = self.v + dt * acc
        src = self._source(dt, Vt, vp, False, True)
        flux = 2.0 if mirror else cfg.projectionWallFlux
        if flux != 1.0 and walls:                                                                   # (flux - 1) more of the wall term -dt (v* . mu grad lambda - int v_b . grad W)
            extra = -dt * (vp * self.gk).sum(1)
            if self.wallDiv is not None:
                extra = extra + dt * self.wallDiv
            src = src + (flux - 1.0) * extra
        if cfg.projectionDensity > 0.0:
            src = src - cfg.projectionDensity * (self.rho - 1.0).clamp(min=0.0)
        rhs = Vt * src
        D = None
        if walls and not mirror and cfg.projectionGradient == "symmetric":                         # the wall row: the source's wall term dt^2 a . gk of the wall force a = -(p / rho^2 + p) gk
            D = -Vt * dt * dt * (1.0 / self.rho ** 2 + 1.0) * (self.gk * self.gk).sum(1)
        surf = self._surface()
        if self._closed() and D is None and surf is None:
            rhs = rhs - rhs.mean()
        off, cnt = self._csr[0], self._csr[1]
        P = int(cnt.sum())
        i, j = self.pi[:P], self.pj[:P]
        w = compact_weights(self._delta(self.x, i, j), self.gW[:P], Vt[i], Vt[j], self.h[i], self._morris_cal(), MORRIS_ETA2) * dt * dt
        if getattr(self, "_proj", None) is None:
            self._proj = CompactProjection(self.dev, block=cfg.projectionBlock, warmStart=cfg.projectionWarmStart, graphs=self._graphs_on())
        p, it, relres = self._proj.solve(off, cnt, j, w, rhs, cfg.projectionTol, cfg.projectionMaxIterations, diag=D, dirichlet=surf)
        p = self._gauge(p, False) if surf is None else p
        self.pDiv = p
        grad = cfg.projectionGradient if (cfg.projectionGradient != "symmetric" or not mirror) else "renormalised"
        if grad in ("difference", "renormalised"):
            i, j = self.pi, self.pj
            gp = self._sum((self.V[j] * (p[j] - p[i]))[:, None] * self.gW)
            if grad == "renormalised":                                                              # L_i = [sum_j V_j grad W_ij (x) (x_j - x_i)]^-1: exact for a linear p with a one-sided support (surface, corners, walls)
                Mx = torch.zeros((len(p), 2, 2), dtype=F64, device=self.dev).index_add_(0, i, self.V[j][:, None, None] * self.gW[:, :, None] * (-self._delta(self.x, i, j))[:, None, :])
                det = Mx[:, 0, 0] * Mx[:, 1, 1] - Mx[:, 0, 1] * Mx[:, 1, 0]
                ok = det.abs() > cfg.renormalisationMinDet                                         # ~1 in the bulk; a degenerate (one-sided, isolated) neighbourhood keeps the plain sum
                Linv = torch.stack([torch.stack([Mx[:, 1, 1], -Mx[:, 0, 1]], 1), torch.stack([-Mx[:, 1, 0], Mx[:, 0, 0]], 1)], 1) / torch.where(ok, det, torch.ones_like(det))[:, None, None]
                gp = torch.where(ok[:, None], torch.einsum("nab,nb->na", Linv.transpose(1, 2), gp), gp)
            pred = -gp / self.rho[:, None]
        else:
            pred = self._fluid_accel(p)
        if walls:
            if mirror:                                                                              # the load of the mirrored pressure p_b = p_i: the wall integral (p_i / rho_i^2 + p_i) mu grad lambda per body
                ab = -((p / self.rho ** 2 + p) * self.sClose)[None, :, None] * self.gkb
            else:
                ab = self._boundary_accel(p, False, perBody=True)
                if grad == "symmetric":
                    pred = pred + (ab - ab0).sum(0)                                                 # ab0 is already in acc
            self.forcePressure = self.forcePressure - (self.V[None, :, None] * ab).sum(1)
            self.forcePressureDiv = -(self.V[None, :, None] * ab).sum(1)
        self.err = relres
        return acc + pred, it

    def _free_surface_on(self):
        fs = self.cfg.freeSurface
        if fs is None:
            fs = not (self.cfg.periodic is not None and all(self.cfg.periodic.flags))
        return bool(fs)

    def _surface(self):
        """the free-surface particles (rho = sum V W + mu lambda < surfaceRho, the summation density of this step), or None without a free-surface treatment."""
        if not self._free_surface_on():
            return None
        rs = getattr(self, "rhoSum", None)
        rs = rs if rs is not None else self.rho
        return rs < self.cfg.surfaceRho

    def _gauge(self, p, density=True):
        g = self.cfg.pressureGauge if density else self.cfg.divergenceGauge
        if g == "auto":
            g = "min" if self._closed() else "none"
        if g == "min":
            return p - p.min()
        if g == "mean":
            return p - (self.V * p).sum() / self.V.sum()
        return p

    def _closed(self):
        c = self.cfg.closedDomain
        if c is None:
            c = self.cfg.periodic is not None and all(self.cfg.periodic.flags)
        return bool(c)

    # ---- viscosity: Morris on the fluid pairs, the shared no-slip wall closure
    def _morris_cal(self):
        if self.cfg.morrisCalibration is not None:
            return float(self.cfg.morrisCalibration)
        if getattr(self, "_calCache", None) is None:
            h = float(self.h.median())
            self._calCache = morris_calibration(h, PACKING * h, float(self.V.median()))
        return self._calCache

    def _viscous_accel(self):
        """the viscous acceleration (fluid Morris pairs + the no-slip wall closure per body), with the wall's viscous force and torque on each body booked (-sum V a_wall; torque with the lever at the particle,
        plus the Morris traction correction -2 nu Omega A of a rotating body, rest density 1)."""
        from .wallclosure import NoSlipClosure
        from .wallmoments import MORRIS_ETA2
        cfg = self.cfg
        cal = self._morris_cal()
        nu_used = cfg.viscosity / cal
        i, j = self.pi, self.pj
        x, v, rho = self.x, self.v, self.rho
        d = self._delta(x, i, j)
        r = d.norm(dim=1)
        K = (d * self.gW).sum(1) / (r * r + MORRIS_ETA2 * self.h[i] ** 2)
        Vt = self.V / rho
        viscf = self._sum((Vt[j] * nu_used * (rho[i] + rho[j]) / rho[i] * K)[:, None] * (v[i] - v[j]))
        if self.scene is None or self.nb == 0:
            return viscf
        if getattr(self, "_wcObj", None) is None:
            self._wcObj = NoSlipClosure(self.scene, dwendland2, float(self.h.max()), PACKING * float(self.h.median()), self.dev, morris=True, cal=cal, wallMass=1.0, complement=True, sums=self._complement_sums)
        wc = self._wcObj
        acc = viscf.clone()
        H = float(self.h.max())
        for bi, b in enumerate(self.scene.bodies):
            dsd, nsd, hit = self.scene.signed_distance(x, body=bi, supportMax=H)
            dd = dsd.clamp(min=0.25 * PACKING * float(self.h.median()))
            on = (hit & (dsd < H))[:, None]
            term = torch.where(on, wc.term(bi, x, v - b.velocityAt(x), nsd, dd, viscf, rho, 8.0 * cfg.viscosity, coverage=getattr(self, "rhoSum", None) if getattr(self, "rhoSum", None) is not None else rho), torch.zeros_like(v))
            acc = acc + term
            F = -(self.V[:, None] * term)
            self.forceViscous[bi] = F.sum(0)
            lev = b.relative(x) if hasattr(b, "relative") else x - b.center
            tq = (lev[:, 0] * F[:, 1] - lev[:, 1] * F[:, 0]).sum()
            self.torque[2, bi] = tq - 2.0 * cfg.viscosity * NoSlipClosure.solid_area(b) * float(b.angularVelocity)
        return acc

    def _complement_sums(self, x, n, d, rho):
        """the discrete fluid moments of the Morris weight in the frame (n_i, d_i) of each particle's wall, over this step's fluid pairs (padding pairs add zero)."""
        from .wallmoments import MORRIS_ETA2
        i, j = self.pi, self.pj
        y = -self._delta(x, i, j)
        r = y.norm(dim=1)
        K = -(y * self.gW).sum(1) / (r * r + MORRIS_ETA2 * self.h[i] ** 2)                      # (x_ij . grad W_ij) / (r^2 + eta^2 h^2), x_ij = -y
        K = torch.where(i != j, K, torch.zeros_like(K))
        Vt = self.V / rho
        w = Vt[j] * (rho[i] + rho[j]) / (2.0 * rho[i]) * K
        st = (y * n[i]).sum(1) + d[i]
        S0 = self._sum(w)
        S1 = self._sum(w * st)
        S2 = self._sum(w * st * st)
        SM = self._sum(w[:, None] * y)
        return S0, S1, S2, SM

    # ---- CUDA-graph replay of the pressure iterates
    def _graphs_on(self):
        ok = bool(self.cfg.graphIterations and self.cfg.fluidPairs == "verlet" and str(self.dev).startswith("cuda") and torch.cuda.is_available())
        if ok and self.cfg.wallPressure == "linear" and self.scene is not None and not self._fused():
            ok = False                                                                          # the MLS wall term per iterate is an oracle scene operation there (not capturable)
        return ok

    def _pin(self, name, t):
        """copy `t` into the persistent buffer `name` (same shape and dtype: in place; else a new buffer, and the captured graphs are dropped): the tensors a captured iterate reads must not move."""
        st = self.__dict__.setdefault("_static", {})
        buf = st.get(name)
        if buf is None or buf.shape != t.shape or buf.dtype != t.dtype:
            st[name] = t.clone()
            self.__dict__.setdefault("_graphs", {}).clear()
            return st[name]
        buf.copy_(t)
        return buf

    def _pin_pairs(self):
        """the pair arrays (pi, pj, gW, W, and the MLS weights wy of the 'linear' wall pressure) in persistent buffers of a fixed CAPACITY (padding: pair (0, 0) with W = grad W = 0, which adds exactly zero): a Verlet rebuild changes the number of pairs, the
        capacity keeps the buffers (and the captured graphs) unless it is exceeded (then 25 % headroom on the new count)."""
        st = self.__dict__.setdefault("_static", {})
        P = len(self.pi)
        cap = st["pi"].shape[0] if "pi" in st else 0
        if P > cap:
            cap = int(math.ceil(1.25 * P / 1024.0) * 1024)
            st["pi"] = torch.zeros(cap, dtype=self.pi.dtype, device=self.dev)
            st["pj"] = torch.zeros(cap, dtype=self.pj.dtype, device=self.dev)
            st["gW"] = torch.zeros((cap, 2), dtype=F64, device=self.dev)
            st["W"] = torch.zeros(cap, dtype=F64, device=self.dev)
            st["wy"] = torch.zeros((cap, 2), dtype=F64, device=self.dev)
            self.__dict__.setdefault("_graphs", {}).clear()
        names = ("pi", "pj", "gW", "W") + (("wy",) if getattr(self, "wy", None) is not None else ())
        for name in names:
            buf = st[name]
            buf[:P].copy_(getattr(self, name))
            buf[P:].zero_()
            setattr(self, name, buf)

    def _pin_step(self):
        """the per-step tensors of the iterates in persistent buffers (called after `rho` is set)."""
        self._pin_pairs()
        for name in ("rho", "V"):
            setattr(self, name, self._pin(name, getattr(self, name)))
        self._off = self._pin("off", self._csr[0].to(torch.int32))
        self._cnt = self._pin("cnt", self._csr[1].to(torch.int32))
        if self.scene is not None:
            for name in ("gk", "gkb", "sClose"):
                setattr(self, name, self._pin(name, getattr(self, name)))
            if self.cfg.wallPressure == "hydrostatic":
                pp0 = torch.zeros_like(self.V)
                self._out1 = None
                self._out1 = self._pin("out1", self._a1_part(pp0))
                if self.cfg.clampWallPressure:
                    self._theta_q = None
                    self._wall_excess(pp0)
                    self._theta_q = self._pin("theta_q", self._theta_q)
            elif self.cfg.wallPressure == "linear":
                self.Minv = self._pin("Minv", self.Minv)                                          # wy: with the pair buffers
                if self._fused():
                    self.covb = self._pin("covb", self.covb)

    def _iterate(self, B, density, wall, clampP, clampWall):
        """one omniSPH pressure iterate on the persistent buffers B (p2 updated in place, err written): two fused Warp kernels (dfsph_kernels.py) for the hydrostatic / mirror wall pressure, the torch
        expressions for the MLS ('linear') one."""
        if self.cfg.wallPressure != "linear":
            return self._iterate_fused(B, wall, clampP, clampWall)
        i, j = self.pi, self.pj
        p2 = B["p2"]
        pred = (self._boundary_accel(p2, clampWall) if wall else torch.zeros_like(self.x)) + self._fluid_accel(p2)
        ks = B["dt2"] * self._sum(B["Vt"][j] * ((pred[i] - pred[j]) * self.gW).sum(1))
        if wall:
            ks = ks + B["dt2"] * (pred * self.gk).sum(1)
        pn = p2 + self.cfg.omega / B["alpha"] * (B["src"] - ks)
        if clampP:
            pn = pn.clamp(min=0)
        bad = (B["alpha"].abs() < 1e-25) | ~torch.isfinite(pn) | (pn > 1e25)
        p2.copy_(torch.where(bad, torch.zeros_like(pn), pn))
        res = torch.where(bad, torch.zeros_like(ks), ks - B["src"])
        B["err"].copy_(torch.maximum(res, torch.full_like(res, -0.001)).mean())

    def _iterate_fused(self, B, wall, clampP, clampWall):
        import warp as wp
        from .dfsph_kernels import dfsph_accel_kernel, dfsph_update_kernel
        N = len(self.x)
        if "acc" not in B:
            B["acc"] = self._pin("acc_it", torch.zeros((N, 2), dtype=F64, device=self.dev))
            B["res"] = self._pin("res_it", torch.zeros(N, dtype=F64, device=self.dev))
        f1 = lambda t: wp.from_torch(t.reshape(-1).contiguous(), dtype=wp.float64)
        f2 = lambda t: wp.from_torch(t.reshape(-1, 2).contiguous(), dtype=wp.vec2d)
        hasWall = wall and self.scene is not None and self.nb > 0
        if hasWall:
            gkb, out1 = self.gkb, (self._out1 if self._out1 is not None else torch.zeros_like(self.gkb))
            excess = bool(self.cfg.clampWallPressure and self.cfg.wallPressure == "hydrostatic")
            q = self._theta_q if excess else torch.zeros(self.gkb.shape[:2], dtype=F64, device=self.dev)
            nb = self.nb
        else:
            gkb = out1 = torch.zeros((1, 1, 2), dtype=F64, device=self.dev)
            q = torch.zeros((1, 1), dtype=F64, device=self.dev)
            excess, nb = False, 0
        gk = self.gk if hasWall else torch.zeros((N, 2), dtype=F64, device=self.dev)
        sC = self.sClose * self.cfg.wallPressureFactor if hasWall else torch.ones(N, dtype=F64, device=self.dev)
        import contextlib
        ctx = contextlib.nullcontext() if torch.cuda.is_current_stream_capturing() else wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream()))   # inside a capture the Warp stream is set by the caller
        with ctx:
            wp.launch(dfsph_accel_kernel, dim=N, inputs=[wp.from_torch(self._off, dtype=wp.int32), wp.from_torch(self._cnt, dtype=wp.int32), wp.from_torch(self.pj, dtype=wp.int64), f2(self.gW),
                                                         f1(self.V), f1(self.rho), f1(B["p2"]), int(hasWall), int(clampWall), int(excess), f1(sC), f2(gkb), f2(out1), f1(q), nb, N, f2(B["acc"])])
            wp.launch(dfsph_update_kernel, dim=N, inputs=[wp.from_torch(self._off, dtype=wp.int32), wp.from_torch(self._cnt, dtype=wp.int32), wp.from_torch(self.pj, dtype=wp.int64), f2(self.gW),
                                                          f1(B["Vt"]), f2(B["acc"]), f2(gk), int(hasWall), f1(B["dt2"].reshape(1)), float(self.cfg.omega), f1(B["alpha"]), f1(B["src"]), int(clampP),
                                                          f1(B["p2"]), f1(B["res"])])
        B["err"].copy_(torch.maximum(B["res"], torch.full_like(B["res"], -0.001)).mean())

    def _solve(self, acc, density):
        if self._graphs_on():
            return self._solve_graphed(acc, density)
        return self._solve_eager(acc, density)

    def _solve_graphed(self, acc, density):
        cfg, dt = self.cfg, self.dt
        Vt = self.V / self.rho
        vp = self.v + dt * acc
        wall = bool(density or self._bdiv)
        clampP = bool((density and cfg.densityClamp) or (not density and cfg.divergenceClamp))
        clampWall = bool(clampP or self._clampWallDiv)
        B = {"Vt": self._pin("Vt", Vt), "alpha": self._pin("alpha", self._alpha(dt, Vt, wall)), "src": self._pin("src", self._source(dt, Vt, vp, density, wall)),
             "p2": self._pin("p2", 0.5 * self.p if density else self.cfg.divergenceWarmStart * self.pDiv), "dt2": self._pin("dt2", torch.tensor(dt * dt, dtype=F64, device=self.dev)),
             "err": self._pin("err", torch.zeros((), dtype=F64, device=self.dev))}
        graphs = self.__dict__.setdefault("_graphs", {})
        key = (bool(density), wall, clampP, clampWall)
        g = graphs.get(key)
        if g is None:
            p_init = B["p2"].clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):                                                       # warm-up (lazy initialisations) on a side stream, then restore the iterate
                self._iterate(B, density, wall, clampP, clampWall)
            torch.cuda.current_stream().wait_stream(side)
            B["p2"].copy_(p_init)
            import warp as wp
            g = torch.cuda.CUDAGraph()
            cap = torch.cuda.Stream()
            cap.wait_stream(torch.cuda.current_stream())
            with wp.ScopedStream(wp.stream_from_torch(cap)), torch.cuda.graph(g, stream=cap, capture_error_mode="thread_local"):
                self._iterate(B, density, wall, clampP, clampWall)
            torch.cuda.current_stream().wait_stream(cap)
            graphs[key] = g
            self.stats["graphCaptures"] = self.stats.get("graphCaptures", 0) + 1
        eta = cfg.densityEta if density else cfg.divergenceEta
        maxit = cfg.maxIterations if density else cfg.divergenceMaxIterations
        counter = 0
        while True:
            g.replay()
            counter += 1
            if counter >= cfg.minIterations and not (float(B["err"]) > eta and counter < maxit):
                break
        p2 = B["p2"].clone()
        p2 = self._gauge(p2, density)
        pred = self._fluid_accel(p2)
        if wall and self.scene is not None:
            ab = self._boundary_accel(p2, clampWall, perBody=True)
            pred = pred + ab.sum(0)
            self.forcePressure = self.forcePressure - (self.V[None, :, None] * ab).sum(1)
            if not density:
                self.forcePressureDiv = -(self.V[None, :, None] * ab).sum(1)                   # the divergence solve's share (the physical pressure; the density solve's is a position correction)
        if density:
            self.wallForce = self.forcePressure.sum(0) if self.scene is not None else self.wallForce
            self.p = p2
        else:
            self.pDiv = p2
        self.err = float(B["err"])
        return acc + pred, counter

    def _solve_eager(self, acc, density):
        cfg, dt = self.cfg, self.dt
        i, j = self.pi, self.pj
        Vt = self.V / self.rho
        vp = self.v + dt * acc
        wall = density or self._bdiv
        alpha = self._alpha(dt, Vt, wall)
        src = self._source(dt, Vt, vp, density, wall)
        p2 = 0.5 * self.p if density else self.cfg.divergenceWarmStart * self.pDiv
        p1 = p2.clone()
        eta = cfg.densityEta if density else cfg.divergenceEta
        maxit = cfg.maxIterations if density else cfg.divergenceMaxIterations
        counter = 0
        while True:
            clampP = (density and cfg.densityClamp) or (not density and cfg.divergenceClamp)
            pred = (self._boundary_accel(p2, clampP or self._clampWallDiv) if wall else torch.zeros_like(self.x)) + self._fluid_accel(p2)
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
        p2 = self._gauge(p2, density)
        pred = self._fluid_accel(p2)
        if wall and self.scene is not None:
            ab = self._boundary_accel(p2, clampP or self._clampWallDiv, perBody=True)                         # [B,N,2]: acceleration of the fluid by each body
            pred = pred + ab.sum(0)
            self.forcePressure = self.forcePressure - (self.V[None, :, None] * ab).sum(1)    # force of the fluid on each body (m_i = V_i, rest density 1)
            if not density:
                self.forcePressureDiv = -(self.V[None, :, None] * ab).sum(1)                   # the divergence solve's share (the physical pressure; the density solve's is a position correction)
        if density:
            self.wallForce = self.forcePressure.sum(0) if self.scene is not None else self.wallForce
            self.p = p2
        else:
            self.pDiv = p2
        self.err = float(err)
        return acc + pred, counter

    def step(self):
        cfg, dt = self.cfg, self.dt
        self._bdiv = cfg.boundaryInDivergence if cfg.boundaryInDivergence is not None else (self._moving() or not cfg.densitySolve)
        self._clampWallDiv = cfg.wallDivergenceClamp if cfg.wallDivergenceClamp is not None else self._bdiv
        self.forcePressure = torch.zeros((self.nb, 2), dtype=F64, device=self.dev)
        self.forcePressureDiv = torch.zeros((self.nb, 2), dtype=F64, device=self.dev)
        self.forceFriction = torch.zeros((self.nb, 2), dtype=F64, device=self.dev)
        self._prepare()
        i, j = self.pi, self.pj
        v0 = self.v.clone()
        self.rho = self._sum(self.V[j] * self.W) + self.lam
        if cfg.densityMode == "continuity":
            if getattr(self, "_rhoC", None) is None:
                self._rhoC = self.rho.clone()
            self.rhoSum = self.rho
            self.rho = self._rhoC
        if self._graphs_on():
            self._pin_step()
            i, j = self.pi, self.pj
        g = torch.tensor([cfg.gravity[0] + cfg.bodyForce[0], cfg.gravity[1] + cfg.bodyForce[1]], dtype=F64, device=self.dev)      # gravity + the body force (both per unit mass)
        acc = g.expand(len(self.x), 2).clone()
        self.forceViscous = torch.zeros((self.nb, 2), dtype=F64, device=self.dev)
        self.torque = torch.zeros((3, self.nb), dtype=F64, device=self.dev)                     # (pressure, friction, viscous) torque of the fluid on each body about its centre (lever at the particle)
        if cfg.viscosity > 0.0:
            acc = acc + self._viscous_accel()
        nd = 0
        if cfg.divergenceSolve:
            acc, nd = self._solve_compact(acc) if cfg.projection == "compact" else self._solve(acc, False)
        accDiv, fpDiv = acc, self.forcePressure.clone()
        nq = 0
        if cfg.densitySolve:
            acc, nq = self._solve(acc, True)
        self.iters = (nd, nq)
        shift = None
        if cfg.densityShift:
            shift = dt * dt * (acc - accDiv)
            acc = accDiv
            self.forcePressureShift = self.forcePressure - fpDiv
            self.forcePressure = fpDiv
        # XSPH (pairwise antisymmetric: conserves momentum)
        w = 2.0 * self.V[j] / (self.rho[i] + self.rho[j]) * self.W
        self.v = self.v + cfg.xsph * self._sum(w[:, None] * (self.v[j] - self.v[i]))
        # integrate; prescribed bodies move with the fluid
        self.v = self.v + dt * acc
        if cfg.densityMode == "continuity":                                                   # with the new velocity on this step's pairs (symplectic Euler)
            drho = self._sum((self.V[j][:, None] * (self.v[i] - self.v[j]) * self.gW).sum(1))
            if self.scene is not None and self.nb > 0:
                for bi, b in enumerate(self.scene.bodies):
                    drho = drho + ((self.v - b.velocityAt(self.x)) * self.gkb[bi]).sum(1)
            self._rhoC = self.rho + dt * drho
        if cfg.shifting != "none":
            gC = self._sum(self.V[j][:, None] * self.gW) + (self.gkb.sum(0) if self.nb > 0 else 0.0)
            hs = 0.5 * self.h
            D = cfg.shiftA * hs * self.v.norm(dim=1) * dt if cfg.shifting == "fickian" else cfg.shiftA * hs * hs * torch.ones_like(self.V)
            dxs = -D[:, None] * gC
            surf = self._surface()
            if surf is not None:                                                                    # Lind et al. 2012: in the surface layer (surface particles and their neighbours) only the tangential part
                layer = self._sum(surf[j].to(F64)) > 0.5
                nrm_ = gC / gC.norm(dim=1, keepdim=True).clamp(min=1e-300)
                dxs = torch.where(layer[:, None], dxs - (dxs * nrm_).sum(1, keepdim=True) * nrm_, dxs)
            cap = cfg.shiftCap * PACKING * self.h
            nrm = dxs.norm(dim=1)
            dxs = dxs * torch.where(nrm > cap, cap / nrm.clamp(min=1e-300), torch.ones_like(nrm))[:, None]
            shift = dxs if shift is None else shift + dxs
            self.lastShift = dxs
        self.x = self.x + dt * self.v
        if shift is not None:
            self.x = self.x + shift
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
        expected = dt * (mtot * g - self.forcePressure.sum(0) - self.forceFriction.sum(0) - self.forceViscous.sum(0))
        self.balance = float(((self.V[:, None] * (self.v - v0)).sum(0) - expected).norm())
        if cfg.recordForces:
            self.history.append(dict(t=self.time, dt=dt, pressure=self.forcePressure.clone().cpu().numpy(), pressureDiv=self.forcePressureDiv.clone().cpu().numpy(), friction=self.forceFriction.clone().cpu().numpy(),
                                     viscous=self.forceViscous.clone().cpu().numpy(), torqueViscous=self.torque[2].clone().cpu().numpy(), balance=self.balance))
        vmax = float(self.v.norm(dim=1).max())
        self.dt = float(np.clip(cfg.cfl * float(self.h.min()) / max(vmax, 1e-12), cfg.minDt, cfg.maxDt))
        if cfg.viscosity > 0.0:
            self.dt = min(self.dt, cfg.viscousDt * (PACKING * float(self.h.min())) ** 2 / cfg.viscosity)   # the spacing, not the support: the no-slip wall term acts over the wall distance (~dx / 2)
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
