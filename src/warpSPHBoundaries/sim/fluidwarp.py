"""The fluid-fluid terms of `DeltaSPH2D` through warpSPH's modules (docs/plan-wall-evaluation.md phase 3): one `ParticleState` and one Verlet adjacency (warpSPHCore `buildVerletList`) per position set,
then `computeMomentum` (continuity), `computeScalarFieldDiffusion` (fourtakas2019 density diffusion), `computePressureSurfaceAwareWarp` (Antuono) and `computeVelocityDiffusionDeltaSPH`
(alpha viscosity) on it, all `FluidToFluid` / `AllToAll` over fluid particles only (the wall part of every term is the scene aggregate, added by the solver).

The solver's torch pair sums these replace agree with the modules to round-off in float64 (continuity 5e-13, density diffusion 3e-14, pressure 8e-11 of 1.4e2 relative to the largest value, viscosity
1e-16 after the `xi` compensation below).  Precision follows warpSPHCore (`warpSPHCore_PRECISION`): the state is cast to it, the returned accelerations to float64.

`xi`: warpSPH's `sphKernel_xi` is the constant rounded to nine digits (2.821384729 for Wendland C2 in 2D), the solver's own `KERNELS` the double-precision value; the viscosity prefactor
`alpha c0 h / xi` is made identical by passing `alpha xi_warp / xi` (a 2e-8 relative change of the dissipation would otherwise move the regression baselines).
"""
from types import SimpleNamespace

import torch
from warpSPHCore import (DomainDescription, OperationDirection, OperationProperties, ParticleState, SupportScheme, buildVerletList, sphKernel_xi)
from warpSPHCore.type_config import get_torch_precision
from dataclasses import replace

from warpSPHCore import GradientScheme, RenormalizationState, scalar_t, warpOperation
from warpSPH.modules.surfaceDetection.wp_dilate import dilateSurfaceMaskWarp
from warpSPH.modules.util import countNeighbors
from .modules.shifting import computeDeltaShiftRawWarp, computeMinNeighbourNormalDotWarp
from .modules.surfaceDetection import computeBarecascoConeCountWarp, computeBarecascoCoverWarp
from warpSPH.enumTypes import DensityDiffusionScheme, PressureForceScheme
from warpSPH.modules.deltaSPH import computeScalarFieldDiffusion
from warpSPH.modules.deltaSPH.wp_viscosityDelta import computeVelocityDiffusionDeltaSPH
from warpSPH.modules.momentum import computeMomentum
from warpSPH.modules.pressure.wp_surfaceAware import computePressureSurfaceAwareWarp
from warpSPHCore import WarpOperation

F64 = torch.float64


class FluidWarp:
    def __init__(self, sim, verletScale: float = 1.2):
        self.sim = sim
        self.tp = get_torch_precision()
        self.verletScale = float(verletScale)
        dev, H = sim.dev, sim.H
        pts = [sim.x]
        for b in (sim.scene.bodies if sim.scene is not None else []):                 # the hash grid must contain every particle the run can produce: the bodies' boxes and the initial cloud, generously padded
            lo, hi = b.obb()
            pts.append(b.pose.toWorld(torch.stack([lo, hi])))
        P = torch.cat(pts)
        lo, hi = P.amin(0) - 20.0 * H, P.amax(0) + 20.0 * H
        self.domain = DomainDescription(lo.to(self.tp), hi.to(self.tp), torch.zeros(2, dtype=torch.bool, device=dev), 2)
        self.config = SimpleNamespace(kernel=sim.cfg.kernel, domain=self.domain)
        self.xiWarp = float(sphKernel_xi(sim.cfg.kernel.value, 2))
        self._prior = None

    def state(self, x, v, rho):
        """ParticleState of the fluid at (x, v, rho) and the Verlet adjacency of its positions (the prior list is reused while the displacement allows it)."""
        s, tp = self.sim, self.tp
        ps = ParticleState(positions=x.to(tp), supports=s.Hvec.to(tp), masses=torch.full_like(rho, s.m).to(tp), kinds=s.kinds, densities=rho.to(tp))
        ps.velocities = v.to(tp)
        adj = buildVerletList(ps, self.domain, verletScale=self.verletScale, supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=self._prior, verbose=False)
        self._prior = adj
        return ps, adj

    def continuity(self, ps, adj, vel):
        """-rho div v over the fluid pairs (`computeMomentum`)."""
        ps.velocities = vel.to(self.tp)
        return computeMomentum(ps, self.config, None, adj).to(F64)

    def density_diffusion(self, ps, adj):
        """delta h c0 / xi times the fourtakas2019 divergence (fluid to fluid), the solver's xi."""
        cfg, s = self.sim.cfg, self.sim
        raw = computeScalarFieldDiffusion(ps, self.config, adj, DensityDiffusionScheme.fourtakas2019, gradField=None, gradFieldL=None, operationMode=OperationDirection.FluidToFluid,
                                          rho0=cfg.rho0, c0=cfg.c0, gravity=s.g.to(self.tp))
        return cfg.delta * s.H * cfg.c0 / s.xi * raw.to(F64)

    def pressure(self, ps, adj, P, surfaceDilated):
        """-(1/rho) sum V_j (P_j + s_i P_i) grad W with the Antuono switch s_i = +1 where P_i >= 0 or the (dilated) surface mask is set (`computePressureSurfaceAwareWarp`)."""
        from warpSPHCore import OperationProperties as OP
        out = computePressureSurfaceAwareWarp(ps, OP(kernel=self.sim.cfg.kernel, supportMode=SupportScheme.SuperSymmetric), self.domain, adjacency=adj, pressureTerm=PressureForceScheme.Antuono,
                                              queryPressures=P.to(self.tp), querySurfaceMask=surfaceDilated.to(torch.int32))
        return out.to(F64)

    def viscosity(self, ps, adj, v):
        """alpha c0 H / xi sum_j V_j / mean(rho) mu_ij grad W (all neighbours, `approachOnly=False`)."""
        cfg, s = self.sim.cfg, self.sim
        ps.velocities = v.to(self.tp)
        out = computeVelocityDiffusionDeltaSPH(ps, OperationProperties(kernel=cfg.kernel, operation=WarpOperation.Laplacian, supportMode=SupportScheme.SuperSymmetric, operationMode=OperationDirection.AllToAll),
                                               self.domain, adjacency=adj, queryVelocities=ps.velocities, inviscid=True, c_s=cfg.c0, alpha=cfg.alpha * self.xiWarp / s.xi, nu=0.0, approachOnly=False)
        return out.to(F64)

    def kernels(self, ps, adj):
        """the detector / shifting sums (`FluidSums`) of the fluid state `ps` on the adjacency `adj`."""
        return FluidSums(self, ps, adj)


class FluidSums:
    """the partial sums of the free-surface detector and of the delta+ shifting over the fluid pairs of one position set, as modules in warpSPH's layout (`sim/modules/`) and warpSPH's own modules where they exist:
    `pass1()` = dict(C: Barecasco cover vector, nAll: number of neighbours within the support, Mf: fluid renormalisation matrix sum V_j (-d) (x) grad W, raw: raw shift sum), `cone_count`, `dilate`, `lam_gradient`,
    `min_dot`.  A boundary representation adds its own part to these sums before the decisions (`DeltaSPH2D._detect_surface`, `shift`).  Float64 results whatever the precision of warpSPHCore."""

    def __init__(self, fw, ps, adj):
        self.fw, self.ps, self.adj, self.sim = fw, ps, adj, fw.sim
        self.tp = fw.tp
        self.op = OperationProperties(kernel=fw.sim.cfg.kernel, supportMode=SupportScheme.SuperSymmetric, operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive)
        self._p1 = None

    def pass1(self):
        if self._p1 is None:
            fw, ps, adj, s = self.fw, self.ps, self.adj, self.sim
            C = computeBarecascoCoverWarp(ps, self.op, fw.domain, adjacency=adj)
            nAll = countNeighbors(ps, fw.config, None, adj).to(F64) - 1.0                     # the neighbours of the list within the support, the particle itself excluded
            Mf = warpOperation(ps, replace(self.op, operation=WarpOperation.Covariance), fw.domain, adjacency=adj, covarianceReturnNumNeighbors=True)[0]
            raw = computeDeltaShiftRawWarp(ps, self.op, fw.domain, R=s.cfg.shiftR, n=4, W0=s._w0, adjacency=adj)
            self._p1 = dict(C=C.to(F64), nAll=nAll, Mf=Mf.to(F64), raw=raw.to(F64))
        return self._p1

    def cone_count(self, c, half):
        return computeBarecascoConeCountWarp(self.ps, self.op, self.fw.domain, coverAxes=c.to(self.tp).contiguous(), halfAngle=scalar_t(half), adjacency=self.adj).to(F64)

    def dilate(self, surface):
        out = dilateSurfaceMaskWarp(self.ps, replace(self.op, kernel=self.sim.cfg.kernel), freeSurfaceMask=surface.to(self.tp), domain=self.fw.domain, adjacency=self.adj)
        return out > 0.5

    def lam_gradient(self, lam):
        """sum V_j (lam_j - lam_i) grad_i W_ij (warpOperation Gradient, `Difference`, scatter)."""
        op = OperationProperties(kernel=self.sim.cfg.kernel, operation=WarpOperation.Gradient, supportMode=SupportScheme.Scatter, operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Difference)
        return warpOperation(self.ps, op, queryValues=lam.to(self.tp).contiguous(), domain=self.fw.domain, adjacency=self.adj).to(F64)

    def min_dot(self, F, nrm):
        return computeMinNeighbourNormalDotWarp(self.ps, self.op, self.fw.domain, surfaceMask=F.to(torch.int32), surfaceNormals=nrm.to(self.tp).contiguous(), adjacency=self.adj).to(F64)
