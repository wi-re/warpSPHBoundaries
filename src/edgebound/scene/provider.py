"""The analytic boundary as a PROVIDER for an SPH scheme (docs/audit-warpsph-boundary-hooks.md; the future `warpSPHBoundaries` public surface).

A scheme (the warpSPH weakly-compressible delta+ scheme, or `sim.DeltaSPH2D` here) asks a provider for the geometry of its bodies at the fluid particles and keeps the boundary-condition physics (pressure
extrapolation and clamp, free-slip mirror, no-slip flux, the no-penetration law, particle shifting) to itself.  Nothing is called when there are no analytic bodies.  The contract, structural (`BoundaryProvider`):

    aggregate(ps, support, ...)   the integrals of every body at the particle positions of `ps` (a warpSPHCore `ParticleState`): a `WallAggregate` (= `FusedWall`), whose `.out` holds per body and particle
                                  (full-length arrays, no index lists, no host synchronisation, fixed shapes: capturable in a CUDA graph)
                                      lam   [B, N]         int_solid W
                                      G     [B, N, 2]      int_solid grad W
                                      Cov   [B, N, 2, 2]   int_solid y (x) grad W      (renormalisation matrix of the wall)
                                      cover [B, N, 2]      free-surface cover vector (degree-1 cone kernel)
                                      lap   [B, N]         wall Laplacian (when `laplacian`)
                                      tens  [B, N, 2]      tensile vector of the shifting
                                  and the methods `evaluate((WallOutput("A", 0, "a1g1"),), a1=...)` (the hydrostatic term for a per-body vector field a1),
                                  `cone_area(axes, half_angle)` ([2, N]: cone and full-disk area of the solid), `copy_from(other)` / `reset_derived()` (persistent buffers for a carry across steps).
    signed_distance(x, ...)       (d, n, hit[, body]) of points to the nearest body: d > 0 in the fluid, n into the fluid
    bodies                        the rigid bodies (pose, velocity, acceleration: `velocityAt`, `accelerationAt`), bound to the integrated state of the scheme while a step runs
    supported(...)                whether the fused (graph-capturable) path serves the scene

The boundary-condition CLOSURES (`BCType`: constant / zeros, freeSlip, noSlip) are the scheme's: they combine these integrals with the body kinematics.  Representations: `SurfaceRep`, `BoxRep`, and as exact polygons
`ImplicitRep` (disks) and `SdfRep` (see `Body.fusedReps`); the particle representation of warpSPH implements the same protocol with its own pair sums.
"""
from typing import Protocol, runtime_checkable

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, WarpOperation

from ..edge.warpfused import FusedGroup
from .boundaryOps import kernelName
from .fixedadj import fixed_adjacency
from .fused import FusedWall, WallOutput
from .scene import Scene
from .tensile import tensile_factor
from .viscosity import lap_factor

WallAggregate = FusedWall

_FAMILY = {KernelFunctions.Wendland2: "w2", KernelFunctions.Wendland4: "w4"}


@runtime_checkable
class BoundaryProvider(Protocol):
    """what a scheme needs from a boundary representation (see the module docstring)."""
    bodies: list

    def aggregate(self, ps, support, **kwargs): ...

    def signed_distance(self, x, body=None, supportMax=None, want_body=False): ...


class AnalyticBoundary:
    """the analytic bodies of a `Scene` as a boundary provider: exact edge-reduction integrals from one fused kernel family per position set."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self._layouts = {}                                    # (kernel, laplacian, fixedAdjacency) -> (groups, {role: group index})

    @property
    def bodies(self):
        return self.scene.bodies

    def family(self, kernel):
        """Wendland family of the exact wall operations (tensile, wall Laplacian): C2 and C4 only."""
        fam = _FAMILY.get(kernel)
        if fam is None:
            raise NotImplementedError("exact wall operations: Wendland C2 and C4 only")
        return fam

    def supported(self, fixedAdjacency=True):
        """True when the fused, graph-capturable path serves every body (surface / box loops; disks and sampled distances as their polygon when the fixed-capacity adjacency builds it)."""
        return FusedWall.supported(self.scene, fixedAdjacency)

    def properties(self, kernel, operation=WarpOperation.Density):
        return OperationProperties(kernel=kernel, operation=operation, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)

    def layout(self, kernel, laplacian, fixedAdjacency=True):
        """(groups, roles): the fused kernel groups the scheme needs and the group index of each role ('w' kernel, 'cone', 'lap' when `laplacian`, 'tens'); built once per configuration."""
        key = (kernel, bool(laplacian), bool(fixedAdjacency))
        if key not in self._layouts:
            fam = self.family(kernel)
            groups, idx = [FusedGroup(kernelName(kernel)), FusedGroup("cone", (3, 4))], {"w": 0, "cone": 1}
            if laplacian:
                lap_factor(1.0, fam)                                                      # registers the kernel
                idx["lap"] = len(groups)
                groups.append(FusedGroup("l" + fam))
            tensile_factor(1.0, fam)                                                      # the shifting may be asked for whatever the configuration says: the tensile group (4 terms) is always there
            idx["tens"] = len(groups)
            groups.append(FusedGroup(fam + "p5", (3, 4)))
            self._layouts[key] = (tuple(groups), idx)
        return self._layouts[key]

    def aggregate(self, ps, support, kernel=KernelFunctions.Wendland2, laplacian=False, fixedAdjacency=True):
        """the fused wall evaluation at the positions of `ps`: adjacency (one launch per body and representation when `fixedAdjacency`, else `Scene.adjacency`, the oracle), one stage-1 launch family for the kernels
        of the layout, and every static output of this position set in `.out` (lam, G, Cov, cover, lap, tens; raw: the scheme applies its factors).  `support`: host float >= every support of `ps`."""
        groups, idx = self.layout(kernel, laplacian, fixedAdjacency)
        props = self.properties(kernel)
        adj = fixed_adjacency(self.scene, ps, props, support) if fixedAdjacency else self.scene.adjacency(ps, props)
        fw = FusedWall(self.scene, adj, groups)
        outs = [WallOutput("lam", 0, "lam"), WallOutput("G", 0, "g0"), WallOutput("Cov", 0, "cov"), WallOutput("cover", idx["cone"], "g0")]
        if "lap" in idx:
            outs.append(WallOutput("lap", idx["lap"], "lap"))
        outs.append(WallOutput("tens", idx["tens"], "g0"))
        fw.out = fw.evaluate(outs)
        return fw

    def signed_distance(self, x, body=None, supportMax=None, want_body=False):
        return self.scene.signed_distance(x, body=body, supportMax=supportMax, want_body=want_body)
