# Solid triangle/tetrahedron WCSPH trial handoff

The dedicated C2/C4 kernels are an independent solid-boundary route. No porous
material, permeability, porosity or porous pressure operator is needed to test
ordinary impermeable walls. Nothing in warpSPH or shared Core/Integrator was
changed for this checkpoint. A fluid solver has not yet run with these kernels.

## Implemented contract

`fluxbridge.solid_functions(d, kernel, precision, bundle)` returns reusable Warp
`element` and batch `kernel` entry points. The output layout is `[channel,pair]`;
2D positions are vec3 with z=0. `h` is the **compact support radius**, not a
kernel-dependent smoothing length. Geometry consists of filled 2D triangles or
filled 3D tetrahedra, not 3D surface triangles. Arbitrary winding is accepted.

| bundle | 2D / 3D scalars | outputs |
|---|---:|---|
| `density` | 1 / 1 | lambda = integral W |
| `pressure` | 3 / 4 | lambda, G = integral grad_x W |
| `lean` | 6 / 10 | pressure + symmetric Cov = integral (y-x) tensor grad_x W |
| `wcsph` | 7 / 11 | lean + lap = integral Laplacian W |
| `wcsph_extended` | 11 / 17 | wcsph + m1 = integral (y-x) W + cover = integral unit(x-y) |

All integrals are over the solid within the support. Cov ordering is xx,xy,yy
in 2D and xx,xy,xz,yy,yz,zz in 3D. Units are lambda/Cov: 1; G: length^-1;
lap: length^-2; m1: length; cover: length^d. Cover's wall-particle number density
is a caller multiplier. The cover radius can differ from the SPH support; in
that case evaluate cover with a separate query/support snapshot.

**Viscosity conversion matters:** new `lap` is already the physical integral.
The existing Scene/warpSPH provider exposes a *normalized auxiliary-kernel*
channel, then multiplies by `scene.viscosity.lap_factor(h, family)` in its wall
closure. An adapter must either consume the physical result directly or divide
by that factor before exposing the legacy channel. Applying the factor again
is wrong. This supports the existing integrated-kernel-Laplacian closure;
Morris-type viscosity needs its own operator. Keep slip/no-slip factors,
viscosity coefficients, wall velocities and pressure closure in the solver.

The m1 primitive is ell(r)=integral_1^r t phi(t) dt; wall viscosity uses
phi'(r)/r, and cover uses r-1. The auxiliary reductions share the existing J/C
edge stream and geometry. No quadrature, device fp64 fallback, FEM payload or
large general-moment output array was added. Default fp32 uses protected
all-fp32 two-float arithmetic; native raw fp32 remains an explicit diagnostic
that fails the broad accuracy gate. Default fp64 remains analytic fp64.

## Executable independent evaluation

```python
import numpy as np
from warpSPHBoundaries.fluxbridge import SolidEvaluator

# A filled rectangle below the fluid, decomposed into disjoint triangles.
vertices = np.array([[[-2., -1.], [2., -1.], [2., 0.]],
                     [[-2., -1.], [2., 0.], [-2., 0.]]])
evaluator = SolidEvaluator(vertices, device='cuda:0', kernel='w4',
                           precision='float32', bundle='wcsph', search='device')
batch = evaluator.prepare([[0., .1], [.25, .2]], [.5, .5])
batch.evaluate()
print(batch.labels)
print(batch.aggregate.numpy())  # [7,2], device result until explicit readback
# Re-evaluate this frozen batch, or prepare a new snapshot after particles move.
```

The same interface accepts `[E,4,3]` tets. `prepare(..., periods=(Lx,Ly,...))`
includes periodic images. Caller-provided adjacency is checked against geometry
and the actual dtype-rounded query/support snapshot. Host and Warp BVH search
are available; whole elements are retained conservatively, including all their
regular-flux faces. CSR assembly has a deterministic element/image order and
sums pair values on the selected device without float atomics.

Mesh interiors must be disjoint: aggregation sums volumes and does not compute
unions or detect overlaps. Internal faces cancel geometrically when summing
adjacent tets/triangles. Skin triangles need a closed-volume representation or
volume mesh first. Prepare geometry/BVH once; rebuild them after mesh motion or
deformation. This bridge has no background/complement representation or body
IDs/reaction scatter. Existing Scene disk/cylinder/fibre providers remain
available separately. Transverse 2D fibre flow and longitudinal flow are still
distinct applications.

The current convenience bridge accepts host query/support arrays, freezes a
pair snapshot, allocates pair inputs/outputs, then reduces them. Even device
BVH search returns CSR to the host before this bridge uploads arrays; it uses
the existing fp64 geometric search filter. It is **not** a GPU-resident moving
particle pipeline. Rounded fp32 positions at large translations can represent
different geometry; protected arithmetic cannot restore information lost at
input conversion. Search/assembly timings must accompany kernel throughput.

## What remains before fluid behavior can be assessed

1. Add an opt-in solid-simplex boundary-provider adapter in the consuming solver
   worktree. Map lambda/G/Cov and the physical lap convention explicitly; rebuild
   particle adjacency when positions or support change. Retain the existing
   ordinary disk/cylinder/fibre path. This is ordinary solid integration, not
   the deferred porous Stage 5 handoff.
2. Start with a static, impermeable, single-body wall, C2 or C4, no tensile
   shifting and no porous coupling. Run hydrostatic rest (wall force, density
   drift, penetration), then no-slip Couette/Poiseuille flow (viscous sign and
   profile), comparing against the current provider and mesh refinement in
   both precisions. Repeat with a periodic channel to check image bookkeeping.
3. Wire body kinematics, per-body contractions/reaction/torque, complement
   regions and free-surface geometric queries before enabling their features.
   The current warpSPH provider contract is 2D; a 3D trial also requires the
   consuming solver's 3D boundary data/kinematics contract.
4. Optional tensile shifting uses integral W^4 grad W = (1/5) grad integral W^5.
   Its C2/C4 degree-25/40 profiles are **not** supplied by the new bundles. The
   established 2D Scene route uses conditioned Chebyshev quadrature; do not
   replace it with an unstable high-degree monomial recurrence. Either retain
   that route for an initial 2D trial or leave tensile shifting disabled until
   a dedicated conditioned implementation passes its own precision gate.

There is no missing porous prerequisite for the first static solid-fluid trial.
The remaining prerequisites are the adapter and actual fluid validation, not
more general porous moments. The principal later porous-flow reference remains
Böttcher et al. (2025), *Implicit Incompressible Porous Flow using SPH*,
arXiv:2504.07739v2.
