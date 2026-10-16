# Distance-only semi-analytic SDF reference

`SDF-distance-only` is a standalone reference scheme for the classical local
half-plane approximation. The [report](../../warpSPHDevelopment/curvatureBoundaries/results/figures/sdf-distance-only/report.md)
compares it with the forced curvature series and the executed guarded route,
with separate moment maps and error budgets.

For a radial kernel, mass is `m00 = F0(d/h)`: signed distance alone determines
the fraction of kernel mass behind a plane. Vector and tensor moments also
use a unit normal to orient that plane. There is no curvature recovery,
Laplacian, curvature expansion, validity selection, or fallback in this
reference. It samples values and first differences independently of the
production SDF jet/guard methods.

All twelve moments use the existing exact polygon closed forms on a rectangle
equivalent to a half-plane within support. Its other three edges are outside
the kernel. Tangential symmetry and normal-frame rotation give the world
components. No quadrature is used for these half-plane integrals. The CPU
implementation is intended for reference and diagnosis, not throughput; it
does not change production dispatch or claim to reproduce every implementation
detail of the semi-analytic boundary-handling paper.

## Usage and coverage

```python
from warpSPHBoundaries.validation.distance_only import sampled_distance_only

# rep is a CPU SdfRep; all coordinates and distances are in h=1 support units.
result = sampled_distance_only(rep, points, kernel='w4')
moments = result['values']  # [N,12], canonical channel order
normal_defined = result['normal_defined']
```

`halfplane_channels(distance, normal, kernel)` also evaluates the model directly
without sampling an SDF. With no normal argument, it uses the positive y normal.
The order is:

```
m00, m10, m01, m20, m11, m02, Gx, Gy, g1xx, g1xy, g1yx, g1yy
```

The initial reference study covers the preserved 21 polygon geometries, four
base kernels and 4,480 query/kernel records, plus spatial maps and cuts for
the corner, tight disk, thin strip and narrow gap. All polygon mass values are
finite; the half-plane approximation fails the general float32-accuracy gate
on every polygon fixture because of model bias. This is an informative
baseline, not a replacement for the accurate original-geometry integrator.
Signal-relative errors can be enormous for tiny triangles: the true mass
scales with their area while the half-plane predicts about half a kernel.
That is geometry/model loss, even though the predicted mass does not diverge.
The production SDF representation's partial moment contract is unchanged;
the reference scheme's twelve moments do not complete production M1.

## Different errors from the curvature model

The half-plane model is exact for a single plane and remains free of the
curvature-recovery pole at a convex corner. Its mass is continuous wherever
the sampled distance is continuous. The exact model is bounded in [0,1];
small floating deviations at the endpoints are retained as numerical error,
not clamped away. This does not make the geometry approximation accurate:

- At a right-angle vertex the planar wall limit is mass 1/2, while a quadrant
  contains 1/4 of a normalized radial kernel. The mass error is 1/4.
- Constant distance gives constant predicted mass, but actual corner mass
  varies with angle; one distance value does not encode the support overlap.
- Thin solids, narrow gaps and multiple sheets can truncate/add significant
  solid mass compared with the one-plane model.
- At medial axes a unique normal may not exist. A vanishing sampled gradient
  leaves directional channels NaN, while scalar mass remains available.
  A nonzero reconstructed bisector is a chosen plane orientation, not evidence
  of a physically unique closest face. Directional transitions can therefore
  remain even though the curvature singularity is absent.

Grid queries beyond central-stencil coverage are unavailable. Fully empty or
solid model supports have isotropic moments and need no normal. Undefined
exact normals are also masked in the reference decomposition, rather than
assigning a fabricated geometric direction.

## Error budget and verification

For the original geometry G, separate signed contributions:

```
numerical = float64 plane(sampled d,n) - extended-precision plane(sampled d,n)
SDF reconstruction = plane(sampled d,n) - plane(exact d,n)
half-plane geometry = plane(exact d,n) - I(G)
total = numerical + SDF reconstruction + half-plane geometry
```

The last two lines use extended-precision plane integrals. Exact geometry jets
come from disk formulas or segment projection, independently of the sampled
grid. Contributions add where their normal references are defined. Scalar
mass requires no normal and has a complete split. Maxima of separate
contributions need not add. The precision comparison is empirical: it shares
identities with the implementation and is not a certified bound. Disk target
quadrature has a separate 48/96-node refinement estimate on cuts; this is also
not a bound. No few-eps claim is made from these comparisons alone.

All twelve channels were checked against rotated half-plane polygon integrals
for cubic/w2/w4/w6, with regressions for curvature-path independence, the
face/vertex seam, undefined normals and sampled-grid coverage. The corner
probe mass also agrees with the 50-digit half-plane reference. Maple verifies
the moment frame identities and half-plane/quadrant wall limits in
`maple/19_distance_only_frame.mpl`.

Reproduce:

```bash
python scripts/studies/sdf_distance_only.py
python -m warpSPHBoundaries.validation.audit_report --refresh-stress-only --output "$EVIDENCE"
```

The original numerical survey and production guard checkpoint are preserved.
The broad audit will include the named reference automatically on future runs.
