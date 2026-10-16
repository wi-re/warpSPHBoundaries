# Sampled SDF local-model guard

`SdfRep` uses a conservative domain guard before evaluating its smooth-boundary
curvature series. Rejected supports use the supplied `SurfaceRep`; if none was
supplied, Scene raises an explicit error. The forced fields in the validation
plots bypass this selection deliberately and can still diverge.

The [CPU checkpoint](../../warpSPHDevelopment/curvatureBoundaries/results/figures/sdf-guard-cpu/report.md) and
[CUDA checkpoint](../../warpSPHDevelopment/curvatureBoundaries/results/figures/sdf-guard-cuda/report.md) compare fresh
executed paths against the preserved polygon references. The
[corner mechanism study](../../warpSPHDevelopment/curvatureBoundaries/results/figures/sdf-corner-mechanism/report.md)
explains the distance-only ablation and the curvature pole.

For a standalone baseline with no curvature recovery or guard, use the
[distance-only semi-analytic reference](sdf-distance-only.md). It has separate
plots for half-plane geometry bias, SDF reconstruction and numerical evaluation,
and leaves the production policy unchanged.

The [accepted smooth-SDF audit](sdf-accepted-accuracy.md) addresses the separate
accuracy question on circles, ellipses and inflected walls. It executes Scene
where this guard accepts the local model and compares against actual curved
domains. Planar moments and omitted tangential curvature variation remain
accuracy limitations even when the differential field passes every guard.

## Checks and fallback

The defaults are:

| Check | Requirement |
|---|---|
| Distance, gradient and Laplacian samples | Finite |
| Surface-curvature denominator `D = 1 - d Delta d` | `D >= minCurvatureDenominator`, default 0.25 |
| Gradient norm | Entire interval inside `[1-gradTol, 1+gradTol]`, default `[0.9,1.1]` |
| Recovered curvature | Entire bound satisfies `abs(kappa)*h <= maxKappaH`, default 0.5 |
| Support coverage | All cells in the support's bounding square have valid central-difference stencils inside the sampled grid |
| Evaluated result | Finite mass/gradient and mass in `[0,1]` with 64 float64 eps of roundoff slack |

A positive denominator matters: taking its absolute value would also accept
the wrong side of a curvature-recovery pole. For an exact exterior-corner
distance, `Delta d=1/d` and `D=0`, despite a unit gradient. Curvature at the
projected vertex is undefined. The new denominator margin rejects the
ill-conditioned neighborhood as well as the exact pole. Curvature is masked
before the table series; the output is not clamped into the physical range.

The old sixteen probes could miss a sharp feature or medial axis between
probe locations. The new check covers every cell touched by the support's
axis-aligned bounding square, including cells between those probes. The square
contains the support disk and deliberately rejects some otherwise usable
supports. It does not follow moving bodies in world coordinates: all tests
and cached fields live in the body's local frame, like the SDF itself.

Supports known to be entirely fluid/solid bypass the differential model. The
threshold is `abs(d) >= h + sqrt(2)*spacing`, with the appropriate sign and
query-stencil coverage. The margin is conservative for bilinear interpolation
of samples of a 1-Lipschitz distance. This assumes the supplied distance samples
represent the intended SDF; it is not an independent validation of that input.

## Bounds over an entire cell

The evaluator blends nodal distance, gradient components and Laplacian
bilinearly. Each blended scalar lies between its four nodal extrema.

1. Bound `d` and `Delta d` by intervals. Their product lies between extrema
   of the four endpoint products. This gives a lower bound for `D`.
2. Bound each gradient component by its interval. Distances of those intervals
   from zero bound the smallest norm; the largest absolute endpoints bound
   the largest norm. This catches opposing gradients at a medial axis even
   when the individual nodal gradient norms look reasonable.
3. With `D` bounded positively away from zero,
   `abs(kappa) <= max(abs(Delta d endpoints))/D_lower`.
4. Store each cell's maximum admissible support size. Invalid cells store zero;
   valid flat cells have no curvature-induced support-size limit. Outer cells
   containing replicate-padded derivative nodes are invalid.

Floating-point slack is reserved in the interval operations. These checks
bound the blended jets actually used by this evaluator; they are not a claim
of formally certified integral accuracy.

The minimum support limit over a region is queried through a cached table of
power-of-two square minima at every cell origin. Four overlapping squares
cover each isotropic support's cell rectangle. Region checks therefore avoid
scanning every covered cell for each particle. The implementation uses tensor
operations on CPU/CUDA without transferring query coordinates to the host.
Static storage is O(grid cells times log grid width); per-query tensor work is
O(log grid width). Performance has not been tuned or benchmarked for production
throughput. Changing the guard thresholds rebuilds the table; `to(device)`
clears it. As with the existing nodal-derivative cache, sampled values are
assumed static after construction.

## Accuracy limits

Passing this guard establishes a conservative domain for the interpolated
jets. It does **not** establish that the integral is within float32 epsilon:

- The mass series omits higher curvature terms.
- All twelve Scene moments use a shared order-two expansion; higher geometry
  orders and curvature derivatives remain omitted.
- The distance grid and its differential reconstruction have numerical and
  representation errors.
- Features absent from the samples cannot be recovered by any sample-only
  guard. A sampled contour can lose them too.

For stringent accuracy requirements, use the original polygon as fallback and
retain the exact-geometry integrator as the authoritative path. An independent
model-error estimator or a validated tighter approximation domain is still
needed before the local SDF model can promise a requested integral tolerance.
An automatic change to the distance-only model would remove the pole but
would retain the potentially large corner geometry bias.

## Curvature-variation extension

The [shared order-two moment family](derivations/curvature-moments-2d.md) also
uses κ_s. `maxCurvatureVariationH2=0.5` limits |κ_s|h² over every cell of
the support square, with a conservative Laplacian-gradient interval bound.
Third differences require a two-node stencil; its unavailable outer cells are
rejected. Nonfinite variation cannot enter the model. This is an asymptotic
domain guard, not a requested-error guarantee. Small, rapidly oscillating walls
can satisfy a curvature-only threshold while violating this variation limit.
