# Exact core, cheap approximations (stage 2 results)

**Status:** [V] — `src/warpSPHBoundaries/edge/np2d.py`, `tests/edge/test_np2d.py`, `python scripts/studies/np2d_study.py`
**Audience:** the argument for the method: *what is exact, what is approximated, what the approximations cost.*

## The argument in four points

1. **The reduction is exact, not a model.** The value, gradient and all moments of a polygon (or mesh)
   against a piecewise-polynomial kernel are *identities* (Maple-proven, `derivations/`): polygon ∫ = indicator +
   1D edge integrals. There is no discretisation parameter, no mesh-dependent error, no fitted constant.
   Covering meshes reproduce the full-space kernel moments to rounding.
2. **All kernel algebra is exact and happens once.** The kernel → truncated monomials, the moment recursion,
   the indicator weights (Wendland `1`; cubic `8/7`, `−1/7`) and atan weights are compiled in rational arithmetic
   (`np2d.compile_moment`) and rounded to float exactly once. Rounding can therefore enter *only* through the
   geometry (positions, chord clipping, 1D primitives) — never through the kernel polynomial algebra, the weights, or
   the recursion bookkeeping.
3. **Degenerate placements require shared computed predicates.** In the numpy reduction, the orientation signs, the indicator and the atan jumps are all
   derived from the *same computed* `z_e`; whatever rounding does, the result is the exact value for a slightly
   perturbed configuration, so it is continuous (checked across an edge from 1e-15 to 1e-3, at vertices, for
   `z → 1e-40`). Only quantities that are individually ill-conditioned near a vertex needed care:
   `z = cross(p,q)/ℓ` (not `cross(p,q−p)/ℓ`) and the angle as `atan2(z(hi−lo), z²+hi·lo)`.
   This is a requirement on each backend, not a blanket guarantee for scene integration.
   The original adversarial audit exposed half-unit/full-unit sloped-edge jumps
   when CUDA FMA and the indicator made different rounded orientation decisions.
   Shared rounded cross products and per-query support normalization now pass
   CPU/CUDA boundary regressions; the refreshed fused-mono path improves from
   2/21 to 16/21 fixtures within ε32. Tiny/sliver relative accuracy remains open.
   Current gates and evidence: [2D accuracy plan](../../warpSPHDevelopment/curvatureBoundaries/progress/plan-2d-accuracy.md).
4. **Because the exact object is a 1D integral of a smooth function, cheap approximations stay accurate.**
   Errors of any approximation are *1D-quadrature* or *low-precision-arithmetic* errors on edge integrals, not 2D
   integration errors over a region clipped by a circle. Measured below.

## Measured accuracy (reference: 40-digit golden fixtures; "hard" = all 82 convex-triangle fixtures)

Max absolute error over the set (value: 1; gradient × h; moment_α / h^k). `python scripts/studies/np2d_study.py`.

| variant | generic: value | grad | m₁₁ | m₂₂ | hard: value | grad | m₁₁ | m₂₂ |
|---|---|---|---|---|---|---|---|---|
| **float64, closed form (default)** | 2.3e-14 | 2.5e-13 | 1.1e-14 | 1.1e-14 | 6.0e-14 | 5.3e-13 | 1.1e-14 | 1.1e-14 |
| longdouble, closed form | 5.6e-17 | 2.2e-16 | 3.9e-18 | 3.2e-18 | 1.1e-16 | 3.0e-16 | 3.9e-18 | 7.5e-18 |
| float32, closed form † | 6.0e-05 | 3.5e-04 | 2.5e-05 | 9.8e-06 | 6.0e-05 | 3.5e-04 | 2.5e-05 | 9.8e-06 |
| float64, Gauss 3 nodes × 4 panels | 7.2e-05 | 8.1e-03 | 2.0e-04 | 3.1e-05 | 1.0e-04 | 1.7e-02 | 6.0e-04 | 1.3e-04 |
| float64, Gauss 4 × 4 | 1.7e-05 | 1.6e-03 | 2.5e-05 | 6.6e-06 | 2.7e-05 | 7.0e-03 | 5.6e-05 | 1.6e-05 |
| float64, Gauss 6 × 4 | 4.7e-07 | 1.8e-05 | 2.8e-07 | 1.0e-07 | 4.7e-07 | 1.8e-05 | 6.5e-07 | 5.0e-07 |
| float64, Gauss 8 × 4 | 1.1e-08 | 3.0e-06 | 1.0e-10 | 2.2e-10 | 1.1e-08 | 6.1e-06 | 1.0e-10 | 1.2e-09 |

† float32 cannot even represent the 1e-6·h elements next to a point 0.2 h away (4 cases skipped). The ~1e-4 level is
the cancellation among the large alternating kernel coefficients (|c_n| up to ~10³); it is a *basis conditioning*
issue, to be attacked in stage 3 (e.g. Bernstein/`(1−q)` form), not a property of the method.

* **Tiny elements** — the only genuine float64 failure of the plain split form — is fixed exactly by the unsplit
  form (identity, no indicator/atan): relative error of the value for `x` inside / at a vertex

  | form | L_T/h = 1e-6 | L_T/h = 1e-3 |
  |---|---|---|
  | split form only | 1.6e-04 | 2.3e-10 |
  | with unsplit form (default) | 1.8e-16 | 3.8e-16 |

* The remaining ~1e-13 in float64 comes from the kernel's coefficient magnitudes (longdouble gives 1e-16..1e-18 with the
  *same* algorithm), i.e. it is arithmetic precision, not algorithmic error. The numpy-f64 tolerance of the verification
  table (value 1e-12, gradient 1e-11) is met on every fixture with margin ≥ 20 (`tests/edge/test_np2d.py`).

## Cheap approximation 1: 1D Gauss quadrature of the edge integrals (no asinh/closed forms)

Replace the closed-form chord integrals by Gauss–Legendre on 4 panels split at `−|z|, 0, |z|` (the foot point and the
complex singularities `±iz` of odd powers of `r`); the angle term stays closed form. Max error of the **value** for
random triangles with `x` inside the support, against the number of kernel evaluations per triangle
(`np2d_study.vs_area_quadrature`; N = 300):

| kernel | method | evaluations | max abs error |
|---|---|---|---|
| cubic | 2D tensor Gauss on the triangle, 16² | 256 | 5.0e-05 |
| cubic | 2D tensor Gauss, 32² | 1024 | 2.1e-06 |
| cubic | **edge reduction**, Gauss 4×4 panels | **48** | 3.4e-05 |
| cubic | **edge reduction**, Gauss 6×4 | **72** | 2.4e-06 |
| w4 | 2D tensor Gauss, 16² | 256 | 2.1e-05 |
| w4 | 2D tensor Gauss, 32² | 1024 | 1.0e-07 |
| w4 | **edge reduction**, Gauss 4×4 panels | **48** | 6.4e-06 |
| w4 | **edge reduction**, Gauss 6×4 | **72** | 1.9e-09 |
| w4 | **edge reduction**, closed form | ~6 transcendental calls | 1.9e-14 |

The plain 2D rule has to resolve the kink at `r = h` (and the origin when `x ∈ T`); the edge form integrates smooth 1D
functions, so 5× (cubic, ~4e-5) to ~15× (cubic, ~2e-6) and ~50× (w4, ~1e-9) fewer evaluations reach the same accuracy — and the
closed form is exact to rounding for ~6 transcendental calls. In numpy the closed form is also the *fastest*
(below), so quadrature is not a speed-up on CPU; it matters where `asinh` is expensive or unavailable (fixed-point, some
GPU intrinsics) and as the building block for non-polynomial kernels (tier 3/4).
Weakness (honest): gradient and odd moments degrade at small `z/h` for fixed nodes (kink of `r^odd` at the foot point);
the 4-panel split removes most of it, the closed form removes all of it.

## Cost (numpy, w4, 200 000 triangle/point pairs, one thread, µs per pair)

| variant | value | gradient | m₁₁ | m₂₂ |
|---|---|---|---|---|
| closed form | 4.0 | 2.1 | 3.4 | 12.6 |
| Gauss 3×4 | 5.1 | 2.7 | 6.2 | 19.5 |
| Gauss 4×4 | 6.2 | 3.4 | 7.4 | 23.5 |

(unoptimised numpy, ~100 flops·array passes per edge; absolute numbers are indicative only.)

## Stage 3: float32 — diagnosis and an exact fix

**Diagnosis** (measured with the stage-2 code on the fixtures): rounding the *geometry* (positions, `z`, `s_0`, `s_1`, chords,
indicator) to float32 costs only ≈ 3e-8; evaluating the closed form in float32 costs 6e-5…3.5e-4. The loss is entirely the
**monomial basis of the edge profiles**: the kernel's `P(r) = Σ c_n r^n` (and the iterated potentials Φ of the moment recursion)
are smooth, bounded functions written as sums of huge alternating terms. Amplification `Σ|c_n| R^n / max|P|`:
cubic 1.4e2, w2 3.1e2, w4 2.3e3, w6 1.8e4 (worst over all profiles up to |α| = 4). float32 `ε = 6e-8` times that amplification
is exactly the observed error.

**Fix, exact and offline:** convert every profile to the **Chebyshev basis on [0, R]** with rational arithmetic
(`np2d.cheb_coeffs`; checked exactly at rational points in `tests/edge/test_np2d_stable.py`). The amplification
`Σ|a_k| / max|P|` becomes **1.2 for every kernel and every α ≤ 4** (vs 1e2…2e4), and Clenshaw evaluation is backward stable.
The edge integrals are then done by Gauss quadrature of the (smooth, now well-conditioned) integrand on dyadic panels
`[0,|z|], [|z|,2|z|], [2|z|,4|z|], …` around the foot point (the singularities `±i z` of odd powers of `r`), the angle term stays
closed form, the indicator/atan weights stay exact (`stable=(nodes, panels)` in `np2d`).

float32 tolerance table (max abs error over the fixtures of each class, all four kernels; gradient scaled by `h`, moments by `h^k`;
`python scripts/studies/np2d_study.py`):

| class (cases) | method | value | grad | m₁₁ | m₂₂ |
|---|---|---|---|---|---|
| generic (16) | closed form | 6.0e-05 | 3.5e-04 | 2.5e-05 | 9.8e-06 |
| generic (16) | **stable 8×6** | **8.4e-08** | **3.3e-07** | **2.0e-09** | **6.3e-10** |
| x on edge / vertex / edge line (19) | closed form | 1.8e-05 | 2.0e-04 | 5.1e-06 | 3.1e-06 |
| x on edge / vertex / edge line (19) | **stable 8×6** | **6.6e-08** | **1.8e-07** | **1.6e-09** | **1.4e-09** |
| z → 0, down to 1e-40 (10) | closed form | 1.6e-06 | 2.3e-05 | 3.5e-07 | 2.4e-07 |
| z → 0 (10) | **stable 8×6** | **4.2e-08** | **1.8e-07** | **3.0e-09** | **3.7e-10** |
| tiny chords (5; 1 not representable) | closed form | 1.3e-05 | 2.3e-04 | 7.7e-07 | 2.0e-06 |
| tiny chords | **stable 8×6** | **1.4e-08** | **1.5e-08** | **6.0e-10** | **1.7e-10** |
| elements ≤ 1e-2 h (9) | closed form / stable 8×6 | 3.3e-09 / 3.5e-09 | 4.5e-08 / 3.8e-08 | 3e-10 / 3e-10 | 2e-11 / 1e-11 |
| half-plane, d = 0 … 1, d < 0 (18) | closed form | 7.0e-06 | 2.0e-04 | 0 | 4.2e-07 |
| half-plane | **stable 8×6** | **8.1e-08** | **2.2e-07** | 0 | **4.2e-10** |
| engulf / outside support (2) | both | 0 | 0 | 0 | 1.8e-10 |

Overall: float32 value 8e-8 (≈ 1.4 ε), gradient 3e-7, moments ≤ 3e-9 — a 700–1000× improvement over the float32 closed form,
with *no* change of the identities. Tests assert 5e-7 / 2e-6 / 2e-7 (margin 3–7×).

Caveats, measured:
* **Panels matter at small `z/h`.** The cheap 5×4 variant is fine for value and moments but its gradient degrades to 2e-4…6e-4 for
  `x` on an edge and for half-planes with small `d` (the last panel is too long next to the foot point); 8×6 holds 3.3e-7 everywhere,
  down to `z = 1e-40`. A production kernel should size the panel count from `log2(L/|z|)` (or switch to the closed form for large `z`,
  where both are fine).
* **Cost.** In numpy the stable mode is ~10× the closed form (value 49 µs vs 3.7 µs per triangle; 8×6, w4; gradient 17 vs 2.0 µs):
  more flops per edge, but all of it is branch-free fused multiply-add on positive terms, which is what a float32 GPU kernel wants;
  the closed form remains the choice whenever float64 is available.
* **Not representable in float32:** geometries whose features are below `ε·|v − x|` (e.g. an edge of 2e-12 at distance 0.9, or a 1e-6 h element
  0.2 h away) — `d = q − p` underflows to 0. This is an input-representation limit; for such elements pass element-local coordinates
  (relative to the element, plus the offset of `x` in float32 *relative to the element*).
* Mixed alternative (not needed now): float32 geometry + float64 profile evaluation gives 3e-8, i.e. the whole float32 loss can also
  be bought back with float64 only inside the profile polynomial.

## What is still open (stage 3 and later)

The reproducible cross-method audit is now
`python -m warpSPHBoundaries.validation.robustness --device cuda:0`.
Its registry covers numerical arithmetic, edge quadrature, FEM hybrid rules,
local SDF models, sampled contours, particle lattices, specialized tables,
curvature/width expansions and implicit tier switches. It sweeps tiny triangles,
acute and nearly flat corners, slivers, notches, boxes and ellipses. Outputs retain
the common twelve-channel ordering, signed numerical/model/total errors where
separable, independent reference checks, refinement/truncation estimates and
machine-readable coverage limits. Unknown same-model errors remain unavailable;
they are not relabelled as quadrature errors.

`$EVIDENCE/robustness/findings.json` lists exact numerical failure probes.
The original sloped-edge and circle-boundary expected failures are now ordinary
accuracy gates in `tests/validation/test_robustness.py` and
`tests/scene/test_boundary_predicates.py`. The circle evaluator now uses the
regular full-circle potential y M(r)/(pi r²), avoiding numerical cancellation
between a singular flux and the indicator, including at `D=R` and adjacent
floating values. `M(r)/r²` is evaluated directly to preserve its finite origin
limit. Half-angle geometry is checked in `maple/16_disk_boundary.mpl`.
Kernel profiles, compact primitives and regular M/r² are compiled exactly into
the existing Chebyshev basis, avoiding expanded high-degree cancellation.
The new circle tests also check 384/768-node polar-reference agreement: the
96-node reference was insufficient immediately inside some large disks.
The separate [C1/C2 checkpoint](../../warpSPHDevelopment/curvatureBoundaries/results/figures/accuracy-progress/report.md)
preserves the original audit while re-evaluating the changed paths. Both arc
rules now meet ε32 on all 21 tested radii across nine kernels and the nine
implemented channels. This is a finite measured scope, not few-eps certification
or a result for the missing moments. Polygon tiny/sliver failures remain.
Small features lost by SDF
sampling or particle lattices are geometric/quadrature failures, while accepted
but inaccurate ellipse predictions expose local-model/validity-policy failures.
Absolute accuracy alone is insufficient for tiny elements: the audit also
reports error divided by each case's maximum channel signal. The thinnest
sliver references themselves need higher precision for a few-eps relative
claim; the independent oracle spot checks quantify that limitation.

### Generated survey and geometry error

`python -m warpSPHBoundaries.validation.audit_report` builds
[`results/figures/robustness/report.md`](../../warpSPHDevelopment/curvatureBoundaries/results/figures/robustness/report.md)
from cached measurements. Absolute and signal-relative overviews are separate,
and every scheme has separate moment heatmaps and available contribution/estimate
graphs. Labels describe a priori expected regimes independently of measured
success. The report discovers new cases/studies and paginates wide sweeps.

An SDF can represent a triangle exactly. The approximation at issue is replacing
its nonlocal support intersection by one local smooth/flat boundary description;
this fails when a corner, medial axis or another boundary sheet matters.
Sampling the SDF introduces an additional, distinct representation error.

The [corner mechanism report](../../warpSPHDevelopment/curvatureBoundaries/results/figures/sdf-corner-mechanism/report.md)
separates the distance-only half-plane ablation from the forced curvature
series. Outside a convex vertex, the exact distance has unit gradient and
Delta d=1/d, but recovering closest-surface curvature divides by
1-d Delta d=0. A valid distance field therefore does not imply a finite
smooth-boundary curvature. The previous [0,1] field colour scale hid raw
curvature-series predictions far above 1. The half-plane ablation is continuous;
its geometric bias depends on both distance and angle around the corner.
The focused report isolates that bias from distance-sampling/table errors,
with constant-distance, face/vertex-seam and grid-refinement checks.

The new [sampled-cell guard](sdf-guard.md) rejects the curvature pole and checks
every blended-jet cell in the support bounding square. It replaces the finite
probe policy and catches the known strip/gap misses. Fresh
[CPU](../../warpSPHDevelopment/curvatureBoundaries/results/figures/sdf-guard-cpu/report.md) and
[CUDA](../../warpSPHDevelopment/curvatureBoundaries/results/figures/sdf-guard-cuda/report.md) checkpoints validate its
original-polygon fallback. Passing this domain guard does not certify a
requested integral tolerance on accepted smooth models.

For original geometry G, its sampled contour Gh, and the forced SDF result A:

```text
E_geometry = I(Gh) - I(G)
E_residual = A - I(Gh)
E_total    = E_geometry + E_residual
```

The cached oriented marching-squares contour supplies I(Gh). This quantifies
**geometry-induced integral error per moment/query/kernel**, rather than a
Hausdorff or area-only shape metric. Lost contours have I(Gh)=0 and therefore
explicitly lose the original signal. The residual contains the local
curvature/planar approximation, reconstructed distance/normal/curvature errors,
and table/arithmetic errors. The interpolated SDF jets are not necessarily
the exact differential geometry of its extracted contour, so this split does
not isolate all sampling effects. A further separation requires independent
same-model integrals and differential-geometry references.

Contour integrals have finite reference uncertainty; these are measured
contributions, not certified bounds. Signed contributions add pointwise, while
their separately maximized magnitudes need not add. For curved geometries
approximated by polygons, the cached same-polygon integral similarly separates
geometry-surrogate error from numerical evaluation. Guarded SDF/polygon mixed
paths retain TOTAL where an executed-branch reference is unavailable.
The generated report states these limits alongside the plots.

* float32 panel-count rule from `z/h` and a fused kernel for warp (stage 5); the Chebyshev coefficients are already exact constants.
* Non-convex polygon stress coverage is present for the winding/crossing implementation; general holes and overlapping-body unions still need explicit coverage.
* Cheaper transcendental functions: replace `asinh`/`atan2` by bounded-error approximations and measure (error budget is
  now known: the angle term is multiplied by an exact weight `M_R ≤ 4/7/π`).
* Autodiff/torch/warp: the gradient identity is the analytic derivative; the consistency of sign/indicator/atan is what
  keeps autodiff of the *value* from producing spurious deltas.
