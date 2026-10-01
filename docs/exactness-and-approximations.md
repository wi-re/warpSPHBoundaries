# Exact core, cheap approximations (stage 2 results)

**Status:** [V] — `python/edgebound/np2d.py`, `tests/edge/test_np2d.py`, `python -m edgebound.np2d_study`
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
3. **Degenerate placements need no exact predicate.** The orientation signs, the indicator and the atan jumps are all
   derived from the *same computed* `z_e`; whatever rounding does, the result is the exact value for a slightly
   perturbed configuration, so it is continuous (checked across an edge from 1e-15 to 1e-3, at vertices, for
   `z → 1e-40`). Only quantities that are individually ill-conditioned near a vertex needed care:
   `z = cross(p,q)/ℓ` (not `cross(p,q−p)/ℓ`) and the angle as `atan2(z(hi−lo), z²+hi·lo)`.
4. **Because the exact object is a 1D integral of a smooth function, cheap approximations stay accurate.**
   Errors of any approximation are *1D-quadrature* or *low-precision-arithmetic* errors on edge integrals, not 2D
   integration errors over a region clipped by a circle. Measured below.

## Measured accuracy (reference: 40-digit golden fixtures; "hard" = all 82 convex-triangle fixtures)

Max absolute error over the set (value: 1; gradient × h; moment_α / h^k). `python -m edgebound.np2d_study`.

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

## What is still open (stage 3 and later)

* float32: a better-conditioned polynomial basis for the edge terms (Bernstein in `(1−q)` or Horner in `r` around the
  support radius) and a measured float32 tolerance table from the fixtures.
* Non-convex polygons in numpy (indicator from sign tests is for convex polygons; decompose or use winding).
* Cheaper transcendental functions: replace `asinh`/`atan2` by bounded-error approximations and measure (error budget is
  now known: the angle term is multiplied by an exact weight `M_R ≤ 4/7/π`).
* Autodiff/torch/warp: the gradient identity is the analytic derivative; the consistency of sign/indicator/atan is what
  keeps autodiff of the *value* from producing spurious deltas.
