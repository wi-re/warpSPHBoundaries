# Backends and verification hierarchy

Every result is implemented in stages, and each stage is tested against the one
before it on the **same cases**.

| stage | tool | role | precision |
|---|---|---|---|
| 0 | Maple | symbolic derivation, exact identities, series | exact |
| 1 | mpmath (scalar) | **reference implementation** (same algorithm as production: same branches, clipping, primitives) + independent quadrature oracle | 30–50 digits |
| 2 | numpy float64, vectorised, **branch-free** (masks / `where`) | notebooks, parameter sweeps | 1e-12 … 1e-15 |
| 3 | numpy float32 | isolates conditioning problems from autodiff | see table below |
| 4 | torch | autodiff check of the analytic gradients, shape derivatives | f64 then f32 |
| 5 | warp | GPU kernels; custom adjoints | f32 (f64 where needed) |

The mpmath stage needs **two** oracles: the closed-form reference *and* an
independent quadrature (polar where there is a radial breakpoint, Cartesian
otherwise). A quadrature oracle alone proves the formula, not the clipping and
degenerate-case logic.

## Golden fixtures

mpmath exports a golden set (JSON or npz) of cases: geometry, `x`, kernel, and
`λ`, `∇λ`, moments to ~30 digits. It must deliberately include:

- `x` on an edge and at a vertex; `z → 0`; chords of length ~1e-12;
- elements `<< h` (`L_T/h = 1e-3 … 1`); elements covering the whole support;
- `d = 0`, `d → h`, `d < 0` (tier 3), engulfing support (`2R + d < h`);
- every kernel (cubic, w2, w4, w6) and every field order `p ∈ {0,1,2,3}`.

Every later stage loads this same set; tolerances are tied to dtype:

| stage | value | gradient | notes |
|---|---|---|---|
| numpy f64 | 1e-12 | 1e-11 | after chord clipping, not for `L_T << h, p >= 2` |
| numpy f32 | to be measured | to be measured | recorded per case, becomes the f32 tolerance table |
| torch / warp | as numpy at the same dtype | as numpy | plus autodiff-vs-analytic below |

## Interfaces

Function signatures are identical across stages (same inputs, same outputs) so
tests and notebooks swap the backend. Notebooks run on numpy; mpmath is used
only for the validation points.

## Autodiff (torch)

Analytic gradients (`edge-gradient-identity.md`) are the primary implementation;
autodiff is an independent check. Known traps:

- clipping (`min`/`max`), the indicator and the `arctan` jump: the total is
  continuous across an edge but individual terms are not, so autodiff may be wrong
  exactly at degenerate points;
- `z = 0` (`asinh`, `1/z`): use the double-`where` trick to avoid NaN gradients;
- test the degenerate points explicitly.

Reference derivatives: high-precision finite differences in mpmath (complex-step
is awkward through the clipping).

## Warp, adjoints and forward mode

- Derivatives w.r.t. `x` are the gradient identity itself.
- Derivatives w.r.t. vertex positions (shape derivative, for deforming or optimised
  geometry) are also edge-local: derive by hand (Maple) and implement as explicit
  custom adjoint functions rather than taping loops and clip logic.
- **Open:** whether the installed Warp supports forward-mode derivatives. To my
  knowledge its tape is reverse-mode only; check the docs for the installed version
  before committing to a forward-mode design.

## Order of work

1. Derivation files + Maple checks for primitives and moments.
2. Stage 1 reference + golden fixtures.
3. Stage 2/3 tier 1 library and tests.
4. Notebooks (per tier; FEM; tier comparison).
5. Torch, then Warp.
