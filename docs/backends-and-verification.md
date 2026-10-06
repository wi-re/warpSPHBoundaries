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
| numpy f32 (stable Chebyshev quadrature 8×6) | 5e-7 (measured 8e-8) | 2e-6 (measured 3e-7) | moments 2e-7 (measured 3e-9); per class: `exactness-and-approximations.md` |
| numpy f32 (closed form) | 1e-4 (measured 6e-5) | 5e-4 (measured 3.5e-4) | limited by monomial-coefficient cancellation, see below |
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
- **Resolved (Warp 1.17):** reverse-mode `wp.Tape` + `wp.autograd.jacobian` (from backward passes) only; no forward mode. Explicit adjoint kernels are used
  (`../docs/autodiff-and-gpu.md`).

## Status (Phase 1, 2D tier 1/2)

- Stage 0 (Maple): `maple/10`–`13` — primitives, truncated monomials, moments recursion, half-plane link
  to `λ_2(d)`; see the check tables in `derivations/`.
- Stage 1 (mpmath): `src/warpSPHBoundaries/` + independent polar oracle (`oracle.py`, shares no code with
  `core.py`) + `tests/edge/` (all green). Reference accuracy ≈ 1e-35 (40 dps + 20 guard digits); oracle agreement
  asserted at 1e-30.
- Golden fixtures: `tests/fixtures/edge2d_golden.json`, 92 cases (element and covering-mesh cases), 4 kernels,
  `value`, `grad`, moments `k = 1…4`, moment gradients `k = 0…2`, 40 digits, exact rational geometry (`"a/b"` strings),
  per-case `where` (inside / outside / edge / vertex) and exact expectations (disk moments as rationals, half-plane `λ_2`).
  Tags: `generic`, `h_scaled`, `x_on_edge`, `x_at_vertex`, `x_on_edge_line`, `z_to_0` (`1e-3…1e-40`),
  `tiny_chord` (grazing the support circle, edge ends within `1e-13` of the circle, knot circle of the cubic),
  `small_element` (`L_T/h = 1e-6…1`), `half_plane` (`d = 0, 1e-12, …, 1−1e-12, 1, 3/2, <0`), `engulf`,
  `covering_mesh` (also `x` at a mesh vertex / on a mesh edge / fan), `nonconvex`, `clockwise`, `outside_support`,
  `cubic_knot`. Not yet in the file: tier-3 cases (`d < 0` with curvature) and FEM weights (`p ≥ 1` nodal fields).
- Backends loading the fixtures should read the polygon as exact rationals (`Fraction(str)`), evaluate in their dtype,
  and compare with the tolerance table above; `h != 1` cases carry physical units.

### Float64 behaviour of the guard-free algorithm (53-bit emulation)

`python scripts/studies/precision_probe.py` runs the *same algorithm* with `mp.prec = 53` and no guard digits against the
40-digit fixtures (cubic and w4). Worst absolute errors over the 91 element cases (moments in units `h^k`):

| quantity | worst abs. error | where |
|---|---|---|
| value | 5.7e-14 | generic (coefficient cancellation of the kernel polynomial) |
| gradient | 5.8e-14 | generic, `x` at a vertex |
| `m_1` / `m_2` / `m_3` / `m_4` | 1.6e-14 / 3.0e-14 / 1.7e-14 / 2.8e-14 | half-plane with `x` inside the solid (even `k`) |
| tiny elements, `x` inside / at a vertex, value | ≈ 3e-14 **absolute** | `L_T/h = 1e-3`: relative 3e-8; `1e-6`: value lost (true value 7e-14) |

Everything else (`z → 1e-40`, `x` on an edge/vertex, chords of `1e-13`, grazing the circle) stays at the 1e-14
level, i.e. within the numpy-f64 tolerance table above (value 1e-12, gradient 1e-11) with a factor ≥ 20 margin.
The only genuine float64 failure is the tiny-element value/even-moment cancellation `1 − (1 − ε)`, with a
verified remedy (`derivations/edge-value-identity.md` §6; the unsplit polynomial form reaches 2e-16 relative at `L_T/h = 1e-6`).
float32 will need exact combined indicator weights and a better-conditioned polynomial basis (kernel coefficients are
up to ~10³ in magnitude).

### Stage 3 status (float32)

Measured float32 tolerance table per fixture class and the conditioning fix are in `exactness-and-approximations.md`:
exact Chebyshev compilation of the edge profiles + stable quadrature brings float32 to ≈ machine epsilon
(value 8e-8, gradient 3e-7, moments 3e-9 over all representable fixtures), a 700–1000× gain over the closed form in float32.

### Stages 4/5 status (torch, Warp)

torch autograd equals the analytic gradient and the (new) edge-local shape adjoint at generic and degenerate points (custom angle backward); Warp float64 CPU/CUDA kernels
match the fixtures and give ~120× over numpy for the value (`autodiff-and-gpu.md`).

### Stage 2 status

numpy float64 (`src/warpSPHBoundaries/edge/np2d.py`) meets the table above on every fixture: worst 6e-14 (value), 5e-13 (gradient),
1e-14 (moments) absolute. Details, float32 / quadrature variants and cost: `exactness-and-approximations.md`.

## Order of work

1. Derivation files + Maple checks for primitives and moments.
2. Stage 1 reference + golden fixtures.
3. Stage 2/3 tier 1 library and tests.
4. Notebooks (per tier; FEM; tier comparison).
5. Torch, then Warp.
