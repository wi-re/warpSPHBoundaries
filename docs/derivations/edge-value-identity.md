# Edge reduction of the value integral (2D)

**Status:** [V] Maple (divergence step, half-plane equality with `λ_2(d)` symbolic) + mpmath vs polar quadrature, tiling, subdivision, continuity
**Tier(s):** 1, 2 · **Dimension:** 2D (3D version: `tet-face-edge-chain.md`)
**Depends on:** `../notation.md`, `edge-primitives.md`, `truncated-monomials.md`
**Implemented in:** `src/edgebound/edge/core.py::value` (stage 1, mpmath); `scripts/derivation_checks/edge_identity_check.py` (original quadrature-edge check, [V])
**Verified by:** `maple/11_truncated_monomials.mpl`, `maple/13_halfplane.mpl`, `tests/edge/test_value.py`, `tests/edge/test_degenerate.py`, `tests/edge/test_fixtures.py`

## 1. Statement

For a polygon `T` (triangle, or any simple polygon) and normalised radial kernel `W` with
support 1, `M(r) = ∫_0^r t W(t) dt`, `2π M(1) = 1`:

$$\int_T W(|x-x'|)\,dA' \;=\; \mathbb 1[x\in T] \;+\; \sum_e z_e \int_{\text{chord}_e}\frac{M(r)-M(1)}{r^2}\,ds .$$

## 2. Assumptions and validity

- `W` radial, compactly supported on `r <= 1`, `M` continuous.
- Polygon edges are straight; `T` simple (non-convex allowed: the indicator is then the exact
  winding/crossing test, `geometry.locate`; for convex polygons it equals the `z_e`-sign test —
  checked in `tests/edge/test_value.py`).
- Degenerate placements are *defined*, not excluded (§4): `x` on an edge, at a vertex, on an edge line.

## 3. Derivation  (steps 1–3 re-derived and checked)

1. `∇·( y M(r)/r² ) = W(r)` in 2D (`y = x'−x`). **Maple:** symbolic for monomial `M`, `n = 0…12`
   (`maple/11`), plus `∇·(y/r²) = 0` for `r > R`.
2. Divergence theorem on `T`. The field is singular at `x` but `M(r) ~ r²`, so the small-circle flux
   `∮ M(ε)/ε² · ε dθ → 0` vanishes: no extra term. `n_e·y = z_e` on the edge line.
   `∫_T W = Σ_e z_e ∫_edge M(r)/r² ds`.
3. Split `M = (M − M(1)) + M(1)`; the first part is compact. The second part:
   `M(1) Σ_e z_e ∫_edge ds/r² = M(1) Σ_e Δ_e arctan` is `M(1)` × (total angle subtended by the closed boundary)
   `= 2π M(1) 1[x∈T] = 1[x∈T]`. Distant edges contribute only through the indicator.
   (Boundary cases, §4: the total angle at `x` on an edge is `π`, at a vertex the interior angle.)

Checked in Maple: half-plane (one edge at distance `d`) for each monomial block, symbolic in `d, R`,
both signs of `z`; wedge at a vertex vs polar reference (all chord-clipping cases).

## 4. Special cases and limits

- **Half-plane:** one infinite edge at distance `d` (`n = (0,−1)`, `z = −d`, indicator 0,
  chord `[−L, L]`) equals `λ_2(d)` of `docs/derivation.md` §3 for **all four kernels** —
  Maple symbolic (`simplify(edge − λ_2) = 0`, both cubic branches), mpmath to 1e-35 for
  `d ∈ {0, 1e-12, 1e-3, …, 1−1e-12, 1}`; `d ≥ 1 → 0`; `d < 0` (x inside the solid) gives
  `1 − λ_2(|d|)` (indicator 1, `z = +|d|`).
- **Conventions on an edge / at a vertex** (exact orientation predicate, see `notation.md`):
  `x` in the open interior of an edge ⇒ indicator `1/2`, that edge contributes `0`
  (`z = 0`, atan jump averaged to 0); `x` at a vertex ⇒ indicator `= interior angle / 2π`, both
  incident edges contribute `0`; `x` on an edge *line* outside the segment ⇒ indicator by the
  winding test, atan difference over the chord is 0 anyway. The total is continuous: value at `x`
  on the edge equals the limits from both sides (`z = ±1e-2…1e-40`, agreement `O(z)`), and
  `x` on an edge gives exactly `λ_2(0) = 1/2` for the half-plane.
- Mesh covering the whole support: sum over all elements = 1 (and = `λ`-disk moments for `k ≥ 1`),
  including `x` at a mesh vertex (fan of 8) and on a mesh edge.

## 5. Checks

| check | how | tolerance | status |
|---|---|---|---|
| random triangles vs area quadrature (C4) | `scripts/derivation_checks/edge_identity_check.py` | 1e-12 | [V] |
| random triangles/points vs polar oracle, 4 kernels | `tests/edge/test_value.py` | 1e-30 asserted (≈1e-42) | [V] |
| divergence step, symbolic | Maple (`maple/11`) | exact | [V] |
| half-plane equals `λ_2(d)` (symbolic, 4 kernels, cubic branches A/B) | Maple `simplify = 0` + 50-digit spot values | exact / 5e-47 | [V] |
| half-plane equals `λ_2(d)` incl. `d = 1e-12, 1−1e-12, 0, ≥1, <0` | mpmath, `test_half_plane_equals_planar2d` | 1e-35 | [V] |
| tiled mesh covering support sums to 1 (x generic, on edge, at vertex, fan) | mpmath | 1e-35 | [V] |
| `x` crossing an edge: total continuous (`eps = 1e-6…1e-30`) | `test_crossing_an_edge_total_is_continuous`, `test_degenerate.py` | `O(eps)` | [V] |
| sub-triangle splitting invariance (2 and 4 pieces) | mpmath | 1e-35 | [V] |
| `x` on edge / at vertex / on edge line / `z → 1e-30` / tiny chords / `L_T/h = 1e-6` vs polar | `tests/edge/test_degenerate.py` | 1e-28…1e-30 | [V] |
| non-convex L-shape = sum over triangulation; clockwise input | mpmath | 1e-33 | [V] |
| sign-test indicator = exact winding indicator (random + degenerate) | mpmath | exact | [V] |

## 6. Known failure modes

- `z_e → 0`: handled (no `asinh` blow-up; `atan2` form); `z = 0` exactly requires the exact
  predicate — the stage-1 code computes the *sign* of `z_e` from exact rationals and forces `z = 0`
  when the numerator vanishes.
- **Elements `≪ h` in floating point:** for `x` inside / at a vertex of a tiny element the formula
  evaluates `1 − (1 − area·W(0))`: absolute error ≈ 3e-14 in float64 (53-bit emulation), i.e. the
  value is lost for `L_T/h ≲ 1e-3…1e-6`. Fix for production (not implemented here): when the whole
  polygon lies inside the support, skip the split `M = (M−M(1)) + M(1)` and use `z ∫ M(r)/r² ds`
  (no indicator, no atan; `M(r)/r²` is a polynomial for Wendland), or a short Gauss rule.
  Gradients and odd moments are unaffected (pure edge integrals).
- Orientation predicate mismatch between the indicator and `z_e`: removed by construction (both from
  the same exact rational cross products).

## 7. Open questions

Production predicate in float: how to keep `z_e`-sign and indicator consistent for `x` within
rounding of an edge (cheap robust predicate, or accept `O(eps)` jump because the total is continuous?).
The unsplit form for small elements: switching criterion and continuity of the switch.
