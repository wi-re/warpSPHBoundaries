# Tier 2 in 2D: closed boundary polylines (data on the boundary only)

**Status:** [V] (non-convex polygons in numpy; polygon → exact disk convergence; closed-boundary additivity)
**Tier(s):** 2 · **Dimension:** 2D
**Depends on:** `edge-value-identity.md`, `edge-gradient-identity.md`, `moments-recursion.md`, `fem-nodal-weights.md`
**Implemented in:** `src/edgebound/edge/np2d.py` (general simple-polygon indicator), `core.py` (stage 1), `oracle.py::polar_disk_*` (exact disk oracle)
**Verified by:** `tests/edge/test_tier2.py`

## 1. Statement

A solid bounded by a closed polyline (no triangulated interior, boundary data only) is a simple polygon `S`. Every result of tier 1 holds for
`S` as is — value, gradient, moments, polynomial extension fields — because the edge identities need only the boundary and the indicator of
`x` in `S`:

$$\int_S W = \mathbb 1[x\in S] + \sum_{e\in\partial S} z_e\int_{\rm chord}\frac{M(r)-M(1)}{r^2}ds ,$$

and interior edges of any triangulation cancel pairwise (checked: `L = square + rectangle`, 1e-12). Boundary-only data enter as a polynomial extension
(MLS contact-point extrapolation of degree `q` ⇒ moments up to `k = q`, `moments-recursion.md`); there is no per-element interior state.

## 2. Assumptions and validity

- `S` simple (non-self-intersecting), watertight, non-overlapping with other solids (overlaps double count: see `HANDOFF.md §10`).
- Indicator = crossing number of the ray `y = 0, x > 0` computed from the same x-relative coordinates / the same computed `z_e` as the edge terms
  (so a rounding error moves `x` consistently: continuity is kept); `x` on an edge → ½, at a vertex → interior angle `/2π` (reflex angles allowed).

## 3. Derivation

None beyond tier 1; the only new ingredient is the general-polygon indicator (stage 1 uses exact rational crossing number, stage 2 the float analogue,
branch-free: `where` over `vertex / edge / crossing parity`). For a *curved* boundary approximated by a polyline of `N` edges the error is a pure
geometry-discretisation error (the identities are exact for the polygon).

## 4. Special cases and limits

`N → ∞` for an inscribed regular `N`-gon: value and gradient converge to the exact disk values at `O(N^{-2})`; the combination
`(inscribed + 2·circumscribed)/3` (area deficit : excess = 2 : 1) at `O(N^{-4})`. The exact reference is the polar-coordinate disk oracle (exact ray/circle
intersections, breakpoints at tangent angles and at crossings of the kernel radii).

## 5. Checks

| check | how | tolerance | status |
|---|---|---|---|
| numpy general-polygon indicator = stage 1 (L-shape, star, arrow; interior / exterior / on edge / at vertex / reflex notch) | `test_nonconvex_numpy_matches_stage1` | 1e-11 | [V] |
| closed-boundary additivity (L = square + rectangle) | `test_polygon_union_of_two_polygons_sharing_an_edge` | 1e-12 | [V] |
| inscribed N-gon → exact disk, value and gradient, `R = 0.4, 1, 3`, `d = 0.1…0.3`, cubic and w4 | `test_ngon_converges_to_exact_disk_value_and_gradient` | order 2 (1.7…2.4); combination order ≥ 3.3 | [V] |
| `x` inside the solid; support engulfed by the disk (exactly 1) | `test_ngon_x_inside_disk_and_complement` | 3e-5 / 1e-13 | [V] |
| huge disk → half-plane `λ_2(d)` (oracle sanity) | script (`R = 10^6`: −3e-8 ≈ `O(κ)`) | — | [V] |
| linear/quadratic boundary extension fields, moments `k ≤ 2` vs exact disk | `test_linear_boundary_field_moments...` | 2e-4, order 2 | [V] |
| exact disk covering the support = 1 | oracle | 1e-30 | [V] |

## 6. Known failure modes

- Polygon vertices / edges not resolving curvature: `O(N^{-2})` — this is exactly the regime where tier 3 (curvature expansion) is the cheaper model.
- Overlapping or touching solids (double counting), self-intersections: not handled.
- numpy FEM weights (`np_fem`) remain triangle-only.

## 7. Open questions

Adaptive polyline refinement vs the tier 3 expansion at the same accuracy (cost comparison belongs to the tier-selection study).
