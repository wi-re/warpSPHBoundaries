# Notation and conventions

Shared by every derivation in `docs/derivations/` and by all code. If a
derivation needs a different convention it must say so in its *Assumptions*
section. Status tags: **[V]** verified numerically · **[D]** derived, not yet
verified · **[P]** plan / open (same legend as `HANDOFF.md`).

## Kernel

- `W(r, h) = C_n / h^n · W_hat(r/h)`, `q = r/h`, **`h` is the support radius**
  (`W = 0` for `q > 1`). Matches the paper and warpSPHCore.
- Derivations work at `h = 1`; restore `h` by dimensional scaling
  (`lambda(d, h) = lambda(d/h, 1)` for the value; each derivative or moment
  of order `k` brings a factor `h^(k-1)` per the identities in the derivations).
- `C_n` normalises `∫_{R^n} W = 1`. Kernel table and constants:
  `docs/derivation.md` §1 (cubic spline, Wendland w2/w4/w6).
- Radial moments: `M(r) = ∫_0^r t W(t) dt` (2D), `M_3(r) = ∫_0^r t² W(t) dt`
  (3D); `Ψ(r) = ∫_r^1 t W(t) dt` (compact potential of `y W`).
- Truncated monomials: `f_{n,R}(r) = r^n 1[r <= R]`; any piecewise polynomial
  kernel is a linear combination of these.

## Geometry (2D, tiers 1 and 2)

- Evaluation point `x`; integration variable `x'`; `y = x' - x`; `r = |y|`.
- Edge `e = (p_e, q_e)` with unit tangent `t_e` and **outward** normal `n_e`
  (outward from the element, orientation fixed by the triangle's sign).
- `z_e = n_e · (p_e - x)`: signed distance to the edge line, **positive when
  `x` lies on the inner side**.
- Along the edge line `y = z_e n_e + s t_e`, `r = sqrt(s² + z_e²)`,
  `s_0 = t_e·(p_e - x)`, `s_1 = t_e·(q_e - x)`.
- **Exact orientation predicate.** Polygons are normalised to counter-clockwise order. `sign(z_e) = sign(cross(q−p, x−p))`
  is computed from exact rationals (`src/warpSPHBoundaries/edge/geometry.py`); `z_e = 0` exactly when `x` is on the edge line.
- **Indicator / degenerate placements** (value-type terms): `x` strictly inside → 1, outside → 0, in the open interior of
  an edge → `1/2`, at a vertex → interior angle / `2π`; an edge with `z_e = 0` contributes 0 to every atan term (average of
  the one-sided limits; `atan` over a chord is computed as `atan2(z (hi−lo), z² + hi·lo)`). Verified: the total is continuous
  in `x`. Do NOT use this averaged convention for the far-field moment form (b) (`moments-recursion.md` §4).
- Scaling with the support radius `h` (geometry divided by `h`): value `h⁰`, `∇_x` value `h⁻¹`, `m_α` `h^k`,
  `∇_x m_α` `h^{k−1}` (`k = |α|`).
- Chord for support radius `R`: if `|z_e| < R`, `L = sqrt(R² - z_e²)` and
  `s ∈ [max(s_0, -L), min(s_1, L)]`; otherwise the edge contributes nothing
  (value identity: the far-field flux cancels over a closed boundary).
- `1[x ∈ T]` uses the **same orientation predicate** as the signs of `z_e`, so
  the jump of the `arctan` term and the jump of the indicator cancel exactly
  when `x` crosses an edge.

## Geometry (tier 3, implicit surface; shared with `PLAN.md`)

- `d` = signed distance of the particle to the boundary, **positive outside the
  solid** (fluid side) — the paper's convention and the one used in
  `docs/derivation.md`. `d/h ∈ [0, 1]` is partial support from the outside;
  `d < 0` is the particle inside the solid (`lambda(-d) = 1 - lambda(d)` for a
  flat wall).
- Principal curvatures `κ_1, κ_2` and `H = (κ_1+κ_2)/2`, `K = κ_1 κ_2`.
  **Sign:** `κ > 0` when the solid is locally convex (sphere/disk of radius
  `R`: `κ = +1/R`). To be confirmed by the first tier 3 derivation; if it
  changes, change it here first.
- Dimensionless variables: `q = d/h`, `α_i = h κ_i`.
- Tubular coordinates (Tube Maps): `s` = distance along the normal from the
  closest boundary point. `J(s)` convention and its sign of `K` are
  **[P]** — see `derivations/tier3-curvature-2d.md`.

## Fields and operators

- Field on an element: `A(x') = Σ_i A_i N_i(x')`, Lagrange basis of degree `p`.
- Moments: `m_α(x) = ∫_T y^α W dA`, `|α| = k`.
- Gradient of a moment: `∂_{x_j} ∫_T y^α f = -Σ_e n_{e,j} ∫_chord y^α f ds`
  (purely edge-local) plus the lower-order term in the derivation.
- Differentiation is **with respect to `x`** unless stated; shape derivatives
  (w.r.t. vertices) are named explicitly.

## Numerical conventions

- Reference arithmetic is mpmath (30–50 digits). "Exact" in a test name means
  rational/`Fraction` or symbolic; "oracle" means an independent quadrature.
- Error tolerances per backend: see `backends-and-verification.md`.
