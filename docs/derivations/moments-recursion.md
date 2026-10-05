# Higher moments `m_α = ∫_T y^α f(r) dA`

**Status:** (a) [V] for `k ≤ 4` (Maple symbolic half-plane + wedge, mpmath vs polar quadrature); (b) [V] but **only for `z ≠ 0`** on every edge with a non-empty chord
**Tier(s):** 1, 2, 3, 4 · **Dimension:** any (2D written out)
**Depends on:** `edge-value-identity.md`, `edge-primitives.md`, `truncated-monomials.md`
**Implemented in:** (a) `src/edgebound/edge/core.py::block_moment`; (b) `core.py::block_moment_b` (cross-check only)
**Verified by:** `maple/12_moments_recursion.mpl`, `maple/13_halfplane.mpl`, `tests/edge/test_moments.py`, `tests/edge/test_degenerate.py`, `tests/edge/test_fixtures.py`

## 1. Statement

(a) **Compact-potential recursion (recommended).** With
`Φ[f](r) = -∫_r^R t f(t) dt` (compact, `Φ(R) = 0`, continuous), `y_i f = ∂_i Φ`
and for `α = β + e_i`:

$$\int_T y^{\beta+e_i} f\,dA=\sum_e n_{e,i}\int_{\rm chord}y^\beta\,\Phi\,ds-\beta_i\int_T y^{\beta-e_i}\,\Phi\,dA .$$

Degree drops by 2 per step. Odd `k` → purely edge integrals. Even `k` →
ends in one value-type integral (with `f` replaced by iterated `Φ`). For a block
`f = r^n 1[r ≤ R]`, `Φ = (r^{n+2} − R^{n+2})/(n+2)·1[r ≤ R]` (a combination of the blocks `(n+2, R)`
and `(0, R)`), so the recursion closes in the block family and only `I_m, J_m` with `m ≥ 0`
appear (`S_{j,m}` with `m = n+2, 0, …`). `Φ` is continuous at `R`, so no circle term appears
even though `f` jumps. Implementation: choose `i = 1` if `α_1 > 0` else `i = 2`; memoise on `(α, n, R)`.

(b) **Direct far-field form (cross-check).**

$$\int_T y^\alpha r^n\mathbb 1[r\le R] = \mathbb 1[x\in T]\Big(\oint_{S^1}\omega^\alpha\Big)\frac{R^{n+2+k}}{n+2+k}+\sum_e\frac{z_e}{n+2+k}\int_{\rm chord}(z_en_e+st_e)^\alpha\Big[r^n-R^{n+2+k}r^{-(2+k)}\Big]ds .$$

`∮ ω^α = 2π (a−1)!!(b−1)!!/(a+b)!!` for `α = (a,b)` both even, else 0.

**Gradient of moments** (purely edge-local, exact, no extra term):
`∂_{x_j} ∫_T y^α f = −Σ_e n_{e,j} ∫_chord y^α f ds`. (Not to be confused with the object
`g_α = ∫_T y^α ∇_x W = ∂_x m_α + Σ_j α_j e_j m_{α−e_j}` of the FEM weights, which carries the lower-order
term because `∇_x` there acts on `W` only; see `fem-nodal-weights.md`.) Consequence: for a mesh covering
the support, `m_α` is the full-space moment and its `x`-gradient vanishes (checked).

## 2. Assumptions and validity

As for the value identity; `f` a truncated monomial or sum of such. Exponent conventions as in `notation.md`
(`y = x' − x`, units of `h`; `m_α` scales as `h^k`).

## 3. Derivation

(a): pointwise `∂_i(y^β Φ) − β_i y^{β−e_i}Φ = y^α f` (Maple, symbolic) + divergence theorem with `Φ`
continuous; (b): pointwise `∇·(y P F) = P r^n` inside, `0` outside, with `F = r^n/e` and `F = R^e r^{-2-k}/e`,
continuous at `R` (Maple, symbolic) and `Σ_e z_e ∫_edge P(y) r^{-(2+k)} ds = 1[x∈T] ∮ω^α`
for the divergence-free far field.

## 4. Special cases and limits

`k = 0` reduces to the value identity. Mesh covering the whole support: `m_α` equals the exact rational
disk moment `disk_moment(kernel, α)` (the no-quadrature test; includes `x` at a mesh vertex / on a mesh edge).

**Finding on conventions (b) vs (a):** for `x` on an edge line (`z = 0`, chord non-empty) form (b) is
`0 · ∞`: `z ∫ P r^{-(2+k)} ds` has a finite, *direction-dependent* limit (it carries the partial angular
integral `∫ ω^α dθ` over the half circle), so the averaged convention "indicator ½, edge contributes 0"
used for `k = 0` is **wrong for (b) with `k ≥ 1`** (`∫_{half circle} sin θ ≠ ½ ∮ sin θ`). Form (a) has no such
problem: only the base (value) term needs the indicator/atan convention, which is exact. `core.block_moment_b`
raises at `z = 0`; `tests/edge/test_moments.py::test_b_is_undefined_on_an_edge_line` documents it.

## 5. Checks

| check | how | tolerance | status |
|---|---|---|---|
| (a) pointwise step identity, `n = 0…8`, all `|α| ≤ 4` (126 cases) | Maple symbolic | exact | [V] |
| (b) pointwise: `div(yPF) = P r^n` inside, 0 outside, continuity at `R` (105 cases) | Maple symbolic | exact | [V] |
| `∮ω^α` formula, `a, b ≤ 6` | Maple `int` | exact | [V] |
| half-plane, `x` outside **and** inside, `n ∈ {0,1,2,4}`, all `k ≤ 4`: `d/dR` of (a) and of (b) equals the polar integrand `R^{n+k+1} A_{ab}(d/R)`; `V(R=d)` start value; (a) = (b) as functions of `R` (360 cases, symbolic in `d,R`) | Maple, 40 digits (floor 1e-40) | exact / 1e-40 | [V] |
| wedge, `x` at a vertex (3 edges, two through `x` with `z = 0`, asymmetric chords), (a) vs polar, `k = 0…4`, `n ∈ {0,1,3}` | Maple 40 digits | worst 1.9e-37 | [V] |
| (a) vs polar quadrature, random triangles, `k ≤ 4` (cubic, w6) | `tests/edge/test_moments.py` | 1e-30 asserted (≈1e-43 observed) | [V] |
| (a) vs (b), `k = 2, 3` (and 1, 4), random triangles, all kernels | same | 1e-33 | [V] |
| (a) at `x` on an edge line, `x` on an edge, at vertices vs polar | `tests/edge/test_degenerate.py` | 1e-30 | [V] |
| full-support mesh vs exact disk moment, `|α| ≤ 4`, `x` generic / mesh vertex / on mesh edge | exact rational vs mpmath | 1e-33 | [V] |
| half-plane `m_(0,1)` vs direct radial integral `2∫_d^1 W r √(r²−d²) dr` (**replaces** the planned "vs derivative of λ_2": that is not an identity, `−λ_2' = 2∫W r/√(r²−d²)`) | Maple symbolic (w2/w4/w6, cubic A/B) + mpmath | exact / 1e-28 | [V] |
| gradient of moments vs 4th-order finite differences | mpmath 60 digits | 1e-25 | [V] |
| `h`-scaling: `m_α ∝ h^k`, `∂m_α ∝ h^{k−1}` | mpmath | 1e-35 | [V] |

## 6. Known failure modes

- (b) at `z → 0` (loses digits; undefined at 0) and the downward `I_{m<-2}` recurrence; use (a).
- Even `k` in floating point for tiny elements around `x` (same `1 − (1 − ε)` cancellation as the value
  identity; see `edge-value-identity.md` §6).
- High `k` with small elements: conditioning of the FEM re-expansion, see `fem-nodal-weights.md`.

## 7. Open questions

Non-radial `f` (anisotropic kernels) out of scope. Whether to precombine the per-block recursion
(`R^{n+2}` terms) into a per-kernel polynomial form for speed.
