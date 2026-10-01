# Truncated-monomial representation of kernels

**Status:** [V] Maple (decomposition, normalisation, per-block value/gradient: `maple/11_truncated_monomials.mpl`) + mpmath vs polar oracle
**Tier(s):** all · **Dimension:** 2D, 3D
**Depends on:** `../notation.md`, `docs/derivation.md` §1 (kernel table), `edge-primitives.md`
**Implemented in:** `python/edgebound/kernels.py` (exact blocks, built from `curvbound.kernels`), `python/edgebound/core.py::block_value/block_grad`
**Verified by:** `maple/11_truncated_monomials.mpl`, `tests/edge/test_kernels.py`, `tests/edge/test_value.py`

## 1. Statement

Any piecewise polynomial kernel is

$$W(r)=\sum_j q_j(r)\,\mathbb 1[r\le R_j],\qquad q_{\rm last}=\text{outer piece},\ q_j=\text{piece}_j-\text{piece}_{j+1},$$

so every integral is a linear combination of building blocks
`f_{n,R}(r) = r^n 1[r <= R]`.

- Wendland: a single `R = 1` (coefficients of `C2·(1−q)^n P(q)` expanded in `q`).
- Cubic spline (2D, support 1): `W = C2 [ (1−q)³ − 4 (½−q)³ 1[q ≤ ½] ]`, `C2 = 80/(7π)`,
  i.e. blocks `R = 1` (outer piece) and `R = ½` (inner − outer). Equivalent to the HANDOFF form
  `σ[2(1−q)³ − 8(½−q)³]`, `σ = 40/(7π)`; on `q ≤ ½` the sum is `½ − 3q² + 3q³`.

Per-block value (`α = 0`):

$$\int_T r^n \mathbb 1[r\le R] = \mathbb 1[x\in T]\frac{2\pi R^{n+2}}{n+2}+\sum_e\frac{1}{n+2}\Big[z_e\,(I_n(s_1')-I_n(s_0')) - R^{n+2}\,\Delta\!\arctan\Big]_{\text{chord}} ,$$

where `[s_0', s_1']` is the chord clipped to radius `R` and `Δarctan = atan2(z(hi−lo), z²+hi·lo)`.
Per-block gradient: `∇_x ∫_T r^n 1[r<=R] = −Σ_e n_e [I_n(s)]_chord` — **exact for each block on its own**
(Gauss–Green for a bounded function with a jump; no cancellation between blocks is needed, contrary
to the wording of HANDOFF §4).

Indicator weights and atan coefficients are **per support radius**: for block group `j`
the weight is `2π M_j(R_j)` and the atan coefficient `M_j(R_j)` with `M_j(R_j) = Σ_n c_n R_j^{n+2}/(n+2)`.
They are exact rationals/π: Wendland `1`; cubic spline `8/7` (R = 1) and `−1/7` (R = ½) —
**negative** weight for the inner block, they sum to 1. (An equivalent form with a *single*
`arctan` per edge splits the chord of the full kernel at the knot radii and uses the full-kernel
`M(1) = 1/(2π)`; it is a reorganisation of the same sum.)

## 2. Assumptions and validity

Polynomial pieces. The kernel may be discontinuous across a knot for the value/gradient
identities (the blocks are separate), but all supported kernels are continuous. Non-polynomial
kernels are out of scope for the exact tiers.

## 3. Derivation

Apply the value/gradient identities (`edge-value-identity.md`, `edge-gradient-identity.md`)
to `f = r^n 1[r ≤ R]`: `M(r) = min(r,R)^{n+2}/(n+2)`, so `(M(r) − M(R))/r² = (r^n − R^{n+2} r^{-2})/(n+2)`
on the chord, giving `z I_n − R^{n+2} arctan(s/z)`. At `z = 0` an edge contributes `0`
(both terms; one-sided atan limits average to 0) and the indicator carries the half/angle weight.

## 4. Special cases and limits

Normalisation `∫ W = 1` from the blocks (exact). Block value at a vertex of a wedge reproduces the
sector `θ R^{n+2}/(n+2)` when the opposite edge lies outside `R`.

## 5. Checks

| check | how | tolerance | status |
|---|---|---|---|
| decomposition reproduces `W(r)` pointwise (cubic, w2, w4, w6; also `½ − 3q² + 3q³`) | Maple `expand` | exact | [V] |
| decomposition at rational `r` incl. the knot | `tests/edge/test_kernels.py` (exact `Fraction`) | exact | [V] |
| normalisation `∫ W = 1` from blocks, all kernels | Maple + exact `Fraction` | exact | [V] |
| `div(y M/r²) = r^n` inside, `div(y/r²) = 0` outside, `n = 0…12` | Maple symbolic | exact | [V] |
| half-plane, `x` outside and inside, symbolic in `d, R` (derivative in `R` matches the polar integrand; `E(R=d)=0`), `n = 0…12` | Maple symbolic | exact | [V] |
| per-block value vs polar reference, wedge with `x` at a vertex, all clip cases (648 configs, `n ≤ 11`, `R = 1, ½`) | Maple 40 digits | worst 3.4e-40 | [V] |
| per-block gradient, half-plane: `−dV/dd = 2 I_n(L)`, `n = 0…12`, `R > d` symbolic | Maple symbolic | exact | [V] |
| kernel value vs independent polar oracle, random triangles / points, all 4 kernels | `tests/edge/test_value.py` | 1e-30 asserted (≈1e-42 observed) | [V] |

## 6. Known failure modes

- The kernel's monomial coefficients are large and alternate in sign (Wendland w6: |c_n| up to ~10³);
  evaluating the polynomial parts in float64 costs ≈ 1e-14 absolute (measured, see
  `../backends-and-verification.md`). Use exact combined weights (indicator/atan coefficients are
  exact rationals) and, if needed, a Bernstein/`(1−q)`-form for the polynomial edge terms in float32.
- **`monomial_edge_forms.py` has a bug:** its cubic-spline term list uses `−4σ(½−q)³` for the
  `R = ½` block, but `W = σ[2(1−q)³ − 8(½−q)³]`, so the coefficient must be `−8σ` (equivalently
  `−4 C2`). With `−4σ` the kernel at `q = 0` is `1.5σ` instead of `σ`. The script was never
  run to completion (its reference timed out), so this went unnoticed. The new
  `python/edgebound/kernels.py` builds the blocks programmatically from the exact pieces.

## 7. Open questions

Whether a single-atan-per-edge organisation (knot splitting of the polynomial part) is cheaper on
GPU than one atan per radius group (it is for the cubic: 1 vs 2 per edge).
