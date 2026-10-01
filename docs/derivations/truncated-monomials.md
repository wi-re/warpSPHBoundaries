# Truncated-monomial representation of kernels

**Status:** [D] (algebra only; Wendland and cubic spline decomposition to be checked in Maple)
**Tier(s):** all · **Dimension:** 2D, 3D
**Depends on:** `../notation.md`, `docs/derivation.md` §1 (kernel table)
**Implemented in:** `monomial_edge_forms.py`; kernel table in `python/curvbound/kernels.py`
**Verified by:** not yet (planned: `tests/test_monomial_decomposition.py`)

## 1. Statement

Any piecewise polynomial kernel is

$$W(r)=\sum_j q_j(r)\,\mathbb 1[r\le R_j],\qquad q_{\rm last}=\text{outer piece},\ q_j=\text{piece}_j-\text{piece}_{j+1},$$

so every integral is a linear combination of building blocks
`f_{n,R}(r) = r^n 1[r <= R]`.

- Wendland: a single `R = 1`.
- Cubic spline (2D, support 1): `W = σ [2(1-q)³_+ - 8(½-q)³_+]`, `σ = 40/(7π)`,
  i.e. `R = 1` and `R = ½`; check: `1 - 6q² + 6q³` on `q <= ½`.

Per-block value (`α = 0`):

$$\int_T r^n \mathbb 1[r\le R] = \mathbb 1[x\in T]\frac{2\pi R^{n+2}}{n+2}+\sum_e\frac{1}{n+2}\Big[z_e\,I_n(s)-R^{n+2}\arctan(s/z_e)\Big]_{\text{chord}} .$$

Gradient per block: `∇_x ∫_T r^n 1[r<=R] = -Σ_e n_e [I_n(s)]_chord`.

## 2. Assumptions and validity

Continuous kernel (so the circle jumps at `R_j` cancel in the gradient);
polynomial pieces. Non-polynomial kernels are out of scope for the exact
tiers.

## 3. Derivation

Apply the value/gradient identities to each block. After summing a
normalised kernel the indicator weights sum to `1[x∈T]` and the `arctan`
coefficients sum to `M(R)`.

## 4. Special cases and limits

Wendland and cubic spline must reproduce the kernel table values and
`∫ W = 1`.

## 5. Checks

| check | how | tolerance | status |
|---|---|---|---|
| decomposition reproduces `W(r)` pointwise | Maple `simplify` | exact | [P] |
| normalisation `∫ W = 1` from blocks | Maple | exact | [P] |
| per-block value vs polar-coordinate reference | mpmath (radial breakpoint at `R`) | 1e-30 | [P] — the brute-force reference in `monomial_edge_forms.py` timed out |

## 6. Known failure modes

Brute-force Cartesian reference is slow because of the discontinuity at
`r = R`; use a polar reference.

## 7. Open questions

None specific.
