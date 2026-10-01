# Higher moments `m_α = ∫_T y^α f(r) dA`

**Status:** [D] (k = 1 verified as the gradient identity; k >= 2 not verified)
**Tier(s):** 1, 2, 3, 4 · **Dimension:** any (2D written out)
**Depends on:** `edge-value-identity.md`, `edge-primitives.md`, `truncated-monomials.md`
**Implemented in:** (a) not yet · (b) `monomial_edge_forms.py` [D]
**Verified by:** not yet (tasks 1–2 of `HANDOFF.md §12`)

## 1. Statement

(a) **Compact-potential recursion (recommended).** With
`Φ[f](r) = -∫_r^R t f(t) dt` (compact, `Φ(R) = 0`, continuous), `y_i f = ∂_i Φ`
and for `α = β + e_i`:

$$\int_T y^{\beta+e_i} f\,dA=\sum_e n_{e,i}\int_{\rm chord}y^\beta\,\Phi\,ds-\beta_i\int_T y^{\beta-e_i}\,\Phi\,dA .$$

Degree drops by 2 per step. Odd `k` → purely edge integrals. Even `k` →
ends in one value-type integral (with `f` replaced by iterated `Φ`). For
`f = r^n`, `Φ = (r^{n+2} - R^{n+2})/(n+2)` is still polynomial in `r`, so only
`I_m`, `J_m` with `m >= -2` appear.

(b) **Direct far-field form (cross-check).**

$$\int_T y^\alpha r^n\mathbb 1[r\le R] = \mathbb 1[x\in T]\Big(\oint_{S^1}\omega^\alpha\Big)\frac{R^{n+2+k}}{n+2+k}+\sum_e\frac{z_e}{n+2+k}\int_{\rm chord}(z_en_e+st_e)^\alpha\Big[r^n-R^{n+2+k}r^{-(2+k)}\Big]ds .$$

Needs `r^{-(2+k)}`: worse conditioning near edges; `∮ ω^α = 0` for odd `k`.

Gradient of moments: `∂_{x_j} ∫_T y^α f = -Σ_e n_{e,j} ∫_chord y^α f ds` (+ the
lower-order term written out in `fem-nodal-weights.md`).

## 2. Assumptions and validity

As for the value identity; `f` a truncated monomial or sum of such.

## 3. Derivation

(a): integrate by parts with the compact potential; (b): from
`∇·(y P F) = P[(d+k)F + rF']` with far-field `P y r^{-(d+k)}` divergence-free.

## 4. Special cases and limits

`k = 0`: reduces to the value identity. Mesh covering the whole support:
`m_α` equals the closed-form disk moment of the kernel
(`∮ ω^α` times radial integral) — the exact no-quadrature test.

## 5. Checks

| check | how | tolerance | status |
|---|---|---|---|
| (a) vs (b), `k = 2, 3` | Maple + mpmath | exact / 1e-30 | [P] |
| (a) vs polar quadrature, `k <= 4` | mpmath | 1e-30 | [P] |
| full-support mesh vs disk moment | exact | 1e-15 f64 | [P] |
| half-plane `m_1` vs derivative of `λ_2` | Maple | exact | [P] |

## 6. Known failure modes

(b) near `z → 0`; high `k` with small elements (see `fem-nodal-weights.md`).

## 7. Open questions

Non-radial `f` (anisotropic kernels) out of scope.
