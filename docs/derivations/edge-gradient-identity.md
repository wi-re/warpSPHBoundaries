# Edge reduction of the gradient (2D)

**Status:** [V] numerically (same script as the value identity)
**Tier(s):** 1, 2 · **Dimension:** 2D
**Depends on:** `edge-value-identity.md`
**Implemented in:** `edge_identity_check.py` [V, quadrature edges]
**Verified by:** `edge_identity_check.py`; planned `tests/test_edge_gradient.py`

## 1. Statement

$$\nabla_x\!\int_T W(|x-x'|)\,dA' \;=\; -\sum_e n_e \int_{\text{chord}_e} W(r)\,ds ,$$

and the first moment

$$\int_T y\,W\,dA \;=\; -\sum_e n_e\int_{\text{chord}_e}\Psi(r)\,ds,\qquad \Psi(r)=\int_r^1 tW(t)\,dt .$$

## 2. Assumptions and validity

As for the value identity. Both are **purely local**: only edges whose chord
is non-empty contribute; no indicator term (the gradient of `1[x∈T]` is zero
almost everywhere, and the chord jump at `x` crossing an edge cancels it).

## 3. Derivation

`∇_x` acts on `W(|x - x'|)` as `-∇_{x'} W`, so the volume integral becomes a
boundary flux `-∮ n W ds`. For the moment, `y W = ∇_{y}(−Ψ)` with `Ψ`
compact, so the same step applies. Truncated-monomial kernels: the circle
jumps at `r = R_j` cancel between the pieces of a continuous kernel
(`HANDOFF.md §4`).

## 4. Special cases and limits

Half-plane: `∇λ = n · W-line-integral` must equal `-d λ_2/d d` along the
normal from `docs/derivation.md` (differentiate the closed form in Maple).

## 5. Checks

| check | how | tolerance | status |
|---|---|---|---|
| vs finite differences of the value identity | mpmath 40 digits | 1e-25 | [P] |
| vs area quadrature | `edge_identity_check.py` | 1e-12 | [V] |
| half-plane vs derivative of `λ_2(d)` | Maple `diff` + mpmath | 1e-30 | [P] |

## 6. Known failure modes

Same as the value identity; plus kernels with non-continuous value at a knot
(none of the supported kernels).

## 7. Open questions

Shape derivative (w.r.t. vertices): edge-local, to be derived for the adjoint
(see `../backends-and-verification.md`).
