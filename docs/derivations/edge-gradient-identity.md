# Edge reduction of the gradient (2D)

**Status:** [V] Maple (half-plane symbolic, per block and per kernel) + mpmath (polar quadrature, finite differences)
**Tier(s):** 1, 2 · **Dimension:** 2D
**Depends on:** `edge-value-identity.md`
**Implemented in:** `python/edgebound/core.py::gradient`, `::moment_gradient` (stage 1); `edge_identity_check.py` [V, quadrature edges]
**Verified by:** `maple/11_truncated_monomials.mpl`, `maple/13_halfplane.mpl`, `tests/edge/test_gradient.py`, `tests/edge/test_moments.py`

## 1. Statement

$$\nabla_x\!\int_T W(|x-x'|)\,dA' \;=\; -\sum_e n_e \int_{\text{chord}_e} W(r)\,ds ,$$

and the first moment

$$\int_T y\,W\,dA \;=\; -\sum_e n_e\int_{\text{chord}_e}\Psi(r)\,ds,\qquad \Psi(r)=\int_r^1 tW(t)\,dt .$$

More generally `∂_{x_j} ∫_T y^α f = −Σ_e n_{e,j} ∫_chord y^α f ds`.

## 2. Assumptions and validity

As for the value identity. Both are **purely local**: only edges whose chord
is non-empty contribute; no indicator term (the gradient of `1[x∈T]` is zero
almost everywhere, and the chord jump at `x` crossing an edge cancels it). The gradient is
continuous across an edge (the chord is), so there is no convention needed on an edge or at a
vertex.

## 3. Derivation

`∂_{x_j}∫_T g(x'−x) dA' = −∫_T ∂_{y_j} g = −∮ n_j g ds` (Reynolds / Gauss–Green for a bounded
function `g`, jumps included: it holds **per truncated block**, no cancellation between blocks of a
continuous kernel is required). First moment: `y W = ∇_y(−Ψ)` with `Ψ` compact.

## 4. Special cases and limits

Half-plane (solid `{y > d}`, `n = (0,−1)`): `∇λ = (0, ∫_{−L}^{L} W ds)` and `∫ W ds = −dλ_2/dd`
(Maple symbolic, 4 kernels).

## 5. Checks

| check | how | tolerance | status |
|---|---|---|---|
| vs area quadrature (C4, 20 random triangles) | `edge_identity_check.py` | 1e-12 | [V] |
| vs independent polar quadrature of `∫ (−W'/r) y dA` (all 4 kernels, random) | `test_gradient.py` | 1e-30 asserted | [V] |
| vs 4th-order finite differences of the value identity, 80 digits | `test_gradient.py` | 1e-25 | [V] |
| per block, half-plane symbolic: `−dV/dd = 2 I_n(L)`, `n = 0…12` | Maple | exact | [V] |
| per kernel, half-plane vs `−dλ_2/dd` (`diff` of the exported closed forms; w2/w4/w6, cubic A/B) | Maple symbolic (`simplify = 0`) + 50 digits | exact / 7e-46 | [V] |
| same, mpmath (`d = 0.01 … 0.95`; tangential component 0) | `test_half_plane_gradient_is_minus_dlambda_dd` | 1e-30 | [V] |
| edge tangent to the knot circle `R = ½` (cubic) / chord ~1e-13 | `test_gradient_across_knot_radius_cubic`, `test_degenerate.py` | 1e-30 | [V] |
| `x` on an edge / at a vertex vs polar | `test_degenerate.py` | 1e-30 | [V] |
| first moment (`k = 1`) | `maple/12`, `maple/13`, `test_moments.py` | see `moments-recursion.md` | [V] |

## 6. Known failure modes

Float32/autodiff traps listed in `../backends-and-verification.md` (clipping, jumps of individual terms).
In float64 the 53-bit emulation gives ≲ 6e-14 absolute (largest for generic cases: cancellation among
the large alternating kernel coefficients).

## 7. Open questions

Shape derivative (w.r.t. vertices): edge-local, to be derived for the adjoint
(see `../backends-and-verification.md`).
