# FEM nodal weights: P0 to P3 (and what extends beyond tier 1)

**Status:** [D] (k = 1, i.e. P1 interpolant and gradient, reduces to verified identities; P2/P3 and conditioning [P])
**Tier(s):** 1 (nodal re-expansion); the moment machinery is shared by 2, 3, 4 · **Dimension:** 2D
**Depends on:** `moments-recursion.md`, `edge-gradient-identity.md`
**Implemented in:** not yet
**Verified by:** not yet (checks below)

## 1. Statement

Field on `T`: `A(x') = Σ_i A_i N_i(x')`, `N_i` Lagrange of degree `p` (P3 bubble node
is just another node). Re-expand about the evaluation point:

$$N_i(x+y)=\sum_{|\alpha|\le p}B_{\alpha i}(x)\,y^\alpha,\qquad B_{\alpha i}=\partial^\alpha N_i(x)/\alpha! .$$

(`N_i` are polynomials in barycentrics and `λ_j(x+y) = λ_j(x) + ∇λ_j·y`.)

- Interpolant: `⟨A⟩(x) = Σ_i A_i w_i`, `w_i = Σ_α B_{αi} m_α`.
- Gradient `∫_T A ∇_x W`: `Σ_i A_i Σ_α B_{αi} g_α`, with
  `g_α = ∫_T y^α ∇_x W = -Σ_e n_e ∫_chord y^α W ds + Σ_j α_j e_j m_{α-e_j}`.
- Difference / symmetric SPH forms: swap nodal values (`f_i = ρ_i(A_i - A(x))`);
  weights unchanged.
- Two-way coupling: the same weights distribute the reaction to the nodes.
- Variable boundary density: `ρ A` has degree `2p` → moments up to `2p`, or treat
  `ρ A` as its own nodal field (inexact at degree `p`).
- Non-unisolvent samples (quadrature points, MLS samples): L2-project to `P_p`
  per element, `B` via pseudo-inverse; exact with respect to the projection.

## 2. Assumptions and validity

- Conforming Lagrange elements; the field is a polynomial of degree `<= p` on each
  element (this is the *definition* of "exact at order p").
- Tier 2 (data on boundary only) and tiers 3/4 do **not** get an exact FEM order:
  the interior extension of the field is a modelling choice, so "order" refers to
  the extension polynomial, not to nodes in a volume element.
- Conditioning: `B_{αi} ~ L_T^{-|α|}`, `m_α ~ h^{|α|}` → cancellation `~(h/L_T)^p`
  for elements much smaller than `h`.

## 3. Derivation

Expansion of `N_i` about `x` (finite polynomial, exact), then linearity of the
integral in the field, then the moment results of `moments-recursion.md`.

## 4. Special cases and limits

P0: `w = m_0` (constant per element). P1: `w_i = λ_i(x) m_0 + ∇λ_i · m_1`.
Mesh covering the support: sums reduce to disk integrals of polynomials.

## 5. Checks (per order `p ∈ {0,1,2,3}` and per kernel)

| # | check | how | tolerance | status |
|---|---|---|---|---|
| 1 | partition of unity: all `A_i = 1` ⇒ `Σ w_i = λ` | mesh covering support and partial mesh | 1e-30 mp | [P] |
| 2 | polynomial reproduction: nodal values from `A` of degree `<= p` ⇒ `⟨A⟩ = ∫ A W` | disk integral by closed form for covering mesh; polar quadrature for partial | 1e-30 mp | [P] |
| 3 | gradient reproduction: linear `A` ⇒ known constant; `∇ ∫ A W` for polynomial `A` | same | 1e-30 mp | [P] |
| 4 | refinement invariance: split a triangle into 2/4 sub-triangles (field degree `p` represented exactly) | same fields | 1e-30 mp | [P] |
| 5 | continuity: continuous `P_p` field ⇒ result continuous as `x` crosses a shared edge | sweep | 1e-12 f64 | [P] |
| 6 | degree exceeded: field of degree `p+1` ⇒ error appears and scales as expected | convergence under refinement | observed order | [P] |
| 7 | conditioning: error vs `L_T/h` for each `p` | mpmath reference vs f64/f32 | report only | [P] |

Check 2 is the test in the setup of the 2025 boundary paper (Winchenbach & Kolb,
arXiv:2507.21686; PDF in `warpSPH/literature/`) — **read it and import its exact
test cases before fixing the tolerances**; the table above is not yet taken from it.

## 6. Known failure modes

Elements `<< h` with `p >= 2` in float32; non-unisolvent data.

## 7. Open questions

Mitigations: scale `y` by `h`, hybrid quadrature for elements entirely inside the
support, edge-local coordinates. Which one is needed is decided by check 7.
