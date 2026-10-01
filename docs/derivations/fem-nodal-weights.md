# FEM nodal weights: P0 to P3 (and what extends beyond tier 1)

**Status:** [V] for P0–P3 in 2D (stage 1 exact/mpmath, stage 2/3 numpy with conditioning remedies); checks 1–7 green; the 2025-paper test cases are not imported (paper PDF not available here)
**Tier(s):** 1 (nodal re-expansion); the moment machinery is shared by 2, 3, 4 · **Dimension:** 2D
**Depends on:** `moments-recursion.md`, `edge-gradient-identity.md`
**Implemented in:** `python/edgebound/fem.py` (stage 1), `python/edgebound/np_fem.py` (numpy, hybrid), `np2d.grad_moment`, `fem_fixtures.py`, `fem_study.py`
**Verified by:** `tests/edge/test_fem.py` (checks 1–6, exact/mpmath), `tests/edge/test_fem_np.py` (golden fixtures, float32, conditioning), `tests/fixtures/edge2d_fem_golden.json`

## 1. Statement

Field on `T`: `A(x') = Σ_i A_i N_i(x')`, `N_i` Lagrange of degree `p` (P3 bubble node
is just another node). Re-expand about the evaluation point:

$$N_i(x+y)=\sum_{|\alpha|\le p}B_{\alpha i}(x)\,y^\alpha,\qquad B_{\alpha i}=\partial^\alpha N_i(x)/\alpha! .$$

(`N_i` are polynomials in barycentrics and `λ_j(x+y) = λ_j(x) + ∇λ_j·y`; the basis is built exactly in
`Fraction`s and `B` is exact rational for rational geometry. Node order: vertices, then for each edge
(0,1),(1,2),(2,0) the nodes at 1/3 and 2/3 from the first vertex (P2: the midpoint), then the centroid.)

- Interpolant: `⟨A⟩(x) = ∫_T A W = Σ_i A_i w_i`, `w_i = Σ_α B_{αi} m_α`.
- Gradient `∫_T A ∇_x W = Σ_i A_i G_i`, `G_i = Σ_α B_{αi} g_α`, with
  `g_α = ∫_T y^α ∇_x W = ∇_x m_α + Σ_j α_j e_j m_{α-e_j}`, and `∇_x m_α = −Σ_e n_e ∫_chord y^α W ds`.
- Difference / symmetric SPH forms: swap nodal values (`f_i = ρ_i(A_i - A(x))`); weights unchanged.
- Two-way coupling: the same weights distribute the reaction to the nodes (`Σ_i w_i = λ` ⇒ total force is conserved).
- Variable boundary density: `ρ A` has degree `2p` → moments up to `2p` (the recursion is general), or treat
  `ρ A` as its own nodal field (inexact at degree `p`).
- Non-unisolvent samples (quadrature points, MLS samples): L2-project to `P_p` per element, nodal values from the
  pseudo-inverse; exact with respect to the projection (not implemented here; needs nothing beyond the weights).
- Pressure special cases (HANDOFF §8): contact-point MLS pressure → `m_0, g_0` only; MLS-linear → `k ≤ 1`.

## 2. Assumptions and validity

- Conforming Lagrange elements; the field is a polynomial of degree `<= p` on each element (this is the *definition*
  of "exact at order p").
- Tier 2 (data on boundary only) and tiers 3/4 do **not** get an exact FEM order: the interior extension of the field is a
  modelling choice, so "order" refers to the extension polynomial.

## 3. Derivation

Expansion of `N_i` about `x` (finite polynomial, exact), then linearity of the integral in the field, then the moment
results of `moments-recursion.md`. Lower-order term of `g_α`: `∂_{x_j}` acts on `y^α` as well in `∇_x m_α`, whereas `g_α`
has `∇_x` on `W` only, hence `+ α_j m_{α−e_j}`.

## 4. Special cases and limits

P0: `w = m_0`. P1: `w_i = λ_i(x) m_0 + ∇λ_i · m_1` (checked exactly). Mesh covering the support: sums reduce to exact rational
disk integrals of polynomials.

## 5. Checks (per order `p ∈ {0,1,2,3}`; kernels cubic, w2/w4/w6 where stated)

| # | check | how | tolerance | status |
|---|---|---|---|---|
| 1 | partition of unity: `Σ_i w_i = λ`, also summed over a covering mesh | `test_fem.py` | 1e-30 mp | [V] |
| 2 | polynomial reproduction: nodal values from `A` of degree `≤ p` ⇒ `⟨A⟩ = ∫ A W`: covering mesh vs EXACT rational disk integral (4 kernels); single element vs independent polar quadrature | `test_fem.py` | 1e-32 / 1e-30 | [V] |
| 3 | gradient reproduction: `∫ A ∇_xW = ∫ ∇A W` on the covering mesh (exact rational); single element vs polar quadrature of `−(W'/r) y` | `test_fem.py` | 1e-32 / 1e-30 | [V] |
| 4 | refinement invariance: split a triangle into 4, field of degree `p` represented exactly (value and gradient) | `test_fem.py` | 1e-33 mp | [V] |
| 5 | continuity: continuous `P_p` field on two triangles sharing an edge, `x` swept across it (`1e-3 … 1e-20`) | `test_fem.py` | `O(ε)` | [V] |
| 6 | degree exceeded: field of degree 3 interpolated at `p = 1, 2` on a refined covering mesh | `test_fem.py` | observed order `> p + 0.7` (≈ p + 1) | [V] |
| 7 | conditioning vs `L_T/h` for each `p`, float64/float32 | `fem_study.py`, `test_fem_np.py` | report + regression thresholds | [V] (§6) |
| – | golden fixtures: 19 cases × p 0…3 × 4 kernels (w, G) + covering-mesh exact reproduction | `edge2d_fem_golden.json` | f64 1e-10 / 1e-8; f32 2e-4 rel. | [V] |

## 6. Conditioning (check 7) — measured, and what removes it

Relative error `max|Δw_i| / max|w_i|` against the exact reference for the same float inputs (`python -m edgebound.fem_study`):
the **plain** edge-reduction + monomial re-expansion loses accuracy fast when `L_T ≪ h`, in two distinct ways, each with a
**provably exact** remedy (no approximation of the identities):

1. **Moment recursion for an element inside the support** — the R-normalised potential `Φ(r) = −∫_r^R t f` carries a constant
   of size `R^{n+2}` that cancels over the closed polygon; numerically a `(R/L)²` loss per recursion level. Any antiderivative
   works, and `Φ~ = ∫_0^r t f` (vanishing at 0) differs by a constant that drops out *exactly*
   (`C(∮ n_i y^β ds − β_i ∫ y^{β−e_i}) = 0`), so for polygons inside the support the compiled recursion uses `Φ~`
   (`np2d.compile_moment(..., inner=True)`; selected per support radius by `max_v|v−x| ≤ R`). Effect for `x` inside the element,
   p = 3: **float64 error 1e-15 for every `L_T/h` down to 1e-4** (plain: 1e-8 at 1e-2, 0.85 at 1e-4); float32 closed form 6e-7, flat.
2. **Gradient weights `g_α = ∇m_α + α_j m_{α−e_j}`**: two `O(L^{|α|+1})` terms cancelling to `O(L^{|α|+3})`. The same trick with the
   potential of `−W'/r`, `Ψ~ = −(W(r) − W(0))` (vanishes at 0): `g_j = −Σ_e n_j ∫ y^α Q ds + α_j ∫_T y^{α−e_j} Q dA`,
   `Q = πW − πW(0)` (both terms scale like `r²`; `np2d.grad_moment`). Measured: 1e-15 relative for all `|α| ≥ 1` down to `L_T/h = 1e-6`
   (plain: 1e-8 at 1e-3, garbage at 1e-6).
3. **Far elements** (`x` outside, `d ≫ L_T`): the re-expansion `Σ_α B_{αi} m_α` has terms `~ (d/L)^{|α|}` that cancel
   (error `~ ε (d/L)^{p+1}`; p = 3, `d = 0.3 h`: 4e-7 at `L/h = 0.01`, 5e-2 at 1e-3, 4e3 at 1e-4). The integrand `N_i W` is **smooth**
   on the element, so tensor-Gauss on the element (8×8 nodes, kernel evaluated from the exact `(R−r)^k` expansion, no cancellation) is
   spectrally accurate: **1e-15 for `|centroid − x| ≥ longest edge`** (checked at ratios 1…3, 6–14 nodes, p = 1 and 3; the singularity of `W`
   at `x` stays ≥ 0.33 L from the element). `np_fem.weights_hybrid` selects Gauss for far elements and the exact edge reduction otherwise
   (so exactness is kept wherever the edges matter: `x` inside / adjacent). Measured: float64 hybrid **1e-15 everywhere** (all `p`, `L/h` 1…1e-4,
   `d` up to 0.6 h); at the support rim (elements straddling `r = h`) the kernel is only `C^k`, Gauss is algebraic: absolute 3e-12 (weights there are ≲ 5e-7).
4. **float32:** with the three remedies the float32 error is at epsilon level (`3e-7` relative, flat in `L_T/h`; `p = 3` near elements of size `~h`: ≤ 8e-5 relative,
   i.e. ≤ 1e-6 absolute) instead of `10^3 … 10^{22}`. Elements whose features are below float32 resolution at their distance remain unrepresentable
   (input limit, not arithmetic).

Selected rows (p = 3, kernel w4; relative error of the nodal weights; full table in `python -m edgebound.fem_study`):

| x | L_T/h | plain f64 | plain f32 | **hybrid f64** | **hybrid f32** |
|---|---|---|---|---|---|
| inside | 1e-2 | 9e-16 | 3e-7 | 9e-16 | 3e-7 |
| inside | 1e-4 | 1e-15 | 2e-7 | 1e-15 | 2e-7 |
| d = 0.3 h | 1e-1 | 1e-11 | 3e-3 | 1e-15 | 2e-7 |
| d = 0.3 h | 1e-2 | 4e-7 | 90 | 2e-15 | 1e-7 |
| d = 0.3 h | 1e-3 | 5e-2 | 2e7 | 1e-15 | 3e-7 |
| d = 0.6 h | 1e-2 | 5e-4 | 7e4 | 1e-15 | 8e-8 |
| d = 0.6 h | 1e-4 | 9e5 | 7e14 | 7e-16 | 2e-7 |
| d = 0.95 h (rim) | 1e-2 | 3e3 | 3e12 | 2e-15 | 3e-6 |

(`plain f32` here is the stage-2 float32 closed form *with* the inner-potential remedy for x-inside rows; far rows are the plain edge route.)

## 7. Known failure modes

- Rim-straddling elements in the Gauss branch (kernel `C^k` only): absolute ~3e-12, negligible weights.
- Elements `≪` float32 resolution at their distance (see 6.4).
- Non-convex or non-triangular elements: not supported by the numpy FEM path (decompose).

## 8. Open questions

- Import the exact test cases of Winchenbach & Kolb 2025 (paper not available in this checkout) and compare accuracy/speed.
- Tolerances for the near-but-outside regime in float32 at `p = 3` (≈ 1e-4 relative worst case) if tighter bounds are needed: use float64
  profile evaluation inside the edge terms (cost: only the near elements).
- Variable density `ρA` (degree `2p`) and L2 projection of non-unisolvent samples: straightforward with the existing moments, not implemented.
