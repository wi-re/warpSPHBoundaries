# Tier 3 in 2D: closest-point curvature expansion of the value integral

**Status:** [V] (F₀, F₁, F₂ derived in Maple, evaluated in closed form with the edge machinery, verified against the exact disk by two independent routes; convex and concave)
**Tier(s):** 3 · **Dimension:** 2D (3D version: `PLAN.md` §8, §19; exact oracles in `docs/derivation.md`)
**Depends on:** `../notation.md`, `docs/derivation.md` (2D planar closed forms), `moments-recursion.md`, `tier4-slender-series.md` (circle edge identity / exact disk)
**Implemented in:** `src/warpSPHBoundaries/edge/tier3.py`; exact references `tier4.arc_value`, `oracle.polar_disk_*`
**Verified by:** `maple/14_tier3_2d.mpl`, `tests/edge/test_tier3.py`

## 1. Statement

For a smooth boundary with curvature `κ` (> 0 convex solid, notation.md) and distance `d` of the particle from the surface (`h = 1`, `α = hκ`):

$$\lambda(d,\kappa)=F_0(d)+\kappa F_1(d)+\kappa^2F_2(d)+O(\kappa^3),$$

with, over the half plane `y_2 ≥ d` (`y_1 = t` tangential, `y_2 = d + s`, `s` = depth into the solid), `Φ(σ) = W(√σ)`, `σ = |y|²`:

$$F_0=\int\Phi,\quad F_1=\int\big[\Phi' y_1^2(2d-y_2)-(y_2-d)\Phi\big],$$
$$F_2=\int\Big[\Phi'\big(-d(y_2-d)y_1^2-\tfrac{y_1^4}{12}\big)+\tfrac12\Phi''y_1^4(2d-y_2)^2-(y_2-d)\Phi'y_1^2(2d-y_2)\Big].$$

`F_0 = λ_2` (planar closed form). Every term is a half-plane **moment** `m_{ab}[f] = ∫_{y_2≥d} y_1^a y_2^b f(r)` of one of the exact polynomial profiles
`W`, `Φ' = W'/(2r)`, `Φ'' = (W'/r)'/(4r)` (powers down to `r⁻¹` for Wendland and `r⁻³` for the cubic spline), i.e. a single-edge instance of `moments-recursion.md`:
no new quadrature, no new special function. Gradient: `∇_xλ = (∂_dλ) n̂` (the closest-point curvature of a circle is constant), obtained from the same closed forms.

## 2. Assumptions and validity

- Reach `≥ h` on the solid side, single boundary sheet in the support, curvature smooth over one support (`κ` evaluated at the closest point).
- Asymptotic in `α = hκ` (error `O(α^{K+1})` for the K-th truncation); `R > h − d` so that the depth `s` of the support stays inside the disk.
- Concave boundary: `κ < 0` (the solid is the complement of a disk, `x` in the hole): the same series (checked below).

## 3. Derivation  (`maple/14_tier3_2d.mpl`, all checks PASS)

1. Tubular coordinates of a disk obstacle: `x' = c(t) − s n(t)`, `s` = depth into the solid. Polar area element `ρ' dρ' dθ` with `ρ' = R − s`, `θ = κ t` gives
   **`J = 1 − κ s`** (exact; with `s` measured towards the fluid it is `1 + κ s`).
2. Exact distance: `|y|² = (d+s)² + (1+κd)(1−κs)·2(1−cos κt)/κ²`  (Maple, exact identity), whose series is
   `σ_0 + κ t²(d−s) + κ²(−d s t² − t⁴/12) + κ³ t⁴(s−d)/12 + …`, `σ_0 = (d+s)² + t²`.
3. Taylor expansion of `(1−κs) Φ(σ)` to `κ²` (Maple, generic `Φ`) gives exactly the integrands of `F_1, F_2` above.
4. Moments: expand the polynomial in `y` and read off `m_{ab}` of the profiles `Φ, Φ', Φ''` (sympy), evaluate by the edge recursion (stage 1, with negative powers via `I_{-1}, I_{-3}`).

Sign / the Tube Maps `K` question: in 2D only `κ` enters and the **measured convergence orders (κ¹, κ², κ³ for the 0th/1st/2nd truncation, both signs of κ) confirm `J = 1 − κs`** with `κ > 0` for a
convex solid. The 3D analogue by the same construction is `J = (1−κ_1 s)(1−κ_2 s) = 1 − 2Hs + Ks²` with `s` towards the solid — `K` enters with `+` in this convention; a sign inconsistency in a source can only
arise from measuring `s` towards the fluid (`(1+κ_1 s)(1+κ_2 s)`). The 3D check itself is deferred with the rest of 3D.

## 4. Special cases and limits

`κ → 0`: planar (F₀ = λ_2 to 1e-42). `F_1 < 0` (a convex solid occupies less of the support). Finite at the wall: `F_0(0) = ½`, `F_1(0) ≈ −0.0463` (w4), and for w4 `F_2(0) ≈ 0` (`F_2 = O(d)`).
A disk of radius `R` is reproduced for `R > h − d`; otherwise "support engulfs the obstacle" (tier 4 / tier 2).

## 5. Checks

| check | how | tolerance | status |
|---|---|---|---|
| `J = 1 − κs`; exact `|y|²` identity; κ⁰…κ³ coefficients | Maple symbolic | exact | [V] |
| Taylor integrands of `F_1, F_2` for generic `Φ` | Maple symbolic | exact | [V] |
| `F_0` = planar closed form `λ_2` (4 kernels, `d = 0.01 … 0.95`) | edge machinery vs PLAN closed form | 1e-35 | [V] |
| `F_1, F_2` closed form vs direct polar quadrature of the Taylor integrands (w4, `d = 0.3`) | independent of the moment recursion and Maple | 1e-15 | [V] |
| truncation error ∝ κ¹ / κ² / κ³ against the EXACT disk (circle edge identity = polar oracle, 1e-42), 4 kernels, `d = 0.05, 0.3, 0.6`, **κ > 0 and κ < 0** | `test_truncation_error_orders_convex_and_concave` | orders within (o+0.65 … o+1.6) | [V] |
| gradient `∂_dλ` orders vs exact disk gradient (polar oracle), convex | `test_gradient_orders` | orders 1, 2; 3rd order ≥ 5× better | [V] |
| regular at `d → 0` | `d = 1e-3, 1e-5` | `F_k` stable | [V] |
| tier 3 vs tier 2 polygon of the same disk | below | — | [V] |

Measured (w4, `d = 0.3 h`, absolute error of the value; tier 2: regular polygon with the given edge length):

| κh | R/h | exact λ | planar F₀ | + κF₁ | + κ²F₂ | polygon edge h/2 | edge h/4 | edge h/8 |
|---|---|---|---|---|---|---|---|---|
| 1/2 | 2 | 0.09017 | 9.0e-03 | 9.9e-04 | 1.4e-04 | 7.3e-03 | 1.9e-03 | 4.6e-04 |
| 1/4 | 4 | 0.09444 | 4.7e-03 | 2.6e-04 | 1.8e-05 | 3.8e-03 | 9.5e-04 | 2.4e-04 |
| 1/8 | 8 | 0.09674 | 2.4e-03 | 6.8e-05 | 2.3e-06 | 1.9e-03 | 4.9e-04 | 1.2e-04 |
| 1/16 | 16 | 0.09794 | 1.2e-03 | 1.7e-05 | 3.0e-07 | 9.8e-04 | 2.5e-04 | 6.2e-05 |

The planar (2020-paper) model is as accurate as a polygon with edges `h/2`; the first-order term already matches edges `h/8`; the second-order expansion beats a polygon with `h/8` edges
by `> 10×` at `κh = 1/4` without any mesh or edge loop (it needs `κ`, `d` and 3 precomputed 1D functions `F_k(d)`).

## 6. Known failure modes

`α` not small (`κh ≳ 0.5` still improves with order but the series is asymptotic); thin solids and multiple sheets (tier 4 regime); curvature noise from an SDF; curvature
discontinuities inside the support. `d = 0` needs the limit form of the negative-power primitives (`d = 1e-5` is used; the closed form at exactly `d = 0` is not implemented).

## 7. Open questions

Curvature at the particle vs at the closest point (differs at `O(κ' d)`; for a disk both agree); gradient and first-moment expansions as 1D tables
`F_k'(d)`, `G_k(d)` (the gradient here is the `d`-derivative of the closed forms); tier-selection metric: the table above suggests `κh ≲ 0.25` for second order at ~1e-5.
