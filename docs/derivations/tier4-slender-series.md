# Tier 4: fibres / codimension-2 boundaries

**Status:** 2D: [V] (strip and disk mean-value series, exact references incl. the circle edge identity, crossover vs tier-2 polygons); 3D ball primitive and strand series: [D]/[P] (3D is deferred until all of 2D is done)
**Tier(s):** 4 · **Dimension:** 2D (strip and disk) and 3D (strand, ball)
**Depends on:** `edge-primitives.md`, `../notation.md`
**Implemented in:** `src/warpSPHBoundaries/edge/tier4.py` (2D: `disk_series`, `strip_series`, `strip_exact`, `arc_value`), `oracle.polar_disk_*`
**Verified by:** `tests/edge/test_tier4.py`

## 1. Statement

**Slender limit `a << h` (3D strand of radius `a`, centreline segments):**

$$\rho_B\approx\rho_0\pi a^2\sum_{\rm seg}\int_{\rm chord}\Big[W+\tfrac{a^2}{8}\Delta_\perp W+\tfrac{a^4}{192}\Delta_\perp^2W+\dots\Big]ds$$

(2D-disk mean-value series `Σ_k Δ^k f a^{2k} / (4^k k! (k+1)!)`). Leading term is the
2D edge-chord primitive with `z` the distance to the segment line. 2D strip
analogue: `Σ_k a^{2k} ∂_⊥^{2k} W / (2k+1)!`.

**3D ball (shell theorem, elementary for polynomial kernels):**
`∫_{S²} f(|x - c - ρω|) dω = (2π/(dρ)) ∫_{|d-ρ|}^{d+ρ} r f(r) dr`, clipped at `h`.
This is the same object as the sphere closed forms in `docs/derivation.md`.

**Straight segment, any `a`:** axial integral is closed form with the same `I_m`
primitives; the cross-section integral is an offset disk → 2D divergence identity →
a periodic integral (elliptic in closed form; trapezoid rule converges
exponentially; split at kinks where the support sphere crosses).

## 2. Assumptions and validity

`a << h` for the series (asymptotic: truncate at the kernel's C-order; Wendland C4:
`a²` or `a⁴`), small `h κ_c` for fibre bending. Wendland has no linear term so
`Δ_⊥ W` is regular on the axis.

## 3. Derivation

See `HANDOFF.md §11`. Curved centreline: expansion in `κ_c` (Weyl tube volume is
independent of `κ_c` for `a < 1/κ_c`, but the kernel weighting is not symmetric in
`φ`, giving a first-order correction `∝ κ_c a²`).

## 4. Special cases and limits

`a → 0`: line with weight `π a²`. `a → h`: must match tier 1 prism (continuity at
tier switches, `HANDOFF.md §10`).

## 5. Checks (2D rows are done; 3D rows stay open)

2D results (`tests/edge/test_tier4.py`, reference = circle edge identity / polar oracle / planar closed forms, 40 digits):

* **Circle edge identity (curved boundary, new):** `∫_disk W = 1[x∈disk] + ∮ (n·y)(M(r) − M(R))/r² dℓ` over the arcs inside the support, arcs split at the angles where the circle
  crosses a kernel radius, Gauss–Legendre per arc (smooth integrand). Equals the independent polar-coordinate disk oracle to **1e-42** (x outside and inside the disk, support crossing the
  circle, cubic knot radius crossing the circle). The divergence identity therefore holds on curved boundaries with the same structure as for edges (a *curved element* only changes the
  1D quadrature on the boundary: arcs ⇒ no closed form for the line integral, exponentially convergent Gauss).
* **Disk (2D "point"/fibre cross-section) mean-value series** `π a² Σ a^{2k} Δ^k W / (4^k k!(k+1)!)`: error of the K-th partial sum scales as `a^{2K+4}` (measured orders 4.0/6.0/8.0 for K = 0/1/2,
  w4, w6, cubic). Needs the kernel smooth on `[D−a, D+a]` (raises at a knot / the rim).
* **Strip series** `Σ 2a^{2k+1}/(2k+1)! ∫ ∂_z^{2k}W ds`: orders `a^{2K+3}` (3, 5 asymptotically); the exact strip is `λ_2(d−a) − λ_2(d+a)` = two edges of the polygon machinery (verified: long
  rectangle equals the planar closed-form difference to 1e-33). In 2D the strip series is thus only an *economy* (one chord per centreline segment, varying width), not a necessity.
* **Crossover vs tier 2** (w4, `D = 0.6 h`, absolute error of the value): the series needs 1 evaluation point + `K` derivatives, the polygon `N` edges.

  | a/h | exact | series K=0 | K=1 | K=2 | K=3 | 8-gon | 16-gon | 32-gon |
  |---|---|---|---|---|---|---|---|---|
  | 1/5 | 1.921e-02 | 6.2e-03 | 3.1e-04 | 4.4e-06 | 4.8e-07 | 2.4e-03 | 6.4e-04 | 1.6e-04 |
  | 1/10 | 3.648e-03 | 4.0e-04 | 4.7e-06 | 1.9e-08 | 4.7e-10 | 4.0e-04 | 1.0e-04 | 2.6e-05 |
  | 1/20 | 8.365e-04 | 2.5e-05 | 7.4e-08 | 7.5e-11 | 4.6e-13 | 8.6e-05 | 2.2e-05 | 5.5e-06 |

  K = 1 already beats a 32-gon for `a ≤ h/10` (error 5e-6 vs 2.6e-5), K = 2 by orders of magnitude; at `a = h/5` K ≥ 1 still wins but the series is asymptotic (K = 3 gains little).
  There is no gap in 2D between the disk series and the tier-2 polygon: the polygon is the fallback when `a` is not small or the kernel is non-smooth at `D`.

Table (original plan, 3D rows still open):

| check | how | status |
|---|---|---|
| ball primitive vs 3D quadrature and vs `docs/derivation.md` sphere (`R = a`) | mpmath | [P] |
| slender series vs semi-analytic straight cylinder vs tier-1 prism mesh over `a/h` | find crossover | [P] |
| strip (2D) series vs exact strip | mpmath | [P] |
| curved-centreline `κ_c` first-order term | Maple | [P] |

## 6. Known failure modes

Gap at thickness `~ h`; polyline joints (overlap inner, gap outer); asymptotic
series diverging at high order.

## 7. Open questions

Joint and end-cap treatment; sub-resolution force model (physical caveat in
`HANDOFF.md §11`).
