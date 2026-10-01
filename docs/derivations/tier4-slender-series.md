# Tier 4: fibres / codimension-2 boundaries

**Status:** [D] for the series (mean-value series of the disk) and the 3D ball primitive; [P] for 2D disk/segment implementations and crossover
**Tier(s):** 4 · **Dimension:** 2D (strip and disk) and 3D (strand, ball)
**Depends on:** `edge-primitives.md`, `../notation.md`
**Implemented in:** not yet
**Verified by:** not yet

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

## 5. Checks

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
