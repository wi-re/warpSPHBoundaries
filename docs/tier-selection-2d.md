# Tier selection in 2D: which model for which obstacle size (convex disk obstacle)

**Status:** [V] measured (`python -m edgebound.tier_select`, `tests/edge/test_tier_select.py`); exact reference = circle edge identity (`tier4.arc_value`, agrees with the polar oracle to 1e-42)

Setup: solid disk of radius `R`, particle at distance `d = 0.3 h` from the surface (centre at distance `R + d`), kernel w4, support `h`. Absolute error of the value `λ`
(`λ ≈ 0.01 … 0.1` here, so 1e-4 is ~0.1–1 %). All models see the same particle; tier 2 = regular polygon inscribed in the circle with the given edge length.

| R/h | exact λ | tier 4 series K=2 | tier 3 planar (2020 baseline) | tier 3 order 2 | tier 2 edge h/4 | tier 2 edge h/8 |
|---|---|---|---|---|---|---|
| 0.05 | 0.0077 | **1.2e-09** | n/a | n/a | 7.7e-04 | 7.7e-04 |
| 0.1 | 0.0224 | **2.4e-07** | n/a | n/a | 2.3e-03 | 2.3e-03 |
| 0.2 | 0.0465 | **2.5e-05** | n/a | n/a | 5.2e-03 | 3.4e-03 |
| 0.35 | 0.0627 | 1.2e-03 | n/a | n/a | 7.8e-03 | 2.0e-03 |
| 0.5 | 0.0708 | n/a | n/a | n/a | 5.8e-03 | 1.6e-03 |
| 1 | 0.0827 | n/a | 1.6e-02 | **9.7e-04** | 3.5e-03 | 8.8e-04 |
| 2 | 0.0902 | n/a | 9.0e-03 | **1.4e-04** | 1.9e-03 | 4.6e-04 |
| 4 | 0.0944 | n/a | 4.7e-03 | **1.8e-05** | 9.5e-04 | 2.4e-04 |
| 8 | 0.0967 | n/a | 2.4e-03 | **2.3e-06** | 4.9e-04 | 1.2e-04 |
| 16 | 0.0979 | n/a | 1.2e-03 | **3.0e-07** | 2.5e-04 | 6.2e-05 |

(`n/a`: tier 4 needs the kernel smooth on `[D−a, D+a]`; tier 3 needs `R > h − d` so that the support depth stays inside the disk.)

## Findings

* **Three regimes, with a gap.** (i) `R ≲ 0.2 h`: tier 4 (series, 1 point + derivatives): `1e-9 … 2.5e-5`, thousands of times better than any polygon with ≥ 8 edges.
  (ii) `R ≳ h`: tier 3 second order: `1e-3 … 3e-7`, comparable to a polygon with edges `h/8` at `R = h`, better by 3× (R = 2h) to 200× (R = 16h) beyond, and 16–4000× better than the planar (2020) model; first order already matches edges `h/8`.
  (iii) the **gap `0.35 h ≲ R ≲ 1 h`**: neither expansion applies (series: the disk reaches the rim; tier 3: the support engulfs the obstacle, `hκ ≳ 1`). Here only the exact machinery works: a polygon
  (error `~1e-3` for `h/8` edges, `O(ℓ²)`) — or an exact **curved element**: the circle edge identity (arcs with Gauss–Legendre, `tier4.arc_value`) is exact to 1e-42 with ~40 nodes per arc and removes the discretisation error entirely.
* **Switching without jumps.** The jump at a switch is bounded by the sum of the two models' errors (all errors above are against the truth), so a policy {tier 4 for `R ≤ 0.2h`, exact curved/polygon element for `0.2h < R < 2h`, tier 3 order 2 for `R ≥ 2h`} has jumps
  `≲ 2.5e-5 + 0` at the lower switch (arc element exact) and `≲ 1.4e-4` at the upper one, i.e. ≲ 0.15 % of `λ`, versus `3e-3` (3–5 %) if the gap were filled by an `h/4`-edge polygon. Matching asymptotics instead of switching: tier 3 order 0 → planar (exact by construction),
  tier 4 `a → h` → tier 1/2 disk (not a smooth limit: the series is asymptotic) — a blend over `0.2 h < R < 0.35 h` with weights `w(R)` would remove even the remaining jump; not implemented.
* **Selection metrics (2D, convex obstacle):** `a/h` (obstacle size / local thickness) ≤ 0.2 → tier 4; `κh ≤ 0.5` and `R > h − d` → tier 3 (second order ≤ 1e-3, ≤ 1e-4 for `κh ≤ 0.25`); otherwise exact elements. The curvature of a real surface must be measured over one support
  (see `tier3-curvature-2d.md` §6).
* Concave boundaries (cavity, `κ < 0`): tier 3 behaves identically (`tests/edge/test_tier3.py`); thin solids / multiple sheets remain tier 4/2 territory.

Not covered: non-circular curvature variation inside the support, several obstacles (double counting), the 3D versions of these thresholds (`H, K`; deferred with 3D).
