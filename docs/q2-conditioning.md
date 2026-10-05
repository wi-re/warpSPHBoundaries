# Q2 conditioning: the tensile term `W^5/5` (WORK-002 T2.4)

Status: [V] (numbers from `python scripts/studies/q2_conditioning.py`, log `docs/work/logs/LOG-002.md`).

*Reviewer correction (REVIEW-002):* the model's T_y values (−6.8e-8 / −4.9e-8) used the factor (c2/c25)⁵/π⁴, which is wrong (c2⁵ / (π⁴ c25) is right); the sign statement was and is correct. Corrected values: −0.14101158 (w2) and −0.19782376 (w4), each confirmed by an independent midpoint-grid integral of W⁴ ∂W/∂y over the half disk (−0.14101146, −0.19782346; `docs/work/refs/q2_review_probe.py`). The "vertex finding" below is localised to within ~0.01 H of a vertex (same probe), i.e. it does not matter for particles that sit ≥ dx/√2 from a corner.

## Derivation

1. The wall part of the delta+ tensile control is  T_i = ∫_solid W⁴(x, x′) ∇_i W(x, x′) dA′  (the solver's `shift()`, polar quadrature today).
2. W depends on x only through r = |x − x′|, so ∇_i W = W′(r) (x − x′)/r, and d/dr W⁵ = 5 W⁴ W′, i.e.  ∇_i W⁵ = 5 W⁴ ∇_i W.
3. The repo kernels are W = C (1 − r/H)^m p(r/H) with m = 4 (w2) / 6 (w4), so W⁵ ∝ (1 − r/H)^20 / (1 − r/H)^30: W⁵ and its first derivative vanish at r = H.
4. Hence  T = (1/5) ∫_solid ∇_i W⁵ dA′ = (1/5) ∇_i ∫_solid W⁵ dA′  — the integral has compact support in the disk of radius H around x, so ∇ and ∫ interchange and the boundary term at r = H is zero.
5. The right side is exactly the g(0,0) gradient channel of the kernel W⁵ — the channel the exact edge reduction (`warpbc.edge_channels` / `np2d.gradient`) computes.
6. So Q2 reduces to: how well does the compiled plan of W⁵ (degree 25 for w2, 40 for w4) evaluate that channel?

**Sign / units.** W decreases with r (W′ < 0), so ∇_i W = W′(r) (x − x′)/r points from the particle TOWARD the solid: T has the direction of ∇λ, i.e. it points INTO the wall, like G = μ∇λ in `shift`. Verification at the flat-floor point (0, 0.3), support 1, solid below y = 0 (the square [−2,2] × [−2,0] is exactly the half-plane inside the disk; mpmath reference at 150 + GUARD digits):

```
w2: g0_y = -1.5485900248e-01   T_y = (1/5) c2^5 / (pi^4 c25) * g0_y = -0.14101158   T_y < 0: True   (c2 = 7,  c25 = 37.8968)
w4: g0_y = -7.3201280751e-02   T_y = (1/5) c2^5 / (pi^4 c25) * g0_y = -0.19782376   T_y < 0: True   (c2 = 9,  c25 = 44.8625)
```

## Setup

Unit square, support h = 1, point set = the 6 probe points of `docs/work/refs/q2_probe.py` + 200 random points in [−0.5, 1.5]² (seed 5) + the 4 vertices and 4 edge midpoints (214 points). Routes: (A) Warp `warpbc.edge_channels` (the current monomial-basis plan — the solver's route), (B) `np2d.gradient` float64 plain, (C) `np2d.gradient` stable=(8,6), (D) `np2d.gradient` stable=(16,8); reference: `core.block_grad` at 150 + GUARD dps. All return ∇_x ∫_solid (π W^k) dA′. Kernels: w2p{k} = (shape_w2)^k (degree 5k), w4p{k} = (shape_w4)^k (degree 8k), each renormalised on its own (`c2_pi`); the physical T (h = 1) differs from the g0 of the registered W⁵ kernel only by the fixed factor (1/5) c2⁵ / (π⁴ c25), c2 = `c2_pi` of the family, c25 = `c2_pi` of `W⁵` (for support H: times H⁻⁸), so the RELATIVE errors below are the ones that matter.

## Results

Worst |err|/scale per (family, k, route), scale = max|ref| over the 214 points (mpmath reference 4.9 s; total run 6.0 s):

```
w2 k=1 degree=5  scale= 1.393e+00 | A  5.30e-15  B  7.02e-15  C  1.01e-12  D  2.39e-16
w2 k=2 degree=10 scale= 2.072e+00 | A  4.35e-13  B  3.40e-13  C  6.59e-12  D  2.14e-16
w2 k=3 degree=15 scale= 2.587e+00 | A  2.60e-11  B  1.81e-11  C  7.37e-12  D  3.43e-16
w2 k=4 degree=20 scale= 3.026e+00 | A  1.11e-09  B  1.18e-09  C  6.83e-07  D  5.14e-16
w2 k=5 degree=25 scale= 3.417e+00 | A  6.18e-08  B  4.68e-08  C  1.13e-05  D  5.85e-16
w4 k=1 degree=8  scale= 1.646e+00 | A  6.07e-14  B  6.31e-14  C  1.77e-12  D  2.70e-16
w4 k=2 degree=16 scale= 2.365e+00 | A  2.59e-11  B  2.84e-11  C  1.15e-07  D  3.76e-16
w4 k=3 degree=24 scale= 2.905e+00 | A  1.02e-08  B  9.41e-09  C  1.30e-05  D  4.59e-16
w4 k=4 degree=32 scale= 3.362e+00 | A  4.62e-06  B  4.65e-06  C  3.54e-05  D  5.88e-15
w4 k=5 degree=40 scale= 3.767e+00 | A  1.66e-03  B  2.42e-03  C  7.53e-05  D  2.69e-12
```

The route A/B worst points are interior (e.g. (0.544, 0.975) for A at w2 k=5, (0.983, 0.849) for A at w4 k=5); the route C/D
worst point is the VERTEX (0,0) at every k ≥ 2 — see the finding below. DevicePlan build 0.00–0.02 s per kernel, nE = 16, nV = 5
(all ten kernels).

**Reviewer reproduction** (probe numbers, 6 points; within a factor 10 = reproduced): route A reproduces all eight (w2: 5.30e-15 / 4.35e-13 / 2.60e-11 / 6.18e-08 vs 4.0e-15 / 1.5e-13 / 6.7e-12 / 1.8e-08, ratios 1.3–3.9; w4: 6.07e-14 / 2.59e-11 / 1.02e-08 / 1.66e-03 vs 2.5e-14 / 1.1e-11 / 2.8e-09 / 7.1e-04, ratios 2.3–2.4); w4 k=5 B 2.42e-03 vs 8.4e-04 (ratio 2.9). The stable routes C and D do NOT reproduce on the 214-point set (7.53e-05 / 2.69e-12 vs 3.2e-11 / 5e-17) — but on the reviewer's own 6 points they do (3.246e-11, 1.201e-16): the discrepancy is the vertex finding below, not a failure of the probe numbers.

**Finding (reported, not fixed):** the np2d stable routes lose accuracy at polygon vertices, growing with degree. Per point subset
at k = 5 (worst |err|/scale, B / C / D):

```
w4p5  6 probe (reviewer) 8.359e-04 / 3.246e-11 / 1.201e-16
w4p5  4 vertices         1.332e-05 / 1.505e-04 / 5.389e-12
w4p5  4 midpoints        3.958e-13 / 3.118e-07 / 5.895e-16
w2p5  6 probe (reviewer) 1.212e-08 / 1.039e-11 / 1.982e-16
w2p5  4 vertices         3.005e-09 / 2.254e-05 / 1.170e-15
w2p5  4 midpoints        6.563e-14 / 7.628e-09 / 3.899e-16
```

Both runs localise the worst stable-route error to the vertex (0,0) at comparable magnitude (full run: 2.835e-04 abs for C at
(0,0); probe: vertex-subset worst C 1.505e-04 relative). `np2d.py` is not in the WORK-002 allowed-files list, so this is reported
only.

## Classification (k = 5)

Fixed thresholds (the shifting tensile term is ~0.14 against a net first-row shift of ~0.03, HANDOFF A6; an error of 1e-5 of the term's scale is then < 0.05 % of the net): ≤ 1e-8 GOOD, ≤ 1e-5 ACCEPTABLE, > 1e-5 NOT USABLE.

```
w2: A warp ACCEPTABLE;  B np plain ACCEPTABLE;  C np stable(8,6) NOT USABLE;  D np stable(16,8) GOOD
w4: A warp NOT USABLE;  B np plain NOT USABLE;  C np stable(8,6) NOT USABLE;  D np stable(16,8) GOOD
```

For w2, A is 6.18e-08 — above the 1e-8 GOOD line by a factor ~6 (the reviewer's 6-point value 1.8e-08 would classify GOOD).

## Recommendation

- **w2:** route A (the Warp monomial plan) is usable as is at k = 5 — 6.2e-8 worst (ACCEPTABLE), within 3.4× of the reviewer's 6-point value; no change needed.
- **w4:** route A is not usable at k = 5 (1.7e-3); using the exact edge reduction for the w4 tensile term needs the stable basis in the Warp plan (`warpbc.py`, out of scope here) — the np2d stable route is a drop-in reference for the target accuracy (2.7e-12 on the full 214 set) but carries the vertex caveat above.

## Resolution (WORK-004)

WORK-004 T4.1 adds the stable Chebyshev-quadrature edge plan to `warpbc.py` (opt-in: `STABLE_KERNELS` / the
`stable=(nodes, panels)` argument of `edge_channels`; the monomial plan is the unchanged default). T4.2 routes the tensile
term through it for **both** Wendland families: `tensile.tensile_factor` / `tensile_vector_scene` now take `family` in
`("w2", "w4")` and always set `warpbc.STABLE_KERNELS[family + "p5"] = (16, 8)`, so the C4 W^5 (degree 40) is usable.

Measured this work (`tests/edge/test_warpbc_stable.py` (a), unit square, 214 points, vs the mpmath reference): `w4p5`
g(0,0) worst |err|/max|ref| = **9.43e-16** at the 206 generic points and **2.69e-12** over all 214 (the vertex (0,0) limit,
identical in np2d); `w2p5` = 7.80e-16 both.  The C4 flat-floor tensile value (T4.2 (b)) is `T_y(H = 1, (0, 0.3)) =
-0.1978237590`, `T_y(H = 0.5, (0, 0.15)) = -101.2857646` (T_x = 0, T_y < 0; the 2000^2 midpoint grid is 6.13e-6 off, its own
quadrature error).  This closes the "w4 route A not usable" finding above.
