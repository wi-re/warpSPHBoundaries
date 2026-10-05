# δ⁺-SPH on analytic walls: still-water validation against warpSPH + mDBC (2D)

**Status:** [V] tank, wedge (one open item, §3), Marrone 3.1 dam break (§5), SPHERIC sloshing (§6). Solver `src/edgebound/sim/deltasph2d.py`, scoring `python scripts/deltasph_validation.py tank|wedge`, tests `tests/sim/test_deltasph.py` (10).
Plan and order of the test cases: `deltasph-plan.md` (tank → English wedge → dam break → sloshing). Porting notes for warpSPH: `deltasph-porting-notes.md`.
Reference: warpSPH `scripts/probe_englishWedge.py --dp 0.02 [--no-wedge] --tLimit 4` (scheme `deltaSPH`, isothermal EOS, Wendland C2, h/dp = 2, δ = 0.1, α = 0.01, fourtakas2019 DDT, Antuono pressure force,
symplectic Euler, mDBC + free-slip walls, hydrostatic density initialisation, no shifting). English et al. 2022 §4.1: tank 2.4 × 1.2 m, water 0.5 m, wedge 0.24 m high on the bed.

## 1. What is the same, what is different

Same: kernel, support 4 dx, mass ρ0 dx², isothermal EOS with c0 = 20 √(g H) = 44.29, density evolved by continuity from the hydrostatic profile, δ-diffusion `δ H c0/ξ` with ξ = 2.8214 (fluid–fluid only, exactly as warpSPH),
α-viscosity, Antuono pressure-force switch, Barecasco free-surface detector, symplectic Euler (k0 at tⁿ, half step, k1), the Sun-2017 time step (dt = 2.856e-4 for the tank in both codes; 14008 vs 14006 steps).

Different: **no boundary particles.** The wall enters through λ = ∫W, ∇λ, the hydrostatic wall pressure `P_b = P_i + ρ0 (g − a_w)·(x' − x_i)` (clamped ≥ 0) as a per-query field, the free-slip mirror
in continuity (`2 ρ u_n |μ∇λ|`, n = ∇λ/|∇λ|), and the wall as a continuum of particles in the free-surface detector (`Scene.inside`). The first fluid row sits dx/2 from every wall. The wall term of the α-viscosity, the no-penetration impulse and the shifting (§5) are not used in the tank / wedge runs (zero at rest; shifting off as in the reference probe).

## 2. Flat tank (the hydrostatic baseline)

| | warpSPH + mDBC (dp = 0.02) | analytic walls, tier 2 surface loop (dp = 0.02) | tier 1 slabs | tier 3 SDF | tier 2 (dp = 0.01) |
|---|---|---|---|---|---|
| profile RMSE, bulk (of ρ0 g H) | 0.0225 | **0.0019** | 0.0019 | 0.0019 | 0.0014 |
| profile RMSE, near wall / bed | 0.0278 (max 0.034) | **0.0022** (max 0.013) | 0.0022 (0.013) | 0.0022 (0.013) | 0.0013 (0.006) |
| settled KE (tail mean) | 1.1e-6 | **1.3e-7** | 1.3e-7 | 1.3e-7 | 9.5e-8 |
| KE trend 2nd half | decaying | decaying | decaying | decaying | decaying |
| ρ range | [1.0001, 1.0025] | [1.0001, 1.0025] | same | same | [1.0000, 1.0025] |

All six checks of the probe pass in every column. The profile error is ten times smaller than with mDBC because the wall's pressure comes from the exact hydrostatic extrapolation instead of a cloned ghost value; the three
representations give the same numbers (surface = volume to round-off, SDF within its tier-3 tolerance), and the walls carry the fluid weight to 0.2 % (`test_hydrostatic_tank_stays_at_rest_and_the_walls_carry_the_weight`).
All columns on the final code (after the pair-list fix of §4, which changed the tank results by < 5 %). Wall time 5 min (dp = 0.02) / 13 min (dp = 0.01) on the GPU, warpSPH 3.8 min at dp = 0.02.

## 3. English wedge (smooth sloped faces, sharp apex, acute base corners)

Wedge = `SurfaceRep` triangle with exact corners (apex 0.24 m above the bed, half base 0.2773 m, 98° apex; geometry extracted from warpSPH's `equilateralBottom` SDF). Lattice points closer than dp/2 to the wedge are removed.

| check (dp = 0.02, 4 s) | warpSPH + mDBC | analytic walls |
|---|---|---|
| bulk RMSE | 0.0225 | **0.0091** |
| near wall / bed RMSE (max) | 0.0274 (0.033) | 0.0357 (0.084) |
| wedge faces RMSE (≤ 0.05) | 0.0259 | **0.0136** |
| apex RMSE, max (≤ 0.03) | 0.0226, 0.024 | **0.0061**, 0.018 |
| base corners RMSE, max (≤ 0.03) | 0.0287, 0.031 | **0.0212**, 0.037 |
| settled KE (tail mean) | **1.06e-6** | 3.8e-4 (peak 1.2e-3, decaying) |

All eight checks pass (numbers from the final code, after the dam-break additions; the earlier run gave apex 0.0078, corners 0.0161); the profile near the wedge is better than with mDBC. **Open item: the kinetic energy.** The settled KE is 400 × the reference's: a slow circulation (u ≈ ±0.08 m/s converging on the wedge,
symmetric about the apex, the second sloshing mode of the pool) is excited in the first second and decays with α = 0.01 over several seconds.

What was found (all runs in `.tmp/delta`, reproducer `ramp_build.py`):

* **Not a corner or apex problem.** A single 30° planar ramp (767 particles) reproduces the growth (KE 1.6e-3 at 1 s).
* **The solver terms are right on rectilinear walls and corners**: the same wedge with the wall replaced by the lattice staircase (`english_wedge(staircase=True)`: steps, re-entrant corners, the acute toe) stays at KE ≈ 1e-6, below the mDBC reference.
* **Cause: first-order inconsistency of the layout at a smooth slope.** The first-moment matrix of the pair set (fluid pairs + exact wall covariance, the scene's `Covariance` operation) deviates from the identity by |M − I| = 0.23 (max 0.75)
  in the first layer along the ramp, against 0.014 on a flat wall at dx/2 and 0.006 in the bulk. A regular lattice cut by a sloped face leaves the first layer at irregular distances (0.4–1.8 dx); the force error ρ g (M − I) ≈ 2 m/s²
  has no potential, so even the damped layout (`DeltaSPH2D.settle`, v ← v e^{−γ dt}) is not an equilibrium: its particles keep drifting at F/γ and the KE grows again as soon as the damping is released (9.5e-4).
* Not the cause: wall-pressure clamp, viscosity, DDT (removing the wall continuity term makes the run diverge, as it should).
* Tried: (i) a first-order completion of the wall term `(I − M)ᵀ a1` for every wall-contact particle — much worse (KE 3.6e-2): it also "completes" the free-surface side of contact-line particles; (ii) body-fitted packing along
  `S_i = Σ V∇W + μ∇λ = 0` (`DeltaSPH2D.pack`): reduces the face-layer residual 4.8 → 1.4 and halves the ramp KE (1.1e-3 → 4.2e-4 at 0.6 s) but converges to a floor, and a narrow window creates its own clumping.
* **Conclusion so far:** a smooth wall needs a layout consistent with it, and a regular lattice carved by the wall is not. The unresolved piece is a particle packing that reaches the flat-wall level of |M − I| (relaxation with the
  full first-moment condition, all particles below the free surface). Until then the wedge is usable (profile accuracy better than mDBC) but rings.

## 4. A bug found on the way

`neighbor_pairs` (shared with `dfsph2d.py`) built its cell list with cell = support exactly. With support 4 dx the lattice puts particles on cell borders, where the insertion and the query round differently, and **~2 % of the reverse pairs
went missing**: the pairwise antisymmetric forces no longer conserved momentum (the fluid–fluid pressure force summed to 3.6 % of the weight in the wedge). Cell = 1.01 × support fixes it; the pair list now equals the brute-force list
(`test_pair_list_is_complete_for_a_lattice_whose_spacing_divides_the_support`, fails with the old cell size). DFSPH uses non-commensurate lattices (spacing 0.399 h) and is unaffected (its 12 tests pass).

## 5. Marrone 3.1 dam break (the third case)

Reference: warpSPH `probe_deltaSPHMarrone.py --nx 67 --c0Ratio 40` (H/dx = 40.2, scheme `sun2017DeltaSPH`: δ⁺ with Sun-2017 shifting and Sun-2019 surface treatment, `noPenShiftMode='impulse'`, time-centred continuity,
fourtakas2019 DDT, Antuono, free-slip mDBC walls, c0 = 97.04, 3240 fluid particles, 147 s). Ours: `marrone_dambreak(nx=67)`, tier-2 surface loop for the closed tank (ceiling 0.985 m above the bed), the same lattice up to the 0.13 % x-spacing,
same probes (first-order MLS of the fluid pressure at the impact wall and 1 dx in, 7-point disc average, P* = P/ρ0 g H). 19533 steps (warpSPH 19551), 1222 s in torch (warpSPH's Warp/CUDA-graph step is 8× faster).

New terms for this case: wall term of the α-viscosity (free-slip mirror, `(α c0 H/ξ)(2/ρ) u_n M₂ n`, `M₂ = ∫W'(r)dr ∫ŷ⊗ŷ 1[solid]dφ` by 24 × 96 polar quadrature of the solid, 1 % of a fine half-plane integral, tested),
time-centred continuity, the dilated surface mask for the Antuono switch, δ⁺ shifting (wall part: `G` exact + the tensile-control `∫W⁴∇W dA` by the same quadrature), no-penetration impulse (the mDBC law reduced to an
analytic wall: `v_n ← v_n(1 − f)`, `f = 3 − 4 clip(½ + d/dp, ¼, 1)` for a closing particle with `d < dp/4`; `Scene.signed_distance`), the probes.

| quantity | warpSPH + mDBC | analytic, no shifting / noPen (A) | **analytic, shifting + noPen (B)** |
|---|---|---|---|
| P1 first P* > 0.05 | t* = 2.48 | 2.47 | **2.49** |
| P1 mean, t* ∈ [3.2, 4.8] | 0.353 | 0.574 | **0.319** |
| P1 mean, t* ∈ [5.2, 6.1] | 0.478 | 0.656 | **0.456** |
| P1 maximum (t*) | 1.08 (6.37) | 2.40 (6.12) | **1.09 (6.48)** |
| P1 at t* = 4 / 5 / 5.5 / 6.5 | 0.352 / 0.454 / 0.457 / 0.648 | 0.652 / 0.759 / 0.691 / 0.770 | **0.348 / 0.452 / 0.458 / 0.652** |
| P2 first > 0.05 / mean [5.2, 6.1] / max | 4.31 / 0.211 / 1.28 | 3.39 / 0.267 / 3.12 | 4.50 / 0.168 / 1.40 |
| P3 (ceiling) first > 0.05 / max (t*) | 3.25 / 15.1 (3.29) | 3.15 / 1042 (4.26) | 3.17 / 23.6 (3.66) |
| kinetic energy at t* = 2 / 3 / 4 / 5 / 6 / 7 | 0.78 / 1.02 / 0.93 / 0.74 / 0.75 / 0.58 | 0.75 / 1.00 / 0.94 / 0.80 / 0.78 / 0.59 | 0.75 / 1.01 / 0.96 / 0.77 / 0.73 / 0.60 |
| density range, max \|v\| | [0.983, 1.028], 9.6 | [0.577, 1.554], 37.6 | [0.972, 1.025], 6.8 |

* **B reproduces the reference**: the P1 curve overlaps it (arrival, the slow rise, the peak and the t* = 6–7.5 oscillation), the kinetic energy is within 3 % at all times, the density stays within 3 %, the ceiling spike (P3) has the right
  time and order. The weaker agreement is P2 (a thin run-up sheet on the wall: arrival 0.2 t* late, plateau 20 % low, a later and higher final peak) and the ceiling probe's magnitude (23.6 vs 15.1): both are sheet / splash quantities
  where single particles decide and the flow is chaotic after the ceiling impact.
* **A shows what the two corrections are for.** Without shifting the near-wall layer is disordered and P1 reads 1.6× high (the probe code documents the same bias for warpSPH without PST); at the two ceiling impacts (t* ≈ 4.2 and 4.8)
  particles are flung at 34–37 m/s with the density down to 0.58 and 1.55 and the ceiling probe reads 1000 P*. warpSPH itself, run with shifting and no-penetration off, develops a 116 m/s particle at the first ceiling impact and never
  recovers (stopped at t = 1.34 s, its step collapsing); our run recovers after each spike and completes. With the ceiling-sticking lessons of `dfsph-validation.md` §7 the analytic wall carries `P_b ≥ 0` and never pulls.
* Deviations that remain (docs/deltasph-porting-notes.md): the free-slip mirror uses the particle's own velocity (the reference mirrors the Shepard velocity at the ghost); the normals and λ of the shifting surface treatment come from the
  fluid pairs with the renormalisation matrix of fluid + wall; the wall counts in the free-surface detector, the viscous term and the tensile control by polar sampling of the solid; the x-spacing of the initial lattice is 0.13 % larger.

## 6. SPHERIC test case 10 (sloshing, lateral water)

Reference: warpSPH `examples/sloshingTank/run_sloshingTank.py --scheme wcsph --tLimit 7` (nx = 200, dx = 4.5 mm, Wendland C4, isothermal EOS c0 = 20, α = 0.02, fourtakas2019, symplectic Euler with time-centred continuity,
no-penetration impulse, constant dt = 1e-4, **Michel-2022 shifting**: warpSPH's case default; its Sun variant is not exposed). 70001 steps in 2050 s, not diverged, max|v| 7.7, ρ ∈ [0.82, 1.33].
Ours: `sloshing_tank(nx=200)`, tier-2 surface loop for the 0.9 × 0.508 m tank, 4200 particles, **Sun-2017 shifting** + no-penetration impulse, same dt. The tank is not moved: it rolls in the tank-fixed frame by rotating gravity,
`g(t) = 9.81 (−sin θ, −cos θ)` from the measured roll table, updated after every step exactly as warpSPH's `postStep`, so all walls are static and no moving-wall term is involved. 70001 steps in 5439 s, max|v| 5.8, ρ ∈ [0.83, 1.20].

| smoothed (10 ms) Sensor-1 peak, kPa @ s | measured | warpSPH + mDBC | analytic, Gaussian probe |
|---|---|---|---|
| impact 1 (2.4 s) | 1.80 @ 2.39 (raw 3.7) | 4.87 @ 2.34 | 4.29 @ 2.36 |
| impact 2 (4.1 s) | 2.32 @ 4.07 (raw 3.9) | 5.16 @ 4.00 | 3.83 @ 4.00 |
| impact 3 (5.7 s) | 1.17 @ 5.70 (raw 2.7) | 5.30 @ 5.56 | 7.47 @ 5.59 |

* **The flow agrees with warpSPH through all 7 s**: the kinetic energy curves overlap (within ~10–30 % at the minima after the first impacts, same phase), the three impacts arrive at the same times (both 0.03–0.15 s early against the measurement:
  warpSPH −0.053 s, ours −0.047 s cross-correlation lag; ours vs warpSPH +0.024 s, correlation 0.84), the density stays within a few percent bar the impacts.
* **Impact magnitudes are of the same order as warpSPH and 1.5–4 × the measured smoothed peaks**; the measured raw peaks (3.7, 3.9, 2.7 kPa) are inside the same scatter band as in warpSPH's own plan (2.2–13 kPa). The third impact is higher than warpSPH's (7.5 vs 5.3 kPa) and the probe
  shows a negative excursion of about −1 kPa after it. These are single-impact, thin-sheet quantities; no claim of better or worse than the mDBC reference is made.
* **The sensor model is the weak part.** Our Gaussian probe is the reference's own `sensorPressureProbe` (Tait pressure Shepard-averaged over fluid particles within 2 cm) and is NaN when fewer than three particles are near (43 % of the record here, 47 % in warpSPH's);
  its full-length `sensorPressure` (nearest boundary particle's density) has no counterpart with analytic walls. The **wall MLS probe** (first-order MLS of the fluid pressure at the sensor point, as the Marrone probes) is erratic here (8.6, 15.5 and 66 kPa
  spikes at the impacts): at a point at the free-surface level with few, one-sided neighbours the linear fit extrapolates wildly. A usable wall pressure for analytic walls should come from the wall model itself (the hydrostatic closure `P_b`, or the pressure force on the wall per length).
* Not compared: the shifting scheme (Sun vs Michel), nx = 100 / 400, the no-shift variant, the experimental repeatability band, the dilated-mask effect.

## Wall Laplacian viscosity (WORK-005)

`cfg.viscosityExact` (default off) replaces the pairwise polar-quadrature wall term of the α-viscosity with the exact wall Laplacian term (`viscosity.lap_lambda_scene`, `Δλ = ∫_solid ∇²W dA′` = 2λ[L] − tr Cov[L] of the registered L = W′/r; derivation in `derivations/laplacian-wall.md`).  Wall and bulk share the Laplacian coefficient ν_eff = α c0 H/(8ξ) (derived from the moment identity ∫ r W′ dA = −2, not calibrated; the bulk pairwise term additionally carries 2∇(∇·v), the wall term does not).  The two wall terms are **different operators** near the wall — the Laplacian form damps the wall-normal velocity 3–12× less in the first particle rows (flat-wall ratio 8|A|/B, `derivations/laplacian-wall.md` §6) — so the switch is a change of the discretisation (user decision, HANDOFF Part A Q1), not a quadrature replacement, and the numbers below are the do-no-harm physics gate, not an exactness claim.

`check --physics --cases tank,dambreak --cfg viscosityExact=true` (2026-10-04, `local-model`, full log `docs/work/logs/LOG-005.md` T5.3(1a)):

| line | value | gate | limit |
|---|---|---|---|
| tank rmseBulk | 1.49929753837e-3 | PASS | ≤ 1.873494419e-3 (1.25 × baseline) |
| tank rmseNear | 1.81340794898e-3 | PASS | ≤ 2.267181356e-3 |
| tank keLast | 7.28437955168e-7 | PASS | ≤ 3.503023691e-6 (5 × baseline) |
| dam break KE vs `dambreak_B_nx67` (max rel, 668 samples) | 3.9892e-4 | PASS | ≤ 0.05 |
| dam break P1 arrival | 2.4740 t\* (reference 2.49, \|diff\| 0.0160) | PASS | ≤ 0.05 |
| dam break ke_tstar 1 / 2 / 2.5 | 0.35736468689 / 0.754254694988 / 0.92280834676 | — | — |
| dam break maxVelocityMax | 6.59592351289 (default 6.68104559177) | — | — |
| dam break minDensityMin | 0.999429377347 | PASS | ≥ 0.97 |
| dam break maxDensityMax | 1.00603521646 | PASS | ≤ 1.03 |
| **PHYSICS GATE** | **PASS** | | |

All four exact switches on (`coverExact,coneExact,tensileExact,viscosityExact`; T5.3(1b)): the tank agrees with the single-switch run to within 1 in the last displayed digit (rmseBulk 1.49929753836e-3, rmseNear 1.81340794898e-3, keLast 7.28437955168e-7, same 3502 steps); dam break KE 3.5344e-4, arrival 2.4740 t\*, maxVelocityMax 6.598, densities 0.999428 / 1.006031, ke_tstar 0.357364 / 0.754228 / 0.922767; PHYSICS GATE: PASS.

Sloshing (SPHERIC 10, nx = 200, C4, truncated at T = 1.5 s, `viscosityExact` only; T5.3(2)): 15001 steps, wall 1388 s on the shared GPU (the reviewer's context run: 1917 s with two jobs).  KE max-relative vs `slosh_B_nx200` **2.8231e-4** (1500 samples, gate 5 %; reviewer 2.820e-4), maxVelocity max **0.5919** (default 0.5921), densities **[0.99928, 1.00489]** (default [0.99929, 1.00487]).  The stored `slosh_B_nx200` KE series is the default run, so the number measures the effect of the switch directly: **0.03 % over 1.5 s**, far inside the band.

**Effect.**  The switch changes the operator near the wall, and the trajectories follow at the bit level (the bit-level lines FAIL by design), but only far inside every gate band: the dam-break P1 arrival moves from 2.48970315022 t\* (baseline) to 2.47397042889 t\* (≈ 0.0157 t\* earlier; \|arrival − 2.49\| = 0.0160, gate ≤ 0.05); the dam-break KE vs `dambreak_B_nx67` is 3.9892e-4 relative (single switch) / 3.5344e-4 (all four) (gate ≤ 0.05); the tank gate metrics move by ≈ 1e-7 relative (rmseBulk 1.49929753837e-3 vs baseline 1.49929879553e-3, gate ≤ 1.25 × baseline; the all-four run agrees with the single-switch tank to within 1 in the last displayed digit).  The sloshing KE moves by 0.03 % over 1.5 s.  The default stays the pairwise form (`cfg.viscosityExact` off); whether to keep the pairwise form or switch the default is a user decision after these numbers, not a regression verdict. *(Superseded by WORK-006, below: the user decided; the exact Laplacian is now the default and the pairwise form is kept as `wallViscosityForm = "pairwise"`.)*

## Exact wall operations by default (WORK-006)

The four exact wall operations are now the **only** wall path: `cfg.coverExact`, `coneExact`, `tensileExact`, `viscosityExact` and the polar branches of the surface detector and of the delta+ shift are gone (T6.1). The wall viscosity form is chosen by `cfg.wallViscosityForm = "laplacian"` (default, exact wall Laplacian, Wendland C2/C4) or `"pairwise"` (the warpSPH free-slip-mirror form, polar quadrature, kept). `DeltaSPH2D` therefore requires `SurfaceRep` walls: the polar branches were the only path that stepped `VolumeRep`/`ImplicitRep` walls, and `domain="volume"` now raises `NotImplementedError` (scene-layer representation independence lives in `test_scene.py` / `test_dfsph.py`; the solver-level guard is in `test_deltasph.py`).

**(1) The new default reproduces the WORK-005 all-switches run (T6.1(8)).** `check --physics --cases tank,dambreak` with the new defaults (`.tmp/w006_t61_equiv.log`): tank rmseBulk 0.00149929753837, rmseNear 0.00181340794898, keLast 7.2843795517e-07, rhoMin 1.00008342046, rhoMax 1.00246989449, 3502 steps; dam break KE vs `dambreak_B_nx67` 3.5344e-4 (668 samples), P1 arrival 2.47397042889 t\* (|diff| vs 2.49 = 0.0160), ke_tstar 1 / 2 / 2.5 = 0.357363651811 / 0.754228393303 / 0.922766940745, maxVelocityMax 6.59792039874, minDensityMin 0.999428350557, maxDensityMax 1.00603117793, 6683 steps; **PHYSICS GATE: PASS**. Every line agrees with the WORK-005 T5.3(1b) all-four-switches numbers to ≤ 1e-6 relative (most identical to the last printed digit) — the primary evidence that the refactor changed nothing.

**(2) The pairwise alternative still works (T6.3(1)).** `check --physics --cases tank,dambreak --cfg wallViscosityForm=pairwise` (`.tmp/w006_t63_pairwise.log`) — with exact cover/cone/tensile this is the WORK-004 configuration: tank rmseBulk 0.00149879553508, rmseNear 0.00181374508459, keLast 7.00604738264e-07, rhoMin 1.00008277812, rhoMax 1.00246999588, 3502 steps (every gate line PASS); dam break KE vs B 2.4098e-5 (668 samples, gate ≤ 0.05), P1 arrival **2.4897 t\*** (reference 2.49, |diff| 0.0003 — it equals the old baseline's 2.48970315022 to the full printed precision), ke_tstar 1 / 2 / 2.5 = 0.357284798075 / 0.754059943541 / 0.922531956854, maxVelocityMax 6.67866530633, minDensityMin 0.999436862861 (gate ≥ 0.97), maxDensityMax 1.00628666806 (gate ≤ 1.03), 6683 steps (drift 0); **PHYSICS GATE: PASS**.

**(3) The baseline re-recorded for the new default (T6.3(2)).** `record --cases tank,dambreak` (spread of the two record runs: every tol/|value| ≤ 1.4e-6, `steps` spread 0; `.tmp/w006_t63_record.log`):

| line | new baseline (exact default) | old baseline (polar default) |
|---|---|---|
| tank rmseBulk | 0.001499297538 | 0.00149879553509 |
| tank rmseNear | 0.001813407949 | 0.00181374508459 |
| tank keLast | 7.284379552e-07 | 7.00604738263e-07 |
| tank rhoMin / rhoMax | 1.00008342 / 1.002469894 | 1.00008277812 / 1.00246999588 |
| tank steps | 3502 | 3502 |
| dam break ke_tstar 1 / 2 / 2.5 | 0.3573636518 / 0.7542283933 / 0.9227669407 | 0.357285986624 / 0.754074340911 / 0.922537582805 |
| dam break P1 arrival | 2.473970429 t\* | 2.48970315022 t\* |
| dam break maxVelocityMax | 6.597920399 | 6.68104559177 |
| dam break minDensityMin / maxDensityMax | 0.9994283506 / 1.006031178 | 0.999437856696 / 1.00630752117 |
| dam break steps | 6683 | 6683 |

The move is the expected operator change, not noise: the arrival is 0.0157 t\* earlier (the WORK-005 all-switches value), the dam-break ke_tstar lines move by ≈ 2e-4 relative, the tank metrics by ≈ 3–5e-7 absolute, and the step counts are identical. `check --cases tank,dambreak` against the new baseline: **OVERALL: PASS** (bit-level, every line PASS); `check --physics --cases tank,dambreak`: **PHYSICS GATE: PASS** (tank rmseBulk ≤ 1.25 × baseline, rmseNear ≤ 1.25 ×, keLast ≤ 5 ×, rhoMin ≥ 0.99, rhoMax ≤ 1.01; dam break KE ≤ 0.05, |arrival − 2.49| ≤ 0.05, minDensity ≥ 0.97, maxDensity ≤ 1.03, steps drift 0).

**(4) Sloshing with the new defaults (T6.3(3)).** SPHERIC 10, nx = 200, C4, T = 1.5 s (`.tmp/slosh_default.py`): 15001 steps, wall 1085 s; KE max-relative vs `slosh_B_nx200` **2.8102e-5** (1500 samples, gate 5 %), maxVelocity max **0.5920**, densities **[0.99928, 1.00490]**. The stored series is the OLD default, so 2.8102e-5 (0.0028 % over 1.5 s) is the effect of the whole WORK-006 default change — 10× below the WORK-005 `viscosityExact`-only 2.8231e-4 (0.5919, [0.99928, 1.00489]) and far below the 5e-3 finding threshold; maxVelocity and the densities agree with both the old default (0.5921, [0.99929, 1.00487]) and the WORK-005 run to the last printed digit.

## No-slip wall viscosity (WORK-007)

`cfg.wallViscosityForm = "noslip"` (the **default stays `"laplacian"`**) is a third wall-viscosity form that damps the **all-components** relative velocity (no-slip), instead of the wall-normal component only (free-slip, as in `"laplacian"` and `"pairwise"`). It is the Chiron et al. 2019 Eq. 91–92 one-sided finite-difference wall flux, specialised to the solver's units: per body `b`,

```
acc_wall = -2 nu_eff (v_i - v_w) |G_b| / (rho_i d_b),      nu_eff = alpha c0 H/(8 xi) = fac/8,      v_w = b.velocityAt(x_i),      d_b = max(signed distance to the wall, 0.25 dx)
```

`|G_b|` is the existing wall gradient (`G = wallMass ∇λ`; for a flat wall `|G_b| = wallMass ∫_chord W ds`, exact to 8.6e-16), so no new Warp or scene code is needed. Derivation, the flat-wall model, the constant-ghost rejection, the checks and the open questions: `derivations/noslip-wall.md`. The free-slip forms are unchanged; nothing is re-tuned.

**Couette / Poiseuille through the solver** (T7.1 (c); `hydrostatic_tank(dp=0.04)`, `a_flux = rhs("noslip") − rhs(viscosity=False)`, first component, mean over the 15 columns `|x| < 0.3` in rows `s = (k+0.5) dp`, k = 0..3, ν_eff = 0.00314; the free-slip forms are the negative controls — on a tangential field their wall term is exactly 0): the no-slip term recovers the continuum target in the rows where a first-order flux term can, and the free-slip forms fail by design.

| row (s/dp) | 0.5 | 1.5 | 2.5 | 3.5 |
|---|---|---|---|---|
| Couette `u = s`, bulk only (target 0) | 0.06014 | 0.01405 | 0.00084 | 0.00004 |
| Couette, **no-slip** a_flux | +0.00815 | −0.00855 | −0.00280 | +0.00001 |
| Couette, `laplacian` / `pairwise` a (row 0) | 0.06014 (both) | — | — | — |
| Poiseuille `u = s(0.36−s)`, bulk only (target −2ν_eff = −0.00628) | 0.01625 | −0.00147 | −0.00577 | −0.00600 |
| Poiseuille, **no-slip** a_flux | −0.00143 | −0.00825 | −0.00672 | −0.00601 |

The no-slip Couette a_flux is ≤ 0.142 · |a_bulk(row 0)| in every row (tolerance 0.2); the no-slip Poiseuille is 7 % / 4 % from the target on rows 2 / 3 (tolerance 15 %) and much closer to the target than the bulk on rows 0 / 1 (0.22 / 0.42 of the bulk's distance to it). The free-slip forms give |a(row 0)| = |a_bulk(row 0)| = 0.06014 (> 0.5 · 0.06014), i.e. they damp no tangential velocity, as expected. (Matches the reviewer's solver probe to the digit, `docs/work/refs/review5_noslip_solver_probe.py`.)

**T7.2 gates and sloshing (information, not a pass/fail verdict).** The reference series (`dambreak_B_nx67`, `slosh_B_nx200`) are free-slip runs, so a no-slip run is expected to deviate; the gate is the *default's* physics (WORK-002 limits). A `PHYSICS GATE: FAIL` here is a finding reported with the numbers, not a blocker, and nothing is adjusted.

`check --physics --cases tank,dambreak --cfg wallViscosityForm=noslip` (`.tmp/w007_t72_regress_noslip.log`):

| line | value | gate | limit |
|---|---|---|---|
| tank rmseBulk / rmseNear / keLast | 0.0015052 / 0.0018605 / 5.5667e-7 | PASS | ≤ 1.25× / 1.25× / 5× baseline |
| tank rhoMin / rhoMax | 1.0000818 / 1.0024704 | PASS | ≥ 0.99 / ≤ 1.01 |
| dam break KE vs `dambreak_B_nx67` (max rel, 668 samples) | 0.3165191 | **FAIL (FINDING)** | ≤ 0.05 |
| dam break P1 arrival | nan t\* (P\* never > 0.05 in T = 0.65 s; reference 2.49) | **FAIL (FINDING)** | ≤ 0.05 |
| dam break ke_tstar 1 / 2 / 2.5 | 0.3206926 / 0.5688047 / 0.6396314 (default 0.3573637 / 0.7542284 / 0.9227669) | — | — |
| dam break maxVelocityMax | 3.7982989 (default 6.5979204, ~43 % lower) | — | — |
| dam break minDensityMin / maxDensityMax | 0.9990133 / 1.0014192 | PASS | ≥ 0.97 / ≤ 1.03 |
| dam break steps drift | 0 (6683 = 6683) | PASS | ≤ 0.05 |
| **PHYSICS GATE** | **FAIL** (the two dambreak KE/arrival lines) | | |

The tank is ≈ the default's (the hydrostatic tank is near-rest; the no-slip term damps only the tiny spurious wall velocities, so keLast is even lower than the default's 7.28e-7); the dam break changes a lot — the no-slip wall removes tangential momentum, so the KE at t\* = 1/2/2.5 is 10–31 % below the default's and the peak velocity is ~43 % below — while the density and step gates stay healthy. The bit-level lines FAIL by design (the noslip term is not exactly zero at the tiny spurious velocities).

Sloshing (SPHERIC 10, nx = 200, C4, T = 1.5 s, `wallViscosityForm = "noslip"`; `.tmp/slosh_noslip.py`, 963 s wall, 15001 steps): KE max-relative vs `slosh_B_nx200` **1.4646e-01** (14.6 %, 1500 samples, gate 5 % — a FINDING, expected larger than the default's because the wall now dissipates tangentially), maxVelocity max **0.5683** (default 0.5920), densities **[0.99931, 1.00483]**. The wall of this case rolls (rotating gravity), so it exercises `b.velocityAt`.

**Stability** (T7.2(3), `.tmp/stability_noslip.py`: the no-slip explicit damping rate `k = 2 ν_eff |G_b|/(ρ d_eff)`, `d_eff = max(d, 0.25 dx)`, for the near particles after 300 steps, default config otherwise): dam break max k = **16.5835 s⁻¹**, dt = 9.727e-05 s, **k·dt = 0.0016**; sloshing max k = **20.4154 s⁻¹**, dt = 1.000e-04 s, **k·dt = 0.0020**. Both `k·dt ≪ 1` (the reviewer's estimate k ~ 5 s⁻¹ at d = dx/2 is exceeded only because the nearest particles sit at d ~ 0.25 dx, where 1/d is larger) — the term is stable at the solver's acoustic time step.

**Reading of the numbers (REVIEW-007).** The dam-break damping is the no-slip wall at the scheme's *artificial* viscosity `ν_eff = α c0 H/(8ξ)` (≈ 3000 × water's, diffusion length ≈ 1 dp after 0.5 s; `docs/work/refs/review7_noslip_probe.py`), not a physical wall friction; and `nan` for the P1 arrival means the arrival is later than the T = 0.65 s window (≈ 2.63 t*, default arrival 2.474 t*). Neither the 31.7 % KE difference nor the 43 % lower peak velocity says anything about real boundary layers.

The default stays `"laplacian"`; `"noslip"` is an opt-in third form. Making it the default, partial slip, per-body slip, a per-element `∫W ds/d_n`, and the `1/γ` renormalisation are out of scope (WORK-007).

## Speed after WORK-008

WORK-008 (phase 2a, 2026-10-05) removes work without changing the result: the cover vector and the tensile term now build their wall adjacency restricted to the gradient channels (3, 4) — 4 of the 21 kernel terms, the indicator pseudo-pairs skipped — the monomial `DevicePlan` arrays are cached per `(kernel, device, channels)` (the Chebyshev plans already were), and the `no_penetration` wall adjacency is reused by the next step's first `rhs` (positions, body poses, supports and kinds unchanged; `lam` and `G` do not depend on the densities or the gravity, so `A` is always recomputed). A `sceneOperation` guard rejects a pruned adjacency for anything other than the `Naive` `Gradient` of a constant scalar field. No default changed, nothing re-tuned.

**Physics (the WORK-002 limits, `check --physics --cases tank,dambreak`, `.tmp/w008_check.log`, 2026-10-05).** The default run matches the recorded baseline to the last digit: every bit-level line PASS, the largest margin over all lines is 0.011 (tank rmseBulk: value 0.00149929753837 vs baseline 0.00149929753835). **PHYSICS GATE: PASS.**

| line | value | margin | gate |
|---|---|---|---|
| tank rmseBulk / rmseNear / keLast | 0.00149929753837 / 0.00181340794898 / 7.28437955168e-07 | 0.011 / 0.004 / 0.000 | PASS (≤ 1.25× / 1.25× / 5× baseline) |
| tank rhoMin / rhoMax / steps | 1.00008342046 / 1.00246989449 / 3502 | 0.000 / 0.000 / 0.000 | PASS (≥ 0.99 / ≤ 1.01) |
| dam break KE vs `dambreak_B_nx67` (max rel, 668 samples) | 3.5344e-04 | — | PASS (≤ 0.05) |
| dam break P1 arrival | 2.4740 t\* (reference 2.49) | 0.0160 | PASS (≤ 0.05) |
| dam break ke_tstar 1 / 2 / 2.5 | 0.357363651811 / 0.754228393303 / 0.922766940751 | 0.000 / 0.001 / 0.005 | PASS |
| dam break maxVelocityMax / minDensityMin / maxDensityMax / steps | 6.59792039874 / 0.999428350557 / 1.00603117793 / 6683 | 0.001 / 0.000 / 0.000 / 0.000 | PASS (≥ 0.97 / ≤ 1.03) |
| **PHYSICS GATE** | **PASS** | | |

The 0.011 in rmseBulk is last-digit GPU run-to-run spread (the reviewer's run of the same code was ≤ 0.007, WORK-008.md §1), not a physical change; the stop threshold is a margin > 1.

**Speed (same profiler as `deltasph-profile.md`, back to back with its "before" section):**

| case | before (ms/step) | after (ms/step) | change |
|---|---:|---:|---:|
| dam break `marrone_dambreak(nx=67)` | 58.079 | **36.479** | **−37.2 %** |
| sloshing `sloshing_tank(nx=200)` | 72.043 | **46.490** | **−35.5 %** |

`buildAdjacency` is 10 → 9 calls/step (the `no_penetration` adjacency is reused by the next step's first `rhs`) and `sceneOperation` is 12 → 10 (the pruned cover / tensile adjacencies skip the indicator pseudo-pairs). Remaining cost split (dam break / sloshing, `% of step`): `buildAdjacency` is still the largest single cost at 21.076 / 27.857 ms (57.8 % / 59.9 %; at 9 calls, ≈ 0.56× the per-build cost of the full-channel builds), then `_detect_surface` 5.100 / 5.260 ms (22.2 % / 17.7 %), then the fluid pair sums 9.197 / 12.853 ms (25.2 % / 27.7 % — the same absolute work, a larger share because the step is shorter).

**Sloshing (SPHERIC 10, T = 1.5 s, default config, `.tmp/slosh_default.py` unchanged from WORK-006):** 15001 steps, wall 896 s, KE max-relative vs `slosh_B_nx200` **2.8100e-05** (1500 samples, gate 5 %; WORK-006: 2.8102e-05), maxVelocity **0.5920**, densities **[0.99928, 1.00490]** — unchanged from WORK-006 (the 896 s wall vs 726 s is GPU load on the shared machine, not a gate).

Out of scope (phase 2b, if pursued): fusing the kernels of one `rhs` call into one pass over the shared edge geometry; Verlet-style adjacency reuse across steps; CUDA-graph capture.

## 7. Next

* sloshing (SPHERIC TC10): the rolling tank is a prescribed rotating body (`Body.angularVelocity`, `accelerationAt`, wall velocity in the free-slip mirror, the viscous wall term and the no-penetration law relative to the moving wall: the static-wall
  assumptions in `no_penetration` and `rhs` must be lifted); sensor pressure on the left wall.
* wedge KE: a consistent packing (§3); higher resolution dam break (nx = 134) and the 3D question.
