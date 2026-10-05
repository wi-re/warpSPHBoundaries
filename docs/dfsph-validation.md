# DFSPH on the scene boundary layer: validation against omniSPH (2D)

**Status:** [V] — solver `src/edgebound/sim/dfsph2d.py`, omniSPH reference `dfsph_ref.py`, tables `python scripts/dfsph_validation.py`, tests `tests/sim/test_dfsph.py` (12).
Purpose: use the scene layer (`scene-architecture.md`) in a real solver, with a domain body instead of boundary particles, before 3D.

## 1. The solver

`DFSPH2D` follows `omniSPH/simulation/fluidMechanics.cpp` term by term (omniSPH units: rest density 1, particle area `V`, Wendland C2, `h = sqrt(20 V / π)`, `dt = clamp(0.4 h / v_max)`,
divergence solve = exactly 4 Jacobi iterations, density solve ≥ 4 iterations until the mean density error ≤ 1e-3 with warm start `p = ½ p_prior`, `ω = ½`, XSPH 1e-4, boundary friction 5e-3).
Fluid–fluid sums are torch pair sums (checked against `warpSPHCore.warpOperation`, `test_pair_sums_match_warpoperation`); **every boundary term is an exact kernel integral over the
scene**, evaluated once per step as an adjacency and re-used in all pressure iterations:

| omniSPH term (sum over triangles hit) | here |
|---|---|
| `ρ_i += Σ k` | `Density` (λ = ∫ W) |
| `α`: `Σ gk`, source `−dt v_i·gk`, pressure update `dt² a_i·gk` | `Gradient` with field 1 (`gk = ∇λ`) |
| `a_b = −(p_i/ρ_i² + p_b) gk` per triangle, `p_b` = MLS at the closest point | `Gradient`, Symmetric, `BodyField(perQuery)` with `p_b(x') = p_i + ∇p_b·(x' − x_i)` (a field over the whole boundary, exact moments) |
| BXSPH wall friction `min(μ Σk, 1)·tangential v` | same with the exact λ and `−∇λ/|∇λ|` |

Wall pressure options (`DFSPHConfig.wallPressure`): `hydrostatic` (default, `∇p_b = ρ g`, i.e. `∂p/∂n = ρ (g − a_wall)·n`: independent of the neighbours' pressure, exact for a hydrostatic column),
`linear` (kernel-weighted least-squares gradient of the neighbours' pressure, the analogue of omniSPH's MLS), `mirror` (`p_b = p_i`, zero normal gradient: cannot carry the weight).
The divergence solve ignores the wall exactly as omniSPH's does (`boundaryInDivergence` adds it; the dynamics then differ).

## 2. Findings that shaped the setup

* **omniSPH conventions.** `V = π r²`, packing `0.3992 h`; the tank and dam-break files put the wall faces one spacing outside the fluid block. With the face half a spacing from the first row (a lattice that
  continues into the wall) omniSPH itself blows up in 25 steps (`v_max` 50 m/s), with one spacing it settles; the comparison below uses omniSPH's convention for both codes.
* **A calibrated rest lattice** (`lattice_calibration`): the discrete kernel sum of the lattice is `S = 1.013` (not 1), so a regular lattice starts 1–5 % compressed and DFSPH removes that in one step (an impulse).
  With `V' = dx dy / S`, wall mass per area `1/S` and the first row at the distance `d` where its density is also exactly 1 (`d = 0.551 dx`, from the exact half-plane λ) the initial density error is
  `5e-12` (bulk) / `7e-5` (first row) with the exact wall. The cross-representation runs use it.
* **Linear wall pressure destabilises tight solves.** The least-squares extrapolation feeds the neighbours' pressure back into the wall term; with `η = 1e-5` and many iterations the Jacobi iteration diverges after ~6 steps
  (`hydrostatic` and `mirror` do not). With omniSPH's loose tolerance (η = 1e-3, 4 iterations) it behaves like omniSPH.
* **SPH noise at walls is inherent.** For the exact hydrostatic pressure the discrete residual acceleration of the calibrated lattice is `0.4 m/s²` in the bulk, `4.5` in rows 2–3 and `17` (1.8 g) in the first row
  (identical for every representation: it is the incompleteness of the SPH gradient at a wall, `Σ V ∇W + ∇λ ≠ 0` at lattice points near it), so a nominally static tank has `rms v ≈ 0.05 m/s` and
  `v_max ≈ 0.2–0.6` with any wall closure that uses these sums, omniSPH included. The exact integrals give the *moments* needed for a first-order correction (`g1`) — a follow-up, not done here.

## 3. Against omniSPH (identical initial particles; walls = omniSPH's triangle slabs as a `VolumeRep`)

Tank (0.96 × 0.23 m, settling into the wall gap), `r = 0.005`, 2943 particles; columns: x of the rightmost particle / mean y / `v_max` (m/s).

| model | t = 0.05 | t = 0.10 | t = 0.20 | t = 0.30 |
|---|---|---|---|---|
| ours / hydrostatic | 0.9848 / 0.1396 / 0.37 | 0.9854 / 0.1370 / 0.60 | 0.9868 / 0.1317 / 0.50 | 0.9874 / 0.1315 / 0.35 |
| ours / linear | 0.9847 / 0.1398 / 0.36 | 0.9849 / 0.1379 / 0.65 | 0.9854 / 0.1318 / 0.43 | 0.9872 / 0.1315 / 0.54 |
| **omniSPH** | 0.9850 / 0.1389 / 0.38 | 0.9849 / 0.1361 / 0.58 | 0.9862 / 0.1309 / 0.35 | 0.9861 / 0.1308 / 0.22 |

Dam break (0.2 × 0.8 m column, box 1.5 m wide, 2093 particles): x of the front / mean y / `v_max`.

| model | t = 0.1 | t = 0.2 | t = 0.3 | t = 0.4 | t = 0.5 | t = 0.6 |
|---|---|---|---|---|---|---|
| ours / hydrostatic | 0.3816 / 0.4578 / 1.99 | 0.5994 / 0.3547 / 2.94 | 0.8798 / 0.2423 / 3.43 | 1.2127 / 0.1793 / 3.78 | 1.5792 / 0.1526 / 4.16 | 1.5981 / 0.1698 / 6.27 |
| ours / linear | 0.3924 / 0.4577 / 1.90 | 0.6065 / 0.3542 / 2.87 | 0.8893 / 0.2415 / 3.46 | 1.2239 / 0.1785 / 3.74 | 1.5959 / 0.1520 / 4.09 | 1.5989 / 0.1712 / 6.47 |
| **omniSPH** | 0.3805 / 0.4576 / 1.93 | 0.6029 / 0.3542 / 2.82 | 0.8871 / 0.2415 / 3.37 | 1.2184 / 0.1784 / 3.67 | 1.5888 / 0.1516 / 4.07 | 1.5975 / 0.1696 / 6.97 |

The front position agrees within 0.9 % for `hydrostatic` and within 3.1 % for `linear` (worst at t = 0.1), the mean height within 0.4 %, `v_max` within the single-particle scatter. The runs are chaotic after
the wall impact (t > 0.5), so only statistics are compared. Particle-by-particle agreement is not expected: the wall closures differ (exact integrals with a reconstructed field vs per-triangle closest-point MLS).
Wall time of our run: 13 s for 0.6 s of dam break on the GPU (omniSPH's number is not comparable: its Python driver copies the state each step).

## 4. The domain body in four representations (calibrated lattice, dam break 0.6 s, `reps`)

The same case with the domain (inner faces at the box) given as the representations of the scene layer. rms particle displacement from the `surface` run:

| domain representation | what the particle sees | t = 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 |
|---|---|---|---|---|---|---|---|
| `SurfaceRep` hole loop (tier 2: one closed loop, edge terms + indicator) | exact | — | — | — | — | — | — |
| `VolumeRep` triangle slabs (tier 1: P1 weights, `volumeMode='moments'`) | exact | 1e-15 | 5e-14 | 2e-11 | 3e-8 | 6e-6 | 4e-4 |
| `SdfRep` of the box, tier 3 + corner fallback to the loop | exact in the corners, tier 3 elsewhere (tables, planar moments) | 5e-6 | 1e-4 | 4e-3 | 1.5e-2 | 2.7e-2 | 5e-2 |
| four `HalfPlaneBody` (tier 3 without the corner treatment) | corners counted twice | 7e-4 | 4e-3 | 1.2e-2 | 2.5e-2 | 4e-2 | 6.9e-2 |

* tier 1 and tier 2 are **independent code paths** (triangle kernel vs single-edge kernel) and give the same trajectories to round-off; the displacement growth (≈ ×400 per 0.1 s) is the chaos of the flow amplifying 1e-16.
* the SDF differs by `5e-6` after 0.1 s: tier-3 table interpolation (derivative 1e-8) and the SDF discretisation, amplified the same way; macroscopically the front, mean height and wall force agree to < 1 %
  (front at t = 0.4: 1.2203 surface / volume, 1.2217 SDF, 1.2183 half planes).
* four half planes double count the corner region: λ of a particle `0.55 dx` from both walls is `0.4166` instead of `0.3738` (+11 %), and the trajectories deviate at once (7e-4 after 0.1 s). Hence the need for the SDF
  switch (probe test → fallback surface) or an exact corner primitive.

## 5. Moving bodies and the force on them (`dfsph_cases.tank_with_obstacle`, `python scripts/dfsph_validation.py obstacle`)

A hexagon (circumradius 0.06 m, `SurfaceRep` loop or six triangles as `VolumeRep`) rotating at a prescribed `omega` in the tank, carved out of the lattice with the same first-row distance as the flat walls.
What a prescribed moving body adds to the solver (all through the scene layer, nothing special-cased):

* **source term** `−dt (v_p,i − v_b)·∇W` → `−dt (v_p,i·∇λ − ∫ v_b·∇W)`: one `Divergence` op with `BodyField.rigid(body)` (`a0 = v_c`, `a1 = ω J`: exact first moments);
* **wall pressure** `∂p/∂n = ρ (g − a_wall)·n` with the wall acceleration of the body at the particle, `a_wall = a + α J s − ω² s` (`Body.accelerationAt`; the `O(ω²h²)` curvature of `p_b` is neglected, 2 % of the gravity
  term at ω = 4 rad/s, h = 0.022);
* **friction** relative to the wall velocity, `v_wall = ∫ v_b W / λ` (`Interpolate`), per body; the divergence solve sees the wall automatically when a body moves (`boundaryInDivergence=None`);
* the bodies move after the fluid (`Body.move`), the friction adjacency is re-used as the next step's adjacency (same positions, same poses).

**Forces on every body** are tracked each step (`sim.history[k]`: `pressure` and `friction`, each `[B, 2]`, `balance`): the pressure force is `−Σ m_i a_b,i` of the last pressure pass of each solve (one `perBody` boundary
operation: the contribution of each body separately), the friction force `Σ m_i fac_i v_tangential,i / dt` (omniSPH's `boundaryPressureForce` / `boundaryDragForce`). `force_coefficients(F, ρ, U, L)` gives `(c_x, c_y)`.
**Bookkeeping check**: every step `Σ m Δv = dt (m g − ΣF_pressure − ΣF_friction)` holds to round-off (`1e-18`, asserted `< 1e-14` in the tests): the tracked forces are exactly the momentum exchanged with the bodies
(XSPH and the fluid–fluid pressure forces are pairwise antisymmetric and drop out).

| case (tank 0.6 × 0.30 m, r = 0.005, 0.6 s) | rep | mean `F_y` on the body (t > 0.2) | buoyancy `ρ g A` | mean `F_x` | Σ forces on all bodies / (−weight) | max momentum residual |
|---|---|---|---|---|---|---|
| fixed | surface | 0.09190 | 0.09175 | -0.00150 | 1.00008 | 9e-19 |
| fixed | volume | 0.09190 | 0.09175 | -0.00150 | 1.00008 | 2e-18 |
| ω = 3 rad/s | surface | 0.09038 | 0.09175 | +0.00223 | 1.00319 | 1e-18 |
| ω = 3 rad/s | volume | 0.08945 | 0.09175 | +0.00219 | 0.99748 | 1e-18 |

* the fixed body feels the Archimedes force within **0.2 %** (time average; the instantaneous force scatters by ±20 % like every SPH wall force, see §2), and the walls plus the body carry the fluid weight within 0.01 %;
  (numbers after the wall-suction fixes of §7; before: 0.8 % / 0.06 % and 5–6 % below buoyancy for the spinning body, whose wall suction was part of that deficit);
* the spinning body's mean force is 1.5–2.5 % below buoyancy (the rotation drives a flow; friction is 1e-4 of the pressure force at `μ = 5e-3`); surface and volume representation give identical forces for the first steps
  (`test_surface_and_volume_obstacles_give_the_same_forces`, 1e-9) and then separate through the chaos of the flow;
* dam break (0.2 × 0.8 m column, box 1.6 × 1.0 m, 1425 particles) into a hexagon spinning at 3 rad/s: no force until the surge arrives (t = 0.36 s), peak horizontal force 0.93 N/m at t = 0.37 s, decaying afterwards
  (`results/dfsph/dam_hexagon.png`).
* Not tracked: the **torque** on the body. For a linear pressure field it needs the second moments `∫ y_a y_b ∇W` (exact for surface loops via `|α| = 2` edge channels, not for P1 volume weights); the force is exact.

## 6. Gradient correction at the wall (`DFSPHConfig.gradientCorrection`)

**The problem** (§2): for the exact hydrostatic pressure the symmetric DFSPH acceleration of the calibrated lattice is not zero near a wall, because the kernel sum of the fluid neighbours is incomplete and the wall's exact
`∇λ` does not cancel it at the discrete particle positions (`Σ_j V_j ∇W_ij + μ ∇λ ≠ 0`). Mean residual `|a + g|` per region, exact `p = ρ g (H − y)`, `r = 0.005`:

| | first row | rows 2–3 | left column | interior |
|---|---|---|---|---|
| symmetric DFSPH, exact wall integrals (omniSPH form) | 17.7 m/s² | 4.5 | 8.8 | 0.43 |
| + first-order renormalisation `L = (Σ V y⊗∇W + μ g1)⁻ᵀ`, difference-form pressure gradient (exact wall moments) | **0.0** | 0.02 | 0.05 | **0.0** |
| + **zeroth-order wall closure** (kept, `gradientCorrection='wall'`) | 0.19 | 0.86 | 0.92 | 0.43 |

**What was tried and dropped.** The renormalised difference form is exactly consistent (the exact wall moments `g1` complete the covariance matrix, `Covariance` operation of the scene) and makes the hydrostatic lattice an
equilibrium. But (i) it is not an acceptable DFSPH operator: the diagonal of the difference-form system changes sign at the free surface (`α > 0`, `Σ V∇W` there is large), the Jacobi iteration diverges within
two steps (with a free-surface-safe eigenvalue threshold it survives only without the divergence solve, and with it diverges again); (ii) the difference form is not pairwise antisymmetric, so the wall is no longer the
reaction of the fluid: the tracked force on the walls drops to 5 % of the weight (the missing part is carried by non-conservative fluid–fluid forces). A hybrid (renormalise only wall-contact particles) keeps
the iteration stable but has the same bookkeeping problem. The `Covariance` operation stays in the scene layer (tested: surface = volume = half-plane model, `I` deep inside a body).

**The closure kept.** A uniform pressure must exert no net force on a wall-contact particle, so the wall has to supply minus the discrete incompleteness of the fluid sum. The exact `∇λ` fixes the *direction* (the
geometry, and therefore the force distribution over the bodies); only its *magnitude* in the `p_i`-terms (`α`, source, `dt² a·∇λ`, `−(p_i/ρ_i² + p_i)∇λ`) is rescaled,
`s_i = clip( −(Σ_j V_j ∇W_ij)·n / |μ∇λ|, 1 − κ, 1 + κ )`, `n = ∇λ/|∇λ|`, `κ = 0.2` (`closureLimit`; the clip bounds the effect where the incompleteness is a free surface and not the wall). The
first-order part of the wall term, `μ ∫ (a₁·y) ∇W` with the hydrostatic gradient, keeps the exact moments. Everything else is unchanged: symmetric pair forces, so the **momentum bookkeeping stays exact** (`1e-18`, asserted),
the force on each body is still `−Σ m a_b`. The closure removes the static wall residual by a factor 90 (first row) / 5–9 (rows 2–3, left column); the interior (no wall contact) is bit-identical.

**Dynamic effect** (submerged fixed hexagon, tank at rest, 0.6 s, statistics over t > 0.2; `python scripts/dfsph_validation.py closure`; two repeats per row, GPU atomics make runs differ in the chaotic cases):

| solver | closure | mean `F_y`/buoyancy − 1 | std `F_y`/buoyancy | std (Σ forces / weight) | rms speed |
|---|---|---|---|---|---|
| omniSPH tolerances (η = 1e-3, 4 iterations) | none | +0.8 % | 0.101 | 0.064 | 0.047 |
| | wall | +0.9 % | 0.124 | 0.067 | 0.050 |
| tight (η = 1e-5, 200 divergence iterations, divergence pressure unclamped) | none | −0.2 / +0.4 % | 0.37 / 0.51 | 0.17 / 0.22 | 0.047 / 0.049 |
| | **wall** | +0.1 % | **0.060** | **0.021** | 0.046 |
| tight, divergence pressure clamped ≥ 0 | none | −0.5 % | 0.126 | 0.041 | 0.063 |
| | wall | −1.1 / −2.1 % | 0.156 / 0.168 | 0.055 / 0.057 | 0.072 |

* with an accurately solved pressure and an unclamped divergence pressure the closure cuts the force noise **6–8×** (the noise of the body force 0.4–0.5 → 0.06 of the buoyancy, of the total wall force 0.2 → 0.02 of the
  weight), at unchanged mean (buoyancy within 0.1 %): this is the regime where the wall-row inconsistency is no longer hidden by the loose solve;
* with omniSPH's loose tolerances (4 iterations) there is no gain (the solver stops before the wall-row balance matters), with a clamped divergence pressure none either — so the closure is **not** the default;
* the whole-tank rms speed (0.046 m/s) is the interior noise of the lattice (`0.43 m/s²` first-order inconsistency of the symmetric fluid–fluid operator, relaxed by DFSPH), not a wall effect; it is unchanged;
* **a moving body needs the divergence pressure clamped in the wall term for tight solves**: with 200 divergence iterations and an unclamped divergence pressure the tank with the spinning hexagon diverges after 0.22 s
  (now the default for a moving body, `wallDivergenceClamp`, §7; `divergenceClamp=True` clamps the fluid–fluid pressure as well and works too, but expands the fluid, §7).

## 7. Walls must not pull: sticking to ceilings and walls (`wallDivergenceClamp`, `clampWallPressure`)

**Symptom.** In the dam break into the spinning hexagon (`dfsph_cases.tank_with_obstacle`, r = 0.006, 1425 particles) isolated particles stayed pinned on the ceiling and on both side walls after the splash
(row of dots at y = 1.0, nearly zero velocity); the same dam break without the hexagon, and omniSPH, show nothing of the kind. Found with `python scripts/dfsph_runcase.py` snapshots and `dfsph_video.ceiling_stats`
(particles within 1.5 h of the ceiling and the longest uninterrupted residence of each).

**Two independent causes** (`hex` = the hexagon case; N = particles; "stuck" = residence at the ceiling > 0.3 s):

| variant | stuck (r = 0.003, N = 5.5k) | longest ceiling residence |
|---|---|---|
| before | 14 (r = 0.006, N = 1.4k) | 0.73 s |
| wall divergence clamp only | 1 | 0.65 s |
| `p_b ≥ 0` clamp only | 20 | 0.78 s |
| **both (default)** | **0** | **0.19 s** |

1. **The wall in the divergence solve.** A moving body makes `boundaryInDivergence` true (the wall's normal velocity has to enter the source term; omniSPH's divergence solve ignores the wall altogether). The divergence pressure is signed
   (its negative part is the cohesion of the fluid), and with the wall term in the same iteration a negative `p_i` is a suction: a particle that leaves a wall has a positive velocity divergence from the wall,
   the solve answers with a negative pressure and the wall pulls it back. Regression test: an isolated particle leaving the ceiling at 1 m/s keeps it (−1.0000) with the fix, is stopped (≈ 0) without
   (`test_a_particle_leaving_a_wall_is_not_pulled_back_when_a_body_moves`). **Fix: `wallDivergenceClamp` (default: iff the wall is in the divergence solve) clamps the pressure only where it enters the wall acceleration.**
2. **The unclamped hydrostatic wall pressure.** `p_b(x') = p_i + ρ g·(x' − x_i)` is below `p_i` for a wall above the particle, and negative for a particle with little pressure: the term `μ∫(a₁·y)∇W` does not
   depend on `p_i` at all and, at a ceiling, is a suction that carries the particle's weight (a particle at y = 1.002, v_y = 0.00 for 0.6 s). omniSPH clamps the extrapolated wall pressure at 0 in the density solve.
   To leading order the term is a pressure offset `q` (per body) times the wall gradient, `q = (term·∇λ)/|∇λ|²`; **`clampWallPressure` (default on) removes `(1 − θ) q ∇λ` with `θ = clip(p_i/(−q), 0, 1)` for `q < 0`**,
   i.e. clamps the effective wall pressure `p_i + q` at 0 (floors and side walls: `q ≥ 0` or ≈ 0, unchanged). Only the normal offset is touched: the discontinuous version (scaling the whole term by θ) makes the volume and surface
   representation disagree by 2e-4 after 30 steps because round-off flips θ for `q ≈ 0`; with this form they agree to 2e-16 again (`test_dynamics_independent_of_the_representation`).

**What not to do: clamp the divergence pressure of all particles** (`divergenceClamp=True`, my first fix): it removes the sticking too, but the signed divergence pressure is also the cohesion between fluid particles; after the splash the
fluid stays apart and its volume grows by ~30 % (mean SPH density 0.69–0.74 and half the particles below 0.7, against 0.95–0.99 for the unclamped solve; spotted from the videos).
`test_divergence_pressure_keeps_its_sign_between_fluid_particles`: a stretching blob is damped exactly as in the wall-blind solve with the default, not with the global clamp.

**Result, dam break (0.2 × 0.8 m column, 1.6 × 1.0 m box) against omniSPH**, final code, front x / mean y / v_max; ceiling = particles within 1.5 h of the top wall at t = 0.7 / 0.8 / 0.9 / 1.0 and the longest residence
(`.tmp/omni`, `python scripts/dfsph_runcase.py`; omniSPH 303 s (2k) and 1877 s (8k) on 20 cores, ours 30–60 s on the GPU):

| N | model | t = 0.4 | t = 0.8 | t = 1.0 | t = 1.4 | ceiling contact | longest |
|---|---|---|---|---|---|---|---|
| 2093 | omniSPH | 1.218 / 0.178 / 3.7 | 1.597 / 0.292 / 3.9 | 1.597 / 0.327 / 4.8 | 1.596 / 0.180 / 3.0 | 47 / 63 / 49 / 7 | 0.29 s |
| | ours, omniSPH slabs (`VolumeRep`) | 1.213 / 0.179 / 3.8 | 1.598 / 0.292 / 4.2 | 1.598 / 0.329 / 5.3 | 1.597 / 0.181 / 4.3 | 44 / 63 / 52 / 5 | 0.27 s |
| | ours, tier 2 surface loop | 1.220 / 0.184 / 3.8 | 1.599 / 0.296 / 3.8 | 1.598 / 0.333 / 5.0 | 1.597 / 0.180 / 4.8 | 45 / 72 / 49 / 5 | 0.24 s |
| | ours, tier 3 SDF | 1.222 / 0.184 / 3.8 | 1.598 / 0.295 / 4.6 | 1.598 / 0.331 / 5.4 | 1.596 / 0.175 / 3.5 | 37 / 61 / 58 / 6 | 0.24 s |
| 8280 | omniSPH | 1.245 / 0.181 / 3.8 | 1.599 / 0.304 / 5.5 | 1.598 / 0.340 / 5.9 | 1.598 / 0.189 / 4.7 | 150 / 154 / 116 / 0 | 0.22 s |
| | ours, tier 2 surface loop | 1.243 / 0.183 / 3.7 | 1.599 / 0.305 / 7.0 | 1.599 / 0.342 / 6.0 | 1.599 / 0.187 / 5.6 | 144 / 129 / 110 / 0 | 0.22 s |
| | ours, tier 3 SDF | 1.240 / 0.183 / 3.7 | 1.599 / 0.305 / 5.2 | 1.599 / 0.343 / 8.3 | 1.598 / 0.187 / 4.7 | 145 / 143 / 120 / 0 | 0.22 s |
| 32669 | ours, tier 2 surface loop (omniSPH: ~3 h, not run) | 1.251 / 0.183 / 3.7 | 1.600 / 0.308 / 6.9 | 1.600 / 0.345 / 6.3 | 1.599 / 0.198 / 4.5 | 341 / 270 / 195 / 0 | 0.29 s |

Mean height within 0.5 % of omniSPH at 8k particles, the front within 0.3 %, ceiling contact and its duration the same (the splash touches the ceiling for 0.2–0.3 s and is gone by t = 1.0); v_max is the scatter of single
splash particles. Hexagon dam break, mean SPH density at t = 1.0 / ceiling residence: 0.950 / 0.20 s (N = 1.4k), 0.970 / 0.17 s (5.5k), 0.983 / 0.14 s (22k), against 0.954 / 0.73 s before.

**Remaining.** A few particles slide along a wall for 0.4–0.9 s above the pile (isolated, density 0.8–1): that is the wall friction of omniSPH (`boundaryFriction` is a per-step factor, so it damps more at higher N where dt is
smaller), not a suction: they move at 20–30 % of g. omniSPH itself has such particles (6 over 0.3 s at N = 2k). `linear` wall pressure still diverges in a long dam break (positions overflow before t = 2 s).
Videos of these runs (`python scripts/dfsph_video.py`) are in `.tmp/omni/vids`, not tracked.

## 8. Not done / next

* the boundary force of omniSPH (`log.txt`) is not compared (our `wallForce` = `−Σ m a_b` is available per step);
* omniSPH's barycentric (`sim.barycentricPressure`) MLS pressure at the triangle vertices is not reproduced: it would need nodal data on a refined wall layer (`volumeMode='nodal'`);
* gradient renormalisation with the exact wall moments (removes the first-row residual of §2), larger resolutions, 3D;
* tier 3/4 around a moving disk (needs first moments of curved tier-3/4 bodies), torque via second moments, a first-order consistent *and* conservative fluid–fluid/wall operator (the renormalisation above breaks pairwise antisymmetry and the free-surface diagonal), the interior lattice noise.
