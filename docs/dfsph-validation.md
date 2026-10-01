# DFSPH on the scene boundary layer: validation against omniSPH (2D)

**Status:** [V] — solver `python/edgebound/dfsph2d.py`, omniSPH reference `dfsph_ref.py`, tables `python -m edgebound.dfsph_validation`, tests `tests/edge/test_dfsph.py` (4).
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

## 5. Not done / next

* the boundary force of omniSPH (`log.txt`) is not compared (our `wallForce` = `−Σ m a_b` is available per step);
* omniSPH's barycentric (`sim.barycentricPressure`) MLS pressure at the triangle vertices is not reproduced: it would need nodal data on a refined wall layer (`volumeMode='nodal'`);
* gradient renormalisation with the exact wall moments (removes the first-row residual of §2), larger resolutions, 3D;
* the rotating obstacle (the scene layer already carries pose and `BodyField.rigid`), then tier 3/4 around a moving disk.
