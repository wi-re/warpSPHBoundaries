# δ⁺-SPH on analytic walls: still-water validation against warpSPH + mDBC (2D)

**Status:** [V] tank, wedge (with one open item, §3). Solver `python/edgebound/deltasph2d.py`, scoring `python -m edgebound.deltasph_validation tank|wedge`, tests `tests/edge/test_deltasph.py` (7).
Plan and order of the test cases: `deltasph-plan.md` (tank → English wedge → dam break → sloshing). Porting notes for warpSPH: `deltasph-porting-notes.md`.
Reference: warpSPH `scripts/probe_englishWedge.py --dp 0.02 [--no-wedge] --tLimit 4` (scheme `deltaSPH`, isothermal EOS, Wendland C2, h/dp = 2, δ = 0.1, α = 0.01, fourtakas2019 DDT, Antuono pressure force,
symplectic Euler, mDBC + free-slip walls, hydrostatic density initialisation, no shifting). English et al. 2022 §4.1: tank 2.4 × 1.2 m, water 0.5 m, wedge 0.24 m high on the bed.

## 1. What is the same, what is different

Same: kernel, support 4 dx, mass ρ0 dx², isothermal EOS with c0 = 20 √(g H) = 44.29, density evolved by continuity from the hydrostatic profile, δ-diffusion `δ H c0/ξ` with ξ = 2.8214 (fluid–fluid only, exactly as warpSPH),
α-viscosity, Antuono pressure-force switch, Barecasco free-surface detector, symplectic Euler (k0 at tⁿ, half step, k1), the Sun-2017 time step (dt = 2.856e-4 for the tank in both codes; 14008 vs 14006 steps).

Different: **no boundary particles.** The wall enters through λ = ∫W, ∇λ, the hydrostatic wall pressure `P_b = P_i + ρ0 (g − a_w)·(x' − x_i)` (clamped ≥ 0) as a per-query field, the free-slip mirror
in continuity (`2 ρ u_n |μ∇λ|`, n = ∇λ/|∇λ|), and the wall as a continuum of particles in the free-surface detector (`Scene.inside`). The first fluid row sits dx/2 from every wall. Not yet: wall term of the α-viscosity (zero at rest),
no-penetration, shifting (not used in these cases).

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
| bulk RMSE | 0.0225 | **0.0090** |
| near wall / bed RMSE (max) | 0.0274 (0.033) | 0.0358 (0.076) |
| wedge faces RMSE (≤ 0.05) | 0.0259 | **0.0136** |
| apex RMSE, max (≤ 0.03) | 0.0226, 0.024 | **0.0078**, 0.027 |
| base corners RMSE, max (≤ 0.03) | 0.0287, 0.031 | **0.0161**, 0.032 |
| settled KE (tail mean) | **1.06e-6** | 3.99e-4 (peak 1.36e-3, decaying, 2.9e-4 at 4 s) |

All eight checks pass; the profile near the wedge is about twice as accurate as with mDBC. **Open item: the kinetic energy.** The settled KE is 400 × the reference's: a slow circulation (u ≈ ±0.08 m/s converging on the wedge,
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

## 5. Next

* wedge KE: a consistent packing (above); then the dp = 0.01 wedge.
* dam break (Marrone 3.1): needs the wall term of the α-viscosity (second moments `∫ y⊗y g(r)`), the δ⁺ particle-shifting kernel (`W⁵`), no-penetration (closest point), the Barecasco detector with moving fluid, and probes P1–P3.
* sloshing (SPHERIC TC10): the moving-wall terms (`a_w`, free-slip relative to the wall velocity) are implemented but untested for δ⁺.
