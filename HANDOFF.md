# Handoff: exact SPH boundary integrals (edge reductions → scene layer → δ⁺-SPH on analytic walls → port to warpSPH)

**Last updated 2026-10-03.** Two parts:

* **Part A (this top part): where the project stands and the work plan from here** — written to start the next phase cold: remove the quadrature paths, optimise the scene path, refactor towards warpSPH's operation style, adopt the warpSPH integrator style, port across repos.
* **Part B (§1–§15, below): the original theory handoff** (edge-reduction identities, tiers, fibres), kept with its section numbers because other docs cite them (`HANDOFF.md §9/§10/§11/§13`). Its task list (§12) is now annotated with what was done.

Status legend: **[V]** verified numerically · **[D]** derived, not verified · **[P]** plan / open.
Everything below is committed (HEAD `e58ac7b`) except this file and the one-line `docs/README.md` pointer. The Tube Maps paper now lives in `~/dev/warpSPH/literature/nogina2026_tube-maps-tubular-coordinates.pdf` (synced 2026-10-03: bib key `nogina2026`, manifest row, abstract; `scripts/check_literature.py` passes; PDFs are git-ignored there) and was removed from this repo.

**Decisions (user, 2026-10-03):** (a) the solver **imports warpSPH on branch `dev`** (`warpSPH`, `warpSPHCore`, `warpSPHIntegrators` are editable installs in the `warp` conda env; verified importable) instead of vendoring modules; (b) the finished code **lives in warpSPH**: *scene management is frontend code* (the layer that owns bodies, builds a case, moves bodies, orchestrates adjacency and per-step operations), while the *scene representation and classes* (representations `SurfaceRep` / `VolumeRep` / `ImplicitRep` / `SdfRep`, moment pairs, the edge / pair kernels and tier tables, the boundary operations) **go into `warpSPHCore`**; (c) phase 2's target is set by the measured profile, not fixed in advance (see A4).

---

# Part A — state and plan

## A1. What exists (three layers, all 2D, all float64)

| layer | what | code | docs | status |
|---|---|---|---|---|
| **A. Edge machinery** (tiers 1–4) | exact `∫_T W`, `∇`, moments `m_α`, P0–P3 FEM nodal weights as edge line integrals with elementary primitives; tier 2 closed polylines (boundary data only); tier 3 closest-point curvature expansion `F0/F1/F2` (Hermite tables); tier 4 slender disk / strip series; hard tier switches; autodiff and Warp kernels | `core, primitives, kernels, geometry, oracle` (mpmath reference), `np2d, np_fem` (numpy), `torch2d`, `warp2d`, `warpbc` (Warp pair engine, far-field Gauss branch), `tier3, tier4, tier_select, implicitBodies` | `docs/derivations/*`, `exactness-and-approximations.md`, `backends-and-verification.md`, `autodiff-and-gpu.md`, `tier-selection-2d.md` | [V] 2D. Maple → mpmath → numpy → torch → Warp chain, 92 golden fixtures. The 2025 paper's benchmarks reproduced (`test_paper2025.py`). |
| **B. Scene layer** | `Scene` of `Body`s (pose, velocities, OBB) with `SurfaceRep` / `VolumeRep` / `ImplicitRep` / `SdfRep`, particle-cell broadphase, one adjacency per representation type, warpSPH-style operations (`Density, Interpolate, Gradient, Divergence, Curl, Covariance`, four gradient modes, per-query fields, per-body reactions), moving bodies without rebuilding structures, `inside`, `signed_distance` | `scene.py` (929 lines), `boundaryOps.py` (volume engine, warpSPH call shape) | `scene-architecture.md`, `boundary-operations.md` | [V]; `Laplacian` and per-query *vector* fields are **not** implemented |
| **C. Solvers on the scene layer** | `DFSPH2D` (omniSPH-style, validated vs the compiled omniSPH: tank, dam break, rotating obstacle with exact momentum bookkeeping, wall-suction fix); **`DeltaSPH2D`** (δ⁺-SPH, warpSPH `sun2017DeltaSPH` as live reference: tank, English wedge, Marrone 3.1 dam break, SPHERIC TC10 sloshing) | `dfsph2d.py`, `deltasph2d.py` (650 lines), runners `deltasph_validation.py`, `dfsph_validation.py`, snapshots/videos `deltasph_snap.py`, `dfsph_video.py`, figures `deltasph_compare.py` | `dfsph-validation.md`, `deltasph-validation.md`, `deltasph-plan.md`, **`deltasph-porting-notes.md`** (term map warpSPH ↔ here, ported / not ported, change log), `deltasph-resume.md` | [V] 2D |

Tests: `pytest tests/edge` (~2.5 min for the edge part); the three solver-related files `test_deltasph.py` (10) + `test_scene.py` (50) + `test_dfsph.py` (12) = 72, all passing at the last run (2026-10-02). I did not re-run them when writing this note.
Environment: `cd python; PATH=/home/lu26029/miniconda3/envs/warp/bin:$PATH`; GPU RTX PRO 6000 Blackwell; scratch in `.tmp/` (ignored).

## A2. δ⁺-SPH on analytic walls: results to hold (the regression baselines)

Reference = warpSPH (`~/dev/warpSPH`, branch `dev`, **never modified**; run separately; its own rules: videos on, one GPU run at a time) with identical initial particles. Commands in `deltasph-resume.md` («Reproduce»); series in `results/deltasph/` (tracked); snapshots and videos only in `.tmp/delta/` (not tracked, regenerate).

| case | warpSPH + mDBC | ours (analytic walls) | wall time ours / warpSPH |
|---|---|---|---|
| flat tank (English §4.1, dp 0.02, 4 s) | profile RMSE 0.0225, settled KE 1.1e-6 | **0.0019**, 1.3e-7 (surface = volume = SDF representations agree) | 5 min / 3.8 min |
| English wedge | all 8 probe checks; KE 1e-6 | all 8 pass, profile ~2× better; **settled KE 3.8e-4** (open, A5.4) | 8–13 min / – |
| Marrone 3.1 dam break (nx 67, shifting + noPen impulse = run "B") | P1 arrival t* 2.48, mean P* 0.353 / 0.478, max 1.08; KE series | arrival 2.49, 0.319 / 0.456, max 1.09; KE within 3 %, ρ within 3 %; P2 and the ceiling probe only qualitative (chaotic splash) | **1222 s / 147 s** |
| SPHERIC TC10 sloshing (nx 200, C4, 7 s, fixed dt 1e-4) | impacts 4.9 / 5.2 / 5.3 kPa (smoothed Gaussian probe), measured 1.8 / 2.3 / 1.2 | KE overlaps through 7 s, impact times equal (−0.047 s vs measurement, −0.053 s warpSPH), peaks 4.3 / 3.8 / 7.5 kPa | **5439 s / 2050 s** |

Without shifting + no-penetration (run "A") the dam break reads P1 1.6× high and ceiling impacts fling particles at 37 m/s; warpSPH itself without them reaches 116 m/s and stalls. Details and every number: `deltasph-validation.md`.

**Every later phase must reproduce these numbers** (they are the acceptance tests of the refactors). A fast regression set does not exist yet — each case takes 5–90 min. **First task of the next session: a regression script with truncated runs and tolerances** (proposal: tank 1 s KE + profile; dam break to t* = 3 with the P1 arrival and KE; sloshing to 2.6 s with the first impact time; compare against the stored `.npz` series). A repeating 5–90 min check is not workable for a refactor.

## A3. The quadrature paths (inventory for phase 1)

Everything in the solver that is **not** an exact kernel integral over the scene today. All three are the same wall-adjacent particles sampling the solid on a 24 × 96 polar grid around each particle with `Scene.inside` (`DeltaSPH2D._solid_samples`, `deltasph2d.py:146`; config `surfaceSamples`), recomputed in **every** RHS call (twice per step) and in `shift` (once per step). In 3D this is 24 × 48 × 96 ≈ 10⁵ samples and a 3D `inside` per wall-adjacent particle: not viable, and in 2D it is the main cost of the scene path.

| # | where | stands in for | exact replacement | what is needed | risk / note |
|---|---|---|---|---|---|
| Q1 | `rhs` l.255–266 (`wallViscosity`) | viscous wall term `M₂ = ∫_solid W′(r)/r · ŷ⊗ŷ dA` of warpSPH's pairwise Monaghan form with the free-slip mirror; sampled, 1 % of a fine half-plane integral | **Laplacian operation (user correction, 2026-10-02):** an exact boundary needs the naive Laplacian, not the pairwise form: wall term `ν_eff (v_b − v_i) Δλ / ρ`. **[D]** Green: `Δλ = ∫_T ∇²W dA′ = ∮ ∂ₙW ds = Σ_e z_e ∫_chord W′(r)/r ds` (tier 1/2); tier 3: second derivative of the planar table (+ κ terms). `W′/r` is a *polynomial* for Wendland and the splines (no negative powers; the plan's "negative-power potentials" worry does not arise). For a linear mirror field also the first moments `∫ y_a ∇²W = ∮ (y_a ∂ₙW − nₐ W) ds` **[D]** (Green's 2nd identity, `y_a` harmonic) | new scene op `Laplacian` (+ `Covariance`-like moment channel), new kernel profile `W′/r` through the truncated-monomial compiler; `MomentPairs` gets a `lap` channel (and `lap1`) | changes the operator relative to the reference; `ν_eff` must be **calibrated** (pair form: `α c0 H / (ξ 2(d+2))`, unverified; M₂ n and Δλ n are not proportional in general — design a test, e.g. normal-velocity decay at a wall against the pair form); expect a small effect on the dam break (α ≈ 0.01–0.02) |
| Q2 | `shift` l.308–317 (shifting tensile control) | `T = ∫_solid W⁴ ∇W dA` | `T = ∇_x ∫_solid W⁵/5 dA`: the gradient of the kernel `W⁵/5` (continuous, compact) — an exact tier integral with a new kernel entry | `W⁵/5` as a truncated-monomial kernel: degree 25 for C2, 40 for C4 → needs the **exact Chebyshev compile** (`np2d.cheb_coeffs`, `stable` mode) and a conditioning check in float64; tier-3 table entry for the same kernel (planar `F_k`) | conditioning is the open question; fallback: keep quadrature for this one term only |
| Q3a | `_detect_surface` l.168–173 (Barecasco cover vector, wall part) | `C_w = n_w ∫_solid unit(x_i − x′) 1[r ≤ H] dA` | **[D]** `= n_w ∇_x ∫_solid K dA`, `K(r) = (r − H) 1[r ≤ H]` — *continuous*, so the gradient is edge-local with no circle term (the notes' "gradient of `K = r`" is only right with this truncation: a plain `r 1[r≤H]` jumps at `H` and adds a circle term). Two monomials (n = 1, 0), same radius | the existing monomial machinery, gradient channel only | easy |
| Q3b | `_detect_surface` l.176–189 (cone count and all-neighbour count of the wall, compared with 0.5) | `area(solid ∩ disk(H) ∩ sector(axis c, half-angle π/6))` | a **clipping problem** (convex wedge ∩ polygon, then ∩ disk by the standard origin-centred signed-area sum per edge) — the one genuinely new geometric primitive; branchy per (particle, edge) | only `> 0.5` matters (`n_w·area ≥ 0.5` i.e. half a particle area, and only where the fluid count is 0): a coarse test is acceptable, an exact clip is cleaner. **Read openMaelstrom first** (A5.2): it reportedly does cover-vector surface detection against an SDF boundary and may not need a cone count at all |
| Q4 | consumers of `Scene.inside` | — | stays as a utility (initial layouts, cases) but leaves the per-step path | — | — |

Other approximations in the δ⁺ solver (not quadrature, but must be known): mirror velocity = the particle's own velocity (warpSPH mirrors the Shepard-interpolated velocity at the ghost); mirror normal `n = ∇λ/|∇λ|` per body (flat-wall approximation, curved walls need a per-query vector field); wall pressure `P_b = P_i + ρ(g − a_w)·(x′ − x_i)` clamped ≥ 0; no-penetration law re-derived for a static analytic wall at dp/2; `signed_distance` brute-force over edges; tier-3 model error / SDF validity switch; far-field Gauss branch in `warpbc` (controlled, 8 × 8, only on smooth far elements: stays).

## A4. The plan, in the user's order

Rationale for the order: (1) the quadrature is both the 3D blocker and the dominant cost of the scene path, and removing it fixes the list of per-pair channels the fused kernel must produce; (2) optimisation then has a fixed op list to fuse; (3) the refactor against warpSPH's call style is cheaper once the wall side is one aggregate struct from a few kernels; (4) the integrator style needs the RHS/stage structure to be stable; (5) the port needs all of that. **Each phase ends with the A2 numbers reproduced** (to a stated tolerance) and the 72 tests green.

### Phase 0 — regression harness (before touching anything)
Truncated-run regression script and tolerances (A2). Also profile once and record where the time goes (torch profiler over one dam-break step): the claim "scene adjacency + 3 ops per RHS ≈ 60 ms" is from the plan, the breakdown is not measured.

### Phase 1 — remove the quadrature paths (A3)
Order: Q3a (trivial, same machinery) → Q2 (new kernel entry, conditioning) → Q1 (new operation, derivation + calibration) → Q3b (clip, after reading openMaelstrom). For each: add the exact weight, a unit test against a **fine** polar oracle (`oracle.py`, ≥ 200 × 800) over particle positions around a polygon with a corner and a half plane (the current quadrature agrees only to ~1 %, so test against the oracle, not against it), then switch the solver (`cfg` switch, quadrature kept as a checked fallback until the end of the phase), re-run the A2 cases, record the delta in `deltasph-validation.md` and a change-log line in `deltasph-porting-notes.md` (the notes must be updated with every change: the user's rule). End of phase: delete `_solid_samples`, `surfaceSamples` and the polar branches; `Scene.inside` leaves the per-step path. Expected visible effect: smoother forces near edges, cheaper steps; the dam break should not change beyond the 1 % quadrature error.
Derivations to do (Maple/mpmath, as in Part B's workflow): Q1 `Δλ` edge form and the first moments of `∇²W`; Q3a edge form; Q2 compile conditioning of `W⁵/5`; Q3b clipped area. Add to `docs/derivations/` with the TEMPLATE check tables; golden fixtures as before.

### Phase 2 — optimise the scene path
Current per-step cost structure (`DeltaSPH2D.step`): **4 scene adjacency builds** (`rhs` k0, `rhs` k1, `shift`→`_surface_state`, `no_penetration`→`_wall_data`) and ~12 scene operations (`Density`, `Gradient`, per-query `Gradient`, `Covariance`), a Python loop over bodies for every op, `torch.nonzero` / `int(cnt.sum())` host syncs in the broadphase and adjacency (`scene.py` `buildCellList`, `ParticleCells`, `buildAdjacency`), and (until phase 1 is done) `Scene.inside` on ~2300 samples per wall-adjacent particle three times per step. Timing: dam break 8.3× and sloshing 2.7× slower than warpSPH (`neighbor_pairs` was already moved to a dense distance matrix for N ≤ 8000: 82 → 4 ms).
Candidate items (ranked by expected gain, **profile first**):
1. Reuse across the step: `no_penetration`'s wall data at the final positions is exactly what the *next* step's k0 `rhs` recomputes (same x; the wall ops of the Naive/field path do not depend on ρ) — one build instead of two **[to verify]**; the `near` selection (`lam.sum(0) > 1e-9`) could come from the broadphase candidate list instead of a separate `λ` evaluation **[idea, unchecked]**.
2. One fused wall-aggregate evaluation per query set: `λ, ∇λ, Cov, Δλ, a1-field integral A, cover vector, cone area` from one pass over the adjacency (a struct of per-particle wall data) instead of separate `sceneOperation` calls — this struct is also the interface for phase 3.
3. Fixed-capacity, Verlet-style adjacency (build every few steps with a skin, reuse in between; two `rhs` calls per step share it) — removes the data-dependent shapes and is the prerequisite for CUDA-graph capture (warpSPH captures its RHS in `schemes/deltaSPH.py::_graphedRHS`).
4. Batch bodies of one representation type in one launch (per-edge body index; ~3 ms launch overhead per body today). Needed for fibre bundles, irrelevant for a tank + obstacle.
5. Broadphase without host syncs; signed-distance with a spatial structure (brute force over edges today).
Target (user): **depends on the measured profile** (Phase 0). Comparable speed to warpSPH is the baseline wish; faster is plausible, because the wall side can be cheaper than particle walls. Idea from the user (recalled from openMaelstrom, **to confirm in its code**, A5.2): extrapolate the contact-point pressure from the *local fluid neighbourhood* (MLS at the contact point, as omniSPH / openMaelstrom do) instead of evaluating wall fields per query. Here that would turn the per-query hydrostatic field `P_b(x′) = P_i + ρ(g − a_w)·(x′ − x_i)` (needs the `a1` first-moment channels and the clamp of `_wall_excess`) into a per-pair constant, so the wall operation needs only `λ` and `∇λ` — fewer channels, no grid-based interpolation, one fused kernel. Trade-off to measure: accuracy against the exact hydrostatic closure (tank profile RMSE 0.0019 vs mDBC 0.0225 comes from that closure) and the sloshing / dam-break probes.

### Phase 3 — refactor to warpSPH's operation style (fluid–fluid calling warpSPH directly)
Today the fluid–fluid terms are hand-written torch pair sums on `neighbor_pairs` (checked only for the density / gradient sums against `warpOperation`, `test_pair_sums_match_warpoperation`). Target: each δ⁺ term is the warpSPH module call for the fluid part (`computeDensities`, `computeDensityDiffusion`, `computeVelocityDiffusion`, `computeMomentum`, `computePressureForceSurfaceAware`, `detectFreeSurface`, `computeDeltaShift`, with `operationMode=FluidToFluid`; term → module map in `deltasph-porting-notes.md §1`) plus **the wall part from the scene aggregate** added next to it. Steps:
1. State as warpSPH `ParticleState` / system state (not raw tensors); fluid adjacency from `buildVerletList`; the scene adjacency stays separate.
2. Replace one term at a time, each checked against the torch version to round-off in float64 (the existing pattern), then against A2.
3. The coupling points are the non-additive ones — handle them explicitly in the module interface: Antuono switch (`s_i`, dilated surface mask) and the clamped wall pressure `P_b ≥ 0` need per-particle fluid quantities at the wall term; the detector needs combined fluid + wall counts and covariance; shifting needs `λ_min` of the fluid + wall matrix. So the modules take an optional **wall aggregate** argument (the phase-2 struct) rather than the scene.
4. Drop the self-contained pieces that duplicate warpSPH (`neighbor_pairs`, kernels `wendland2/4`, EOS, dt rule) once the modules replace them; keep the solver as a thin scheme definition.
Decided: the solver **imports** warpSPH `dev` (and `warpSPHCore` / `warpSPHIntegrators`) from the `warp` env; nothing is vendored. Consequences: `~/dev/warpSPH` may move under us (pin the commit used for each A2 baseline run in `deltasph-validation.md`), and the reference runs must keep using a clean checkout (rule: warpSPH is not modified until phase 5).

### Phase 4 — warpSPHIntegrators style
`DeltaSPH2D.step` hand-codes the DualSPHysics symplectic Euler (k0 at tⁿ, half step, k1 at the half step, `x += dt/2 (v + v⁺)`), the time-centred continuity correction, shifting, the no-penetration impulse, `Body.move(dt/2)` twice, rolling gravity after the step, and the adaptive dt. warpSPHIntegrators (`~/dev/warpSPHIntegrators/src/warpSPHIntegrators`) provides: `symplecticEuler(state, dt, f, …)` (`verlet.py`) with `priorStep` first-stage reuse (`reuse.py`: FSAL / reuse analysis per scheme — this is the same saving as phase-2 item 1, in the library's terms), an `IntegrationSystem` protocol (`initializeNewState`, `apply_position/velocity/quantity/state_update`, `drift_rates(velocities, aux)` for the velocity-linear continuity rate = our time-centred continuity, `finalize` = the no-penetration shift), `RHS` objects. Mapping and open issues:
* `rhs` → `f(state, dt)`; time-centred continuity → `drift_rates`; shift and no-penetration → the system's `finalize` / postprocess hooks, as `systems/weaklyCompressible.py` does.
* **Bodies must become part of the integrated state**: `Body.move(dt/2)` at the half steps is an explicit-Euler pose update inside the step; with the library's stage buffers (`initializeNewState` copies) the scene has to see the *stage* pose. Make `Body` state a plain data object that the system copies and updates (`warpSPH.rigidBody.integrateRigidBody` is the model; the scene already tests "moved body = fresh body at the new pose to 1e-13").
* rolling gravity (`gravityFn`) = warpSPH `postStep`; fixed vs adaptive dt = the library's timestep hooks; other schemes (RK, leapfrog) then become available for free, only the symplectic Euler path is validated.
* the scene adjacency must be reusable across stages (phase 2 item 3) and graph-capturable.

### Phase 5 — port across repos
Per `deltasph-plan.md §3b` (strategy by the user: pull the 2D tiers into warpSPH generally and make them stable everywhere first; 3D = research afterwards). Steps: freeze a **dimension-generic interface** for bodies, representations and operations (implement 2D; every hard-coded 2 — polar sampling gone after phase 1, 2 × 2 inverses and eigenvalues, `[-y, x]` rotations, scalar `Pose.angle` — becomes dimension-generic); the scene as a **boundary provider** replacing the `kinds == 1` wall particles and the mDBC stage (`computeMdbcDensityEnglish2025`, `computeBoundaryVelocities`, `computeMdbcNoPenShift` removed: porting notes §1 lists every stage that disappears or changes); front-end support for bodies in cases; regression suite over schemes and integrators with this session's cases (tank, wedge, dam break, sloshing, hexagon, DFSPH); then **fibre-bundle demos** (a bundle of disks in cross-flow, batched primitive instances, one flat array of instances with per-instance reactions — plan §3b «main design point»). warpSPH rules apply inside that repo (its `PLANS.md` bookkeeping, videos on, one GPU run at a time). **Decided (user):** the code lives in warpSPH. Split: **scene management → warpSPH frontend** (bodies in a case definition, motion, per-step orchestration, the provider that feeds the schemes); **scene representation and classes → `warpSPHCore`** (the representation classes, `MomentPairs`, the edge / pair engine, tier tables, boundary operations in the `warpOperation` call shape, so other front ends can use them). Boundary cases to settle when the interface is frozen: `Body` / `Pose` (state vs. frontend object; ties to phase 4), the broadphase / adjacency builder (core operation with fixed-capacity buffers vs. frontend), where `Scene.inside` / `signed_distance` live. **Still open:** whether the mDBC path stays available as an alternative boundary provider, and the 3D scope of the first port (none, per the strategy above).
Parallel track (does not block 1–5): the 3D derivations (face → edge chain, tetrahedra, Tube-Maps `K` sign) of Part B §9, `docs/derivations/tet-face-edge-chain.md` [P].

## A5. Open items outside the phase list (ranked as in `deltasph-resume.md`, plus new ones)

1. **Wall-pressure sensor for analytic walls**: the first-order MLS probe at a wall point is erratic at the sloshing impacts (8.6 / 15.5 / 66 kPa spikes); take the wall pressure from the wall model (`P_b`, or force per length).
2. **openMaelstrom investigation (user TODO, unread):** `~/dev/openMaelstrom` — surface detection reportedly cover-vector based and working with an SDF boundary (`SPH/surface/surfaceDetection.cu(h)`), boundary friction (`SPH/DFSPH/dfsph.cu` `boundaryFrictionKernel` l.452, `volumeFrictionKernel` l.491; `SPH/boundary/volumeBoundary.cu(h)`), shifting near boundaries (`SPH/integration/simple.cu`?), and (user's recollection) the contact-point pressure extrapolated from the local fluid neighbourhood rather than from grid / field interpolation — relevant to the phase-2 speed idea. Questions: how the contact-point pressure is extrapolated and what the wall operation then needs from the geometry (a constant per-pair field → only `λ`, `∇λ`?); how the cover vector / cone test is evaluated against an SDF boundary (analytic or sampled); the friction formulation and whether it needs only `λ`, `∇λ`, distance; what, if anything, replaces the wall part of the tensile control. Expected (to be confirmed): shifting is then the only term whose wall part needs a new weight. **Do this at the start of phase 1** — it can remove Q3b.
3. **Wedge KE 400× the reference**: layout inconsistency (first-order moment error |M − I| = 0.23 on a smooth 30° slope vs 0.014 on a flat wall) — a regular lattice carved by a smooth wall is not an equilibrium. Needs a body-fitted packing; **deferred by the user** (general problem: airfoils, complex geometry). Tried and failed: first-order wall completion (much worse), `pack` along `S_i = 0` (halves KE, floor), damped `settle` (grows back). Staircase wall control is quiet, so the solver terms are right.
4. Moving bodies in δ⁺ (no-penetration law and viscous term assume static walls — the sloshing run uses a tank-fixed frame with rotating gravity), exact force bookkeeping (viscous / shifting / no-penetration contributions are not booked on the bodies; the pressure part is), torque (needs `p = 2` weights for position-dependent fields), curved-wall mirror normals (per-query vector field).
5. Not ported from warpSPH (`deltasph-porting-notes.md §4b`): renormalised DDT / pressure force variants, Michel-2022 and implicit shifting (warpSPH's sloshing default; ours is Sun 2017), `correctdrhodt/dvdt`, physical viscosity, other EOS, RK integrators, forcing hooks, periodic domains, interacting bodies.
6. Studies not run: dam break at H/dx 80 / 320, sloshing nx 100 / 400, no-shift and Michel variants, tiers 1 and 3 on the dam break, the hexagon in δ⁺.
7. Part B cross-cutting gaps still open: tier-switch continuity beyond the 2D disk study, curved exact elements, touching / overlapping bodies (double counting), the thickness ≈ h gap between tiers 3 and 4, 3D.

## A6. Facts and gotchas worth not rediscovering

* The scene layer is **not** limited by torch: its kernels are Warp float64 on torch memory (`wp.from_torch`). What blocks a port is orchestration: data-dependent shapes, per-step adjacency rebuild, Python loops over bodies, host-side representation switches, 2D only.
* warpSPH facts: DDT (`fourtakas2019`) is **fluid-to-fluid only** (no wall term, no new kernel); the Antuono switch sees the **dilated** surface mask (`fs`/`fsm` are bound swapped in `_deltaSPH_rhs`); time-centred continuity is a drift-field correction; PST has the full Sun-2019 surface treatment; isothermal EOS c0 = 20√(gH); Wendland C2 support 4 dx (C4 for sloshing); symplectic Euler = DualSPHysics; dt = Sun-2017 Eq. 5 (same step count in both codes).
* Bug found on the way (fixed, regression-tested): `neighbor_pairs` with cell = support dropped ~2 % of reverse pairs when dx divides the support (momentum not conserved); cell = 1.01 H. Any cell list on a commensurate lattice needs this check — also in the port.
* The scene's wall data consistency: at the first row the shift is a small difference of large terms (+2.24 fluid, −2.07 exact `G`, −0.14 tensile, net +0.03), so a 1 % error in one term is a few percent of the net — one reason the quadrature must go.
* Process gotchas: scripts that run at import time OOM'd the GPU when imported (keep builders in separate modules); a `&&` chain ending in `&` backgrounds the whole chain including `cd` / `export`; `pkill -f pattern` kills its own shell when the pattern appears in the command line (kill by PID from `ps | grep | awk`); in static-residual diagnostics `acc` already includes gravity; Maple/mpmath pitfalls are in `docs/derivations/*` and `backends-and-verification.md`.
* `PLAN.md` (tier 3 / oracle track, paused for review since 2026-10-01) still says "do not edit `HANDOFF.md`" — that was an instruction to the local model executing it at the time; this file was updated on the user's request.

## A7. Reading order for a new session
`HANDOFF.md` Part A → `docs/deltasph-resume.md` (reproduce commands, artefacts) → `docs/deltasph-porting-notes.md` (term map, §4b status, change log) → `docs/deltasph-plan.md §3b` (3D / port strategy, quadrature argument) → `docs/scene-architecture.md` → `python/edgebound/deltasph2d.py` and `scene.py`. Theory only when deriving: Part B and `docs/derivations/`.

---

# Part B — original theory handoff (edge reductions, tiers, fibres)

*Sections keep their original numbers. Status of the task list in §12 and the file list in §14 are updated (2026-10-03); the rest of this part is as written on 2026-10-01 (its status tags were not revisited; `docs/README.md` has the current status per derivation).*

Purpose: continue in a local code session (with Maple for derivation/verification).
Goal: replace the sector/segment/stub + 2F1 machinery of the 2D exact-triangle paper with
divergence-theorem reductions to **edge line integrals with elementary primitives**, extend to
**higher-order FEM nodal fields**, then to **tetrahedra**.

---

## 1. Context (short)

- **Tube Maps** (Nogina & Sellán, SIGGRAPH '26, doi:10.1145/3799902.3811118): second-order
  curvature expansion (tubular coords, $J(s)=1-2Hs+Ks^2$, angular averaging) of the
  Koschier–Bender *smoothed* density integral $\int\gamma(\Phi)W$. Density only; no pressure
  operator (pressure via SPlisHSPlasH host path, i.e. mirroring + grid gradients).
  Their "planar / Winchenbach 2020" baseline is their own $A_0$ term with linear $\gamma$.
  Main-text sign of $K$ in $J$ is inconsistent with their Eq. 16 — re-derive if reused.
- **Winchenbach et al. 2020** (doi:10.1145/3414685.3417829): semi-analytic planar, any quantity,
  pressure via boundary integral with MLS-extrapolated contact-point pressure.
- **Winchenbach & Kolb 2025** (arXiv:2507.21686): exact 2D triangle integrals via decomposition
  + Chebyshev + $_2F_1$. Stated drawbacks: branching (GPU), 2D only, tets "significant challenge".
- Key observation of this session: the same exact results follow from the divergence theorem
  with **only per-edge chord clipping + elementary antiderivatives** (atan, asinh/log, polynomials),
  including interpolants (not just gradients) via an **inside-indicator** term.

## 2. Notation / conventions (2D)

- Evaluation point $x$, $y = x' - x$, $r = |y|$. Normalise support $h=1$ (scale back at the end).
- Kernel normalised: $\int_{\mathbb R^2} W = 1 \Rightarrow 2\pi M(1)=1$, $M(r)=\int_0^r tW(t)\,dt$.
- Edge $e = (p_e,q_e)$, unit tangent $t_e$, **outward** normal $n_e$ (orientation from triangle sign).
- $z_e = n_e\cdot(p_e - x)$ (signed; $>0$ when $x$ is on the inner side of the edge line).
- Along the edge: $y = z_e n_e + s\,t_e$, $r=\sqrt{s^2+z_e^2}$, $s_0=t_e\cdot(p_e-x)$, $s_1=t_e\cdot(q_e-x)$.
- Chord for support radius $R$: if $|z_e|<R$, $L=\sqrt{R^2-z_e^2}$, $s\in[\max(s_0,-L),\min(s_1,L)]$.
- $\mathbb 1[x\in T]$: point-in-triangle test using the **same orientation predicate** as the signs of $z_e$
  (so the atan jump and the indicator jump cancel exactly when $x$ crosses an edge).

## 3. Core 2D identities  [V]

Verified for the full Wendland C4 kernel on 20 random triangles/points vs adaptive area quadrature,
max abs diff $3\times10^{-13}$ (quadrature-limited). Script: `edge_identity_check.py`.

$$\int_T W\,dA = \mathbb 1[x\in T] + \sum_e z_e\int_{\text{chord}}\frac{M(r)-M(1)}{r^2}\,ds$$

$$\nabla_x\!\int_T W(|x-x'|)\,dA' = -\sum_e n_e\int_{\text{chord}} W(r)\,ds$$

$$\int_T y\,W\,dA = -\sum_e n_e\int_{\text{chord}}\Psi(r)\,ds,\qquad \Psi(r)=\int_r^1 tW(t)\,dt\ \ (\text{compact})$$

Why: $\nabla\cdot\big(y\,M(r)/r^2\big)=W$. Beyond the support the flux $M(1)\,y/r^2$ is
divergence-free (non-compact!) → distant edges cancel on a closed boundary, leaving the indicator.
Gradient-type integrands have compact potentials → purely local edge integrals.

## 4. Arbitrary polynomial kernels: truncated monomials  [D]

Write any piecewise polynomial kernel as
$$W(r)=\sum_j q_j(r)\,\mathbb 1[r\le R_j],\qquad q_{\text{last}}=\text{outer piece},\ q_j=\text{piece}_j-\text{piece}_{j+1}.$$
Then everything is linear in **building blocks** $f_{n,R}(r)=r^n\,\mathbb 1[r\le R]$.

- Wendland: single $R=1$.
- Cubic spline (2D, support 1): $W=\sigma\,[\,2(1-q)^3_+ - 8(\tfrac12-q)^3_+\,]$, $\sigma=40/(7\pi)$
  → monomials with $R=1$ and $R=\tfrac12$.  (Check: for $q\le\tfrac12$ gives $1-6q^2+6q^3$.)

Per building block (value, $\alpha=0$):
$$\int_T r^n\mathbb 1[r\le R] = \mathbb 1[x\in T]\frac{2\pi R^{n+2}}{n+2}
 + \sum_e\frac{1}{n+2}\Big[z_e\,I_n(s) - R^{n+2}\arctan(s/z_e)\Big]_{\text{chord}}$$
Summed over a normalised kernel: indicator weights sum to $\mathbb 1[x\in T]$, and the atan
coefficient per edge sums to $M(R)$ → **one atan per edge per support radius**, not per monomial.

Gradient per block (exact per truncated term; circle jumps cancel between pieces of a continuous kernel):
$$\nabla_x\int_T r^n\mathbb 1[r\le R] = -\sum_e n_e\,[I_n(s)]_{\text{chord}}$$

## 5. Edge primitives (closed form)  [D]

With $r=\sqrt{s^2+z^2}$, use **odd** antiderivatives (so chords spanning $s=0$ add, not cancel):

- $I_m(s)=\int r^m ds$:
  $I_0=s$, $I_{-1}=\operatorname{asinh}(s/|z|)$, $I_{-2}=\arctan(s/z)/z$ (only ever appears as $z\,I_{-2}=\arctan(s/z)$),
  recurrence $(m+1)I_m = s\,r^m + m z^2 I_{m-2}$.
  Even $m\ge0$ → polynomial in $s$ (can expand $(s^2+z^2)^{m/2}$ binomially). Odd $m\ge1$ → ends in $I_{-1}$.
- $J_m(s)=\int s\,r^m ds = r^{m+2}/(m+2)$; $J_{-2}=\log r$.
- $\int s^j r^m ds$: substitute $s^2=r^2-z^2$ → linear combination of $I$ (j even) or $J$ (j odd).

Numerics: $z\to0$ — $I_{-1}$ diverges like $\log|z|$ but only enters multiplied by $z^2$ → guard exact $z=0$.
Short chords with both ends same sign → difference of nearby values (use series or short Gauss there).
Downward recurrence for $m<-2$ divides by $z^2$ — **avoid** (see §6, recursion route needs $m\ge-2$ only).

## 6. Higher moments $m_\alpha=\int_T y^\alpha f(r)\,dA$, $k=|\alpha|$

**(a) Recommended: compact-potential recursion (any dimension)  [D, k=1 is V]**
Define $\Phi[f](r) = -\int_r^R t\,f(t)\,dt$ (compact, $\Phi(R)=0$, continuous). Then $y_i f = \partial_i\Phi$ and, for $\alpha=\beta+e_i$:
$$\int_T y^{\beta+e_i} f\,dA = \sum_e n_{e,i}\int_{\text{chord}} y^\beta\,\Phi\,ds\;-\;\beta_i\int_T y^{\beta-e_i}\,\Phi\,dA .$$
Degree drops by 2 per step. Odd $k$ → purely edge integrals (compact). Even $k$ → ends in one
value-type integral (§3 form with $f\to$ iterated $\Phi$). For $f=r^n$: $\Phi=(r^{n+2}-R^{n+2})/(n+2)$,
still polynomial in $r$ → only $I_m, J_m$ with $m\ge -2$.

**(b) Alternative: direct far-field form  [D, implemented in `monomial_edge_forms.py`, unverified]**
$$\int_T y^\alpha r^n\mathbb 1[r\le R] = \mathbb 1[x\in T]\Big(\oint_{S^1}\omega^\alpha\Big)\frac{R^{n+2+k}}{n+2+k}
+\sum_e\frac{z_e}{n+2+k}\int_{\text{chord}}(z_en_e+st_e)^\alpha\Big[r^n-R^{n+2+k}r^{-(2+k)}\Big]ds$$
(from $\nabla\cdot(yPF)=P[(d+k)F+rF']$; far field $P\,y\,r^{-(d+k)}$ divergence-free; $\oint\omega^\alpha=0$ for odd $k$).
Needs $r^{-(2+k)}$ → worse conditioning near edges. Use mainly as an independent cross-check of (a).

**Gradients of moments  [D]:**
$\partial_{x_j}\int_T y^\alpha f = -\sum_e n_{e,j}\int_{\text{chord}} y^\alpha f\,ds$ (purely edge).

## 7. FEM nodal values (P1–P3, interior nodes)  [D]

Field on $T$: $A(x')=\sum_i A_i N_i(x')$, Lagrange basis of degree $p$ (P3 bubble node = just another node).
Re-expand each basis function about the evaluation point:
$$N_i(x+y)=\sum_{|\alpha|\le p} B_{\alpha i}(x)\,y^\alpha,\qquad B_{\alpha i}=\partial^\alpha N_i(x)/\alpha!$$
($N_i$ are polynomials in barycentrics, $\lambda_j(x+y)=\lambda_j(x)+\nabla\lambda_j\cdot y$ → cheap symbolic/numeric expansion.)

- Interpolant: $\langle A\rangle(x)=\sum_i A_i\,w_i$, $\;w_i=\sum_\alpha B_{\alpha i}\,m_\alpha$, $m_\alpha=\int_T y^\alpha W$.
- Gradient $\int_T A\,\nabla_x W$: $\sum_i A_i\sum_\alpha B_{\alpha i}\,g_\alpha$ with
  $g_\alpha=\int_T y^\alpha\nabla_xW = -\sum_e n_e\!\int_{\text{chord}} y^\alpha W\,ds + \sum_j\alpha_j e_j\,m_{\alpha-e_j}$.
- Difference / symmetric SPH forms: as in the 2D paper, swap nodal values ($f_i=\rho_i(A_i-A(x))$ etc.);
  weights unchanged.
- Two-way coupling: the same weights ($w_i$, $g_\alpha$-based) distribute the reaction to nodes →
  exact action = reaction between particle and element.
- Variable boundary density $\rho(x')$ nodal: $\rho A$ is degree $2p$ → just higher moments
  (or treat $\rho A$ as its own nodal field, inexact at degree $p$).
- Samples inside the triangle that are not a unisolvent nodal set (quadrature-point data, MLS samples):
  least-squares / L2-project to $P_p$ per element → $B$ via pseudo-inverse; exact w.r.t. the projection.
- **Conditioning risk [P]:** $B_{\alpha i}\sim L_T^{-|\alpha|}$, $m_\alpha\sim h^{|\alpha|}$ → cancellation
  $\sim (h/L_T)^p$ for elements much smaller than $h$. Quantify with mpmath; mitigations: scale $y$ by $h$,
  hybrid quadrature for elements fully inside the support, edge-local coordinates.

## 8. Pressure (fits as special cases)

- 2020 contact-point MLS pressure (per particle per element, constant) → only $m_0$, $g_0$.
- MLS-extrapolated linear pressure $p(x')=p_c+\nabla p\cdot(y+x-x_c)$ → moments $k\le1$, exact over real
  triangles; MLS order $q$ → moments up to $k=q$.
- Nodal pressure (paper's option (a), needed elements ~$h/4$ at P1) → §7 with P2/P3 now possible.

## 9. 3D plan  [P]

- Value: $\int_{\text{tet}} f = \mathbb 1[x\in\text{tet}]\,4\pi M_3(R) + \sum_f z_f\int_{\text{face}}\frac{M_3(r)-M_3(R)}{r^3}dA$,
  $M_3(r)=\int_0^r t^2 f\,dt$; integrand compact on the face disk $\rho\le\sqrt{R^2-z_f^2}$.
- Face integral = 2D value identity in-plane with radial profile $g(\rho)=z_f(M_3(r)-M_3(R))/r^3$,
  $r=\sqrt{\rho^2+z_f^2}$: indicator(foot in face) term + edge terms with $G(\rho)=\int_0^\rho t\,g\,dt=\int_{|z|}^{r} r'g\,dr'$ (elementary).
- New edge primitive family: $\int r^m/(s^2+w_e^2)\,ds$ with $r^2=s^2+w_e^2+z_f^2$ → elementary
  (solid-angle-type $\arctan\!\big(s z/(w r)\big)$ terms). Cross-check against Van Oosterom–Strackee
  triangle solid angle and closed-form polyhedral Newtonian potentials (Waldvogel 1979; Werner & Scheeres 1997).
- Moments: §6(a) recursion is dimension-independent (tet → faces with $y^\beta\Phi n_i$), then in-plane
  recursion; note the in-plane potential $-\int_\rho^{\rho_R} t\,\varphi\,dt = -\int_r^R r'\varphi\,dr'$ has the same form.
- Watch: $z_f\to0$ face-level jump vs indicator; edges/vertices of the tet inside the support.

## 10. Tier structure (roadmap)  [P]

| Tier | Input geometry | Method | Exact? | Valid when |
|---|---|---|---|---|
| 1 | Volume mesh (tris 2D / tets 3D) + fields inside (P0…Pk nodal) | per-element face/edge reductions (§3–7, §9) | yes (polytopes) | any thickness; conditioning for $L_T\ll h$ |
| 2 | Closed boundary mesh (polyline 2D / tri surface 3D), data on boundary only | same identities, interior faces cancel | yes | watertight, non-overlapping |
| 3 | Implicit surface (SDF + curvature) | closest-point expansion: order 0 = planar (2020), 1 = $H$, 2 = $H^2,K$ (Tube Maps), full shape operator for tensor moments | $O((h\kappa)^{n+1})$ | reach $\ge h$ on solid side, single sheet in support, smooth |
| 4 | Skeleton + radius (medial rep.): point → disk/ball, curve → strip (2D) / strand (3D), surface → shell/plate (3D) | expansion about the skeleton, or semi-analytic exact for straight pieces (§11) | approx / semi-analytic | thickness $a \ll h$ (series); any $a$ for straight segments |

Notes:
- Tiers 1 and 2 use the same machinery; the difference is where the data lives (volume vs boundary).
- Tiers 3 and 4 are complementary regimes (thickness $\gg h$ vs $\ll h$) → **gap at thickness $\sim h$**
  where neither expansion holds → fall back to tier 1/2 or semi-analytic forms.

Orthogonal axes (every tier must state what it supports):
- Field order: P0, P1, Pk nodal, MLS contact-point extrapolation of degree $q$.
- Operator: interpolant, gradient, div/curl, moments $m_0,m_1,m_2$ (kernel consistency corrections),
  viscosity/Laplacian (not linear in the field → single-contact-point treatment as in the 2D paper).
- Kernel: compact polynomial (exact), piecewise via truncated monomials, non-polynomial (tiers 3/4 or quadrature only).
- Geometry motion: rigid / deforming (tier 3 must recompute curvature; tiers 1/2 have no precompute),
  differentiability w.r.t. geometry (shape optimisation).

Cross-cutting gaps:
- **Tier selection per particle–object pair**, from $t/h$ (local thickness or reach), $h\kappa$, $L_T/h$, $a/h$.
  With adaptive resolution an object can move between tiers.
- **Continuity at tier switches**: match asymptotics (tier 4 series ↔ tier 1 exact prism as $a\to h$;
  tier 3 order 0 ↔ tier 2 on flat geometry) or blend; otherwise forces jump.
- **Curved exact elements** (between tiers 2 and 3): circular arcs / quadric patches lead to elliptic integrals.
- **Multiple / touching objects**: tiers 1/2 assume disjoint elements; implicit tiers sum per object →
  double counting at contact (fibre on wall, stacked bodies) — needs union handling.
- **Sub-resolution physics**: tiers give the geometric transfer only (§11 caveat).
- **Baselines**: particle boundaries (Akinci), density/volume maps, Tube Maps.

## 11. Fibres / codimension-2 boundaries (tier 4 detail)  [D/P]

**Why surface expansions (Tube Maps, tier 3) fail for $a<h$ / locally closed geometry:**
- Circumferential curvature $1/a$ → $h\kappa=h/a>1$: the distance expansion is not small.
- Surface tubular coords valid only to the reach ($=a$, the centreline); integrating $s$ to $h$ folds:
  $J=(1-s/a)(1-\kappa_2 s)$ hits zero and goes negative; the far side is never seen as a second sheet.
- Their high-curvature fallback is planar = half-space → overestimates by ~½ kernel vs true $O(\pi a^2\!\int W)$.
- Same failure for thin plates and narrow fluid gaps (multiple sheets in the support).
- I believe the Koschier–Bender smoothed integrand itself assumes the solid is ~$h$ thick → check their paper.
- Partial patch: cut $s$ at the local thickness (distance to the medial axis); $(1-s/a)$ on $[0,a]$ is then
  the exact polar Jacobian, but the kernel-distance expansion is still in $h/a$.

**Centreline parametrisation:** $x'=c(\tau)+\rho(\cos\varphi\,N+\sin\varphi\,B)$, Jacobian $\rho(1-\kappa_c\rho\cos\varphi)$
(Weyl tube formula setting). Small parameter is $h\kappa_c$ (fibre bending), usually small even for $a<h$.

**Straight segment, any $a$ (semi-analytic):** axial integral closed form with the same $I_m$ primitives
($z^2\to\zeta^2$, distance within the cross-section plane). The remaining cross-section integral is a radial
function over an offset disk → 2D divergence identity → one periodic integral around the circumference
(elliptic in closed form; trapezoid rule converges exponentially — split at kinks where the support sphere crosses).

**Slender limit $a\ll h$:**
$$\rho_B\approx\rho_0\pi a^2\sum_{\text{seg}}\int_{\text{chord}}\Big[W+\tfrac{a^2}{8}\Delta_\perp W+\tfrac{a^4}{192}\Delta_\perp^2W+\dots\Big]ds$$
(2D-disk mean-value series $\sum_k \Delta^k f\,a^{2k}/(4^k k!(k+1)!)$). Leading term = the edge-chord primitive with
$z$ = distance to the segment line, i.e. the same code as 2D edges. Wendland has no linear term → $\Delta_\perp W$
regular on the axis. Series is asymptotic (kernel smoothness at $r=h$) → truncate at C-order (C4: $a^2$ or $a^4$).
2D strip analog: 1D mean-value series $\sum_k a^{2k}\partial_\perp^{2k}W/(2k+1)!$.

**Curved centreline:** expansion in $\kappa_c$ (and torsion). Tube volume is independent of $\kappa_c$ (Weyl, $a<1/\kappa_c$),
but kernel weighting is not symmetric in $\varphi$ → first-order correction $\propto\kappa_c a^2$, depending on the
direction to $x$ relative to $N$.

**Point primitives:** 3D ball of radius $a$ at distance $d$ — shell theorem gives
$\int_{S^2} f(|x-c-\rho\omega|)\,d\omega = \frac{2\pi}{d\rho}\int_{|d-\rho|}^{d+\rho} r f(r)\,dr$ → **elementary** for polynomial kernels
(clip limits at $h$). 2D disk: no such reduction ($d\varphi$ is not $r\,dr$) → elliptic / periodic quadrature.

**Joints and ends:** polyline joints overlap (inner) / gap (outer) → mitred segments, ball joints, or
curved-segment correction. Ends: flat cap = 2D face problem; hemispherical cap = ball ∩ ball. Gradients are
elementary including segment-endpoint terms.

**Physical caveat:** for $a<h$ the peak density contribution is $O((a/h)^2)$ of $\rho_0$ → pressure-only
no-penetration is weak (leakage, cf. Tube Maps Fig. 15). Use the exact weights as the transfer operator for a
sub-resolution force model (slender-body / Morison drag, or penalty), and the same weights for the reaction on
fibre nodes (exact action = reaction).

## 12. Task list / test plan

Status 2026-10-03 (`[x]` done, `[~]` partly, `[ ]` open); details in `docs/README.md` and the derivation files.

1. [x] Maple: verify primitives $I_m$, $J_m$, $\int s^j r^m$ (odd forms), and the §4 per-monomial value/gradient (`maple/10`, `11`).
2. [x] Maple: verify §6(a) recursion for $k\le4$; cross-check against §6(b) (b is [V] for $z\ne0$ only) (`maple/12`).
3. [x] 2D closed form in float64 (mpmath reference, numpy `np2d`, torch, Warp); kernels Wendland C2/C4/C6, cubic spline (two radii), plus quartic/quintic/B7/B8/poly6.
4. [x] Exact tests (no quadrature): covering meshes reproduce the closed-form disk integrals (`test_np2d`, `test_boundary_ops`, `test_warpbc`).
5. [~] The 2025 paper's validation problems are reproduced (`test_paper2025.py`, `derivations/fem-nodal-weights.md §9`); no code-to-code speed comparison with their C++/PyTorch implementation.
6. [x] Robustness: on edge / at vertex, $z\to0$, tiny chords and elements, float32 (Chebyshev stable mode) (`test_degenerate`, `test_np2d_stable`).
7. [x] Autodiff (torch custom backward, Warp explicit adjoints, shape-gradient adjoint) vs analytic (`test_torch2d`, `test_warp2d`, `test_shape_gradient`; `docs/autodiff-and-gpu.md`). Not yet through the scene layer's per-query fields.
8. [x] FEM P1–P3 nodal weights and the conditioning study (`fem.py`, `np_fem.py`, `derivations/fem-nodal-weights.md`).
9. [~] GPU suitability: Warp float64 pair engine (a plan interpreter, ~30 M value-pairs/s) and the throughput tables of `boundary-operations.md §2`; no systematic branch-count analysis.
10. [ ] 3D derivation (§9) in Maple → tets (`derivations/tet-face-edge-chain.md` is [P]).
11. [~] Tier selection and switch continuity in 2D (disk obstacle regime map, hard switches, `tier-selection-2d.md`); no blending, no 3D.
12. [x] Tier 4 (2D): slender series vs exact polygon vs tier 2 and the crossover (`tier4.py`, `derivations/tier4-slender-series.md`).
13. [~] Tier 4: 2D disk series and the circle edge identity done; **3D ball primitive (shell theorem) open**.
14. [ ] Curved-centreline correction ($\kappa_c$ order 1) in Maple.

Added since (Part A): scene layer, DFSPH and δ⁺-SPH solvers on it, exact wall terms; the quadrature removal of A3 adds four derivations (Laplacian `Δλ`, `W⁵/5` compile, cover-vector monomials, clipped area).

## 13. Open questions

- Best stable evaluation of odd-$m$ primitives for short chords and large $m$.
- Conditioning of nodal re-expansion for $L_T\ll h$ (§7).
- On-edge / on-vertex conventions (half-plane limit should give exactly ½ of the disk contribution).
- 3D: is the face → edge chain cheaper than a direct tet ∩ ball decomposition in practice?

## 14. Files

Original (2026-10-01):
- `edge_identity_check.py` — [V] §3 identities vs area quadrature (edge integrals by `quad`).
- `monomial_edge_forms.py` — [D] closed-form primitives + §4/§6(b); brute-force reference timed out
  (discontinuity at $r=R$); swap in a polar reference (radial breakpoint at $R$) or Maple.

Since: the package `python/edgebound/` (module map in Part A §A1 and `docs/README.md`), `maple/` (`run_edge.sh`), `tests/edge/`, `tests/fixtures/` (golden fixtures), `docs/` (index in `docs/README.md`), `notebooks/edge2d_demo.ipynb`, `results/deltasph/` (tracked δ⁺ series and figures), `results/tables/` (tier-3 Hermite tables), `results/symbolic/`.

## 15. References

- Nogina & Sellán, Tube Maps, SIGGRAPH '26 — https://gatc.cs.columbia.edu/projects/tubemaps.html
- Winchenbach, Akhunov, Kolb, ACM TOG 39(6), 2020 — doi:10.1145/3414685.3417829
- Winchenbach & Kolb, arXiv:2507.21686 (2025)
- Koschier & Bender 2017 (density maps); Bender et al. 2019 (volume maps)
- Violeau & Mayrhofer 2014; Mayrhofer et al. 2015 (exact wall renormalisation via divergence theorem)
- Band et al. 2018 (MLS pressure boundaries)
- Pottmann et al., integral invariants (CAGD 2009) — curvature expansions of ball-neighbourhood integrals
