# Plan after WORK-008: adjacency = who interacts with whom; integrals are evaluated, not stored

Status 2026-10-05, `main` = `local-model` = `0ccfcf4` (not pushed). Supersedes HANDOFF Part A4 "Phase 2" (the candidate list there is folded in below); phases 3–5 of A4 stand. Written for continuing without the local-model protocol: steps are sized for one sitting each, with a verification route and a stop rule, not as work documents.

> **Prerequisite (user, 2026-10-05):** `docs/work-item-repo-layout.md` — move to `src/edgebound` + `scripts/`, pyproject, cleanup pass — comes first, so the file moves and the renames of step 2 are separate commits.

## 1. The terminology decision (user, 2026-10-05)

| term | means | holds | lives |
|---|---|---|---|
| **adjacency** | which particles interact with which boundary elements | integer pair topology only: `(query, element)` per body / representation, candidate lists, indicator candidates. Kernel-independent (depends on positions and supports only) | rebuilt or kept (Verlet skin) per position set |
| **precompute** | the per-pair integral values (`lam, g0, m1, g1` = 9 channels) of one kernel | floats, 72 B per pair | explicit, opt-in object (`PairMoments`); public API only if a benchmark shows it faster (DFSPH is the candidate, §4 step 6) |
| **evaluation** | integral + contraction with the per-query data, accumulated into per-query outputs | nothing per pair | a fused kernel over the adjacency |

Why it matters, measured: today `Scene.buildAdjacency` returns `SceneAdjacency` whose entries are `MomentPairs` (the precompute). It is consumed once by cover (3 builds/step), tensile (1), lap (2), and 1–3 times by `_wall_data` / `_surface_state` (lam, G, Cov, A read *disjoint* channels). Consequences already visible in the code: a kernel guard and a channels guard in `sceneOperation` (they exist only because the "adjacency" holds data), `SceneAdjacency.channels`, the pruning of WORK-008, a cache for the cross-step reuse. The same confusion sits in `boundaryOps.BoundaryAdjacency` (its docstring: "an adjacency list that already carries the geometry"; fields `weights`, `gradWeights`). warpSPH's own `AdjacencyListWarp` / `CompactHashMap` are topology only and the operations evaluate kernels on the fly; the boundary side should match.

Memory (illustrative, not measured): 10⁷ near-wall queries × 50 faces/query in 3D = 5·10⁸ pairs × 72 B = 36 GB for *one kernel*. Precompute cannot be the default at scale; in 2D with 484 pairs it is irrelevant, which is why it went unnoticed.

Where precompute might stay (user decision 2026-10-05): **the metric is a benchmark, not a principle.** The default for every solver is evaluation (fused, recompute). δ⁺-SPH reads each wall quantity only a few times per step, so storing it never pays. For fluid–fluid sums the ratio of memory reads to arithmetic is poor enough that recomputing terms hides behind latency, so precompute is not worth it there either. DFSPH reuses wall values over the 10–100 pressure iterations, so it is the one candidate: build its wall contribution as a fused evaluation too, then benchmark fused-evaluation-per-iteration against precompute-and-reuse (§4 step 6). If precompute is legitimately faster at the sizes that matter it becomes a public API (`PairMoments`, documented with its memory cost); if not, it stays an internal reference path for tests. Precision (user, 2026-10-05): most production runs are FP32; large FP64 runs go to H200-class hardware (high FP64 rate) elsewhere. So the benchmark metric is **FP32 first**, FP64 second, and nothing may be tuned only to this card's 1/64 FP64 rate. Consequence for the design: the fused kernels are dtype-generic, and the FP32 path uses the stable (Chebyshev) plans, not the monomial ones (monomial-basis cancellation amplifies float32 error 1e2–2e4×; the Chebyshev compile with `stable=(8,6)` gives f32 value 8e-8 / grad 3e-7, `docs/exactness-and-approximations.md`). `warpbc` is f64-only today; an f32 kernel is part of step 1.

## 2. Where the time is (dam break nx 67, 35.5 ms/step; sloshing nx 200, 44.7)

Per step, 9 builds (reviewer probe `docs/work/refs/review8_build_breakdown_probe.py`, GPU otherwise idle):

| consumer | kernel, channels | calls | ms/call | ms/step |
|---|---|---:|---:|---:|
| `_wall_data` | `w2`, all 9 | 2 (the 3rd is the cross-step cache) | 3.20 | 6.39 |
| `lap_lambda_scene` | `lw2`, all 9 | 2 | 2.79 | 5.58 |
| `_surface_state` | `w2`, all 9 | 1 | 3.20 | 3.20 |
| `cover_vector_scene` | `cone`, (3,4) | 3 | 0.97 | 2.90 |
| `tensile_vector_scene` | `w2p5`, (3,4) | 1 | 2.39 | 2.39 |
| | | 9 | | **20.46 (58 % of the step)** |

Also per step: `_detect_surface` 5.1 ms self (host-side torch around the cone area; 3 calls), fluid pair sums 9.0 ms, `neighbor_pairs` 2.1. warpSPH's dam break is 147 s over the same step count ≈ 7.5 ms/step (implied, not re-measured), sloshing ≈ 29 ms/step: we are 4.7× / 1.5× slower.

Two measurements that fix the design (`docs/work/refs/review8_latency_probe.py`, dam-break state, 484 pairs, kernel `w2`, all channels):

| pairs | 484 | 4 840 | 48 400 | 484 000 |
|---|---:|---:|---:|---:|
| `edge_channels` ms | 1.82 | 1.86 | 3.03 | 18.0 |

(`w2` channels (3,4): 0.36 / 0.35 / 0.63 / 3.1 ms; `cone` (3,4): 0.17 / 0.17 / 0.27 / 1.1.)

1. **The edge kernel is latency-bound, not throughput-bound**: a 10× larger pair set costs nothing. One thread per pair walks all terms serially (FP64 on this card, 16 edge + 5 vertex terms, each a polynomial with `atan`/`asinh`/`log` primitives). The remedy is *more parallelism per pair* (a thread per (pair, term group), fixed-order reduction), not fewer launches alone. This is a hypothesis about the cause that the flat curve supports; the prototype in step 1 tests it. **Caveat: the curve was measured in FP64 on a card with a 1/64 FP64 rate, where every term is a long dependent chain of slow FP64 operations; in FP32 (the target) the per-term latency is much smaller and the curve may be flatter for fewer pairs or not at all. Step 1 therefore starts by measuring the same curve with an f32 stable-plan kernel, before any term-parallel restructuring.**
2. About 7.5 of the 20.5 ms is *not* the edge kernel (20.5 − 12.9 estimated from the per-kernel probe numbers): pair list, cell list, `nonzero` syncs, indicator, `toWorld`, allocation. That is the kernel-independent part and it is paid 9 times for 4 distinct position sets (x^n, x^{n+½}, x^{n+1} before shifting, x^{n+1} after shifting).

## 3. Target

```
adjacency  = scene.adjacency(particles)                 # ints; once per position set, later: Verlet skin, fixed capacity
fields     = scene.evaluate(adjacency, particles, spec) # one fused pass: all kernels / outputs the step needs at these positions
```

* `spec` lists named outputs, each = (kernel, contraction): `lam` (w2), `G` = const-field gradient, `Cov` (w2), `A` (gradient with the per-query field `a1`), `lap` (lw2: `2λ[L] − tr Cov[L]`), `cover` (cone, g0), `tensile` (w2p5, g0). The kernel needs, per (query, edge), the local geometry once (`z`, chord interval) and then evaluates the term sets of each requested plan; contraction with per-query data and the pose rotation happen in the kernel (rotate once per query at the end); per-body outputs `[B,N,…]` come from a body index per edge (this also is "batch bodies of one representation" of the old phase-2 list).
* The indicator pseudo-pairs are a per-query term (winding number inside a body): a small per-query pass, not pair rows.
* `fields` is the wall aggregate struct of phase 3 (HANDOFF A4), now with a definition.
* The old path (`PairMoments` + `sceneOperation`) stays as the **reference implementation** while the fused path is built: the new kernels are tested against it, and volume / implicit representations keep it until they get their own fused path. DFSPH moves to fused evaluation too (§4 step 6); whether the old path then survives as a public precompute API is decided by that benchmark.
* The kernel guard and the channels guard disappear from the adjacency (an adjacency has no kernel); the channel check moves to `PairMoments`, where the data is.

## 4. Steps (each ends green: `pytest tests/edge` = 767 + new tests, harness `check --physics --cases tank,dambreak`)

**Step 1 — f32 stable-plan edge kernel, then term-parallel (isolated, no API change).** First an f32 kernel on the Chebyshev plans and the pairs-vs-time curve in f32 and f64 (`review8_latency_probe.py` extended). Only if the f32 curve is also latency-bound, prototype: a thread per (pair, term group) writing to a scratch `[P, groups, 9]`, fixed-order reduction (keep it deterministic: atomics would add run-to-run spread to a harness that is bit-level). Measure at 484 / 4 840 / 48 400 pairs for `w2`, `w2p5`, `cone`. Expected (to be measured): `w2` 1.8 → well under 0.5 ms at 484 pairs. If the speed-up is < 2× the latency hypothesis is wrong: stop, re-read the profile before step 3. Acceptance: pruned and full results equal the old kernel to a stated tolerance (the sums change order if terms are reduced differently; with a fixed per-channel term order they stay bit-identical — prefer that), `test_channel_pruning.py` unchanged.

**Step 2 — split adjacency from precompute (refactor, results bit-identical).**
* `Scene.buildAdjacency` → topology only; `Scene.precompute(adjacency, kernel, channels) -> PairMoments`; `sceneOperation(…, adjacency | pair_moments)` (an adjacency alone triggers a precompute for the call, so every current consumer keeps working; consumers that call several operations pass the `PairMoments`).
* One adjacency per position set shared by wall / lap / cover / tensile: 9 builds → 4 adjacencies (the 0.8 ms kernel-independent part × 5 saved ≈ −4 ms/step, estimate).
* Delete: the kernel and channels guards of `sceneOperation` in their adjacency form (re-home on `PairMoments`), `SceneAdjacency.channels`; keep the `DeltaSPH2D._wallCache` (it caches *results*, `lam`/`G`, keyed on state — rename `_wallResultCache`).
* Rename in code and docs (`scene-architecture.md` §6b needs a rewrite of the section that describes the adjacency as holding moments); `boundaryOps.BoundaryAdjacency` gets the same split in the same step or a follow-up (weights → `BoundaryPrecompute`).
* Tests to edit deliberately: `test_wall_data_reuse.py::(e)` pins "9 builds"; it becomes "4 adjacencies" and a precompute count; `test_channel_pruning.py` (c) guard tests move from the adjacency to `PairMoments`; `test_scene.py::test_adjacency_kernel_guard`. Name them in the commit message.

**Step 3 — fused evaluation (`scene.evaluate`) for `SurfaceRep`.** Order inside the step: (a) `lam` + `G` + `Cov` (wall, `_surface_state`); (b) `A` with per-query `a1`; (c) `lap`; (d) `cover` + `tensile`; (e) per-body reactions (force from the same pair contribution). Each output is checked against the old path on the L-shaped body of `test_cover_scene.py` and the tank (inside-the-body and tangent cases included: the indicator is where silent errors hide, lesson of WORK-004), then `_wall_data`, `_surface_state`, `_detect_surface`, `lap`, `tensile` switch over one by one, harness after each. Expected (estimate): builds 20.5 → 7–9 ms/step, step 35.5 → ~23 ms (dam break), i.e. ~3× warpSPH instead of 4.7×. Without step 1 the fused kernel inherits the latency problem; do 1 first.

**Step 4 — cone area as a Warp kernel; remove the host-side torch around the detector.** `_detect_surface` is 5.1 ms self for 3 calls; `cone_area_scene` is a closed form per (query, edge) (polar sectors, non-local chord test — see `docs/derivations`), so it belongs in the same fused pass or its own launch over the same adjacency. Also: `near` as a mask / fixed-capacity compaction instead of `torch.nonzero` (host sync).

**Step 5 — fixed-capacity, sync-free adjacency (Verlet skin) and a graph-capturable step.** Adjacency rebuilt every k steps with support + skin (rebuild trigger on max displacement, as warpSPH's `verletScale`), capacity from a safe bound with an overflow flag; evaluation keeps the exact distance logic (an extra pair contributes exactly 0). Prerequisite for CUDA-graph capture (`warpSPH/schemes/deltaSPH.py::_graphedRHS`). The pair sums are no longer bit-identical to the rebuild-every-step runs (zero-valued extra pairs change summation order only): the harness margin decides whether this is "within spread"; if not, re-recording the baseline is the user's call, with the physics gate and the sloshing KE line as evidence.

**Step 6 — DFSPH: fuse and benchmark (decides the fate of precompute).** Express DFSPH's wall operations (`Density`, `Gradient`, `Interpolate`, per-body reactions in `dfsph2d.py`) as `evaluate` specs, including the ones inside the pressure iteration. Benchmark on the validated cases (tank, dam break at 2k / 8k / 22k particles, rotating obstacle) against the precompute-and-reuse path: ms per step and per pressure solve at equal solver tolerance and equal iteration counts, wall-time scaling with N, peak memory. Primary: FP32; FP64 second (and on H200-class hardware if available); a second point at larger N where memory traffic dominates. Rule: precompute becomes a documented public API only if it is faster at some size that matters; otherwise it stays a test oracle. Fusing the DFSPH operations may help regardless (fewer launches per iteration).

After step 5 the scene path is a handful of launches per position set on a fixed-shape structure; the remaining distance to warpSPH is the fluid–fluid side (9 ms alone), which is phase 3.

## 5. After the speed work (HANDOFF A4 phases 3–5, with what the steps above change)

* **Phase 3.** `fields` from step 3 *is* the wall aggregate; fluid terms move to warpSPH modules one at a time (`computeDensities`, `computeDensityDiffusion`, `computeVelocityDiffusion`, `computeMomentum`, `computePressureForceSurfaceAware`, `detectFreeSurface`, `computeDeltaShift`, `FluidToFluid`), each checked to round-off against the torch version, then against the A2 numbers. Needs two decisions: pin a warpSPH commit per baseline; whether mDBC stays as an alternative provider. This phase is where the 9 ms of fluid pair sums and the 2.1 ms `neighbor_pairs` go.
* **Before phase 4 (moving bodies in δ⁺).** The no-penetration law and the viscous term still assume static walls (sloshing uses a tank-fixed frame); `wallViscosityForm="noslip"` already reads `velocityAt`. Needed: no-penetration with a wall velocity, force/torque bookkeeping of the viscous, shifting and no-penetration contributions (only the pressure part is booked), an absolute test (a wall moving at constant velocity in a closed box is Galilean-invariant; a piston changes the hydrostatic profile by a known amount). The fused kernel of step 3 should return per-body reactions for exactly this.
* **Phase 4** (warpSPHIntegrators style: `symplecticEuler`, `drift_rates`, `finalize`, bodies in the integrated state), **phase 5** (dimension-generic interface, boundary provider replaces `kinds == 1` + mDBC, scene representation → warpSPHCore, management → warpSPH frontend, fibre-bundle demos with batched primitive instances) unchanged from HANDOFF A4. The adjacency/evaluation split above is the interface freeze item "broadphase / adjacency builder boundary": adjacency is a core data structure (fixed capacity), evaluation is a core operation.

## 6. Research items to fold in (do early, they can remove work)

1. **openMaelstrom read — done 2026-10-05** (`SPH/surface/surfaceDetection.cu`, `utility/SPH/boundaryFunctions.h`, `SPH/DFSPH/dfsph.cu`, `SPH/integration/simple.cu`; everything float32). What it does and what it means here:
   * **Wall quantities from distance only.** The wall enters as a 1-D lookup table over the signed distance `d` along the plane normal / SDF gradient: `λ(d)` and `∇λ(d)` of a *flat* wall (`boundary::lookupValue/lookupGradient`, `splineGradient`, interpolated, with a density-dependent offset); `SdfRep` / volume bodies supply `d` and `n` via `distance_fn`. This is our tier-3 order-0 (`Tier3.planar_moments`) without the curvature terms. Exact for flat walls, error O(κH) at curved walls and wrong near corners (no edge sum at all).
   * **Contact-point pressure.** For a near-wall particle, `p_b` = first-order MLS (SVD pseudo-inverse of the moment matrix) of the *fluid* pressures around the contact point `x_b = x_i − d n`; wall term `−ρ0 (p_i/ρ_i² + p_b/ρ0²) ∇λ(d)`. The wall operation then needs `λ, ∇λ` only (channels 0, 3, 4), not our field integral `A = μ∫(a1·y)∇W` (the `g1` contraction); the cost moves into one extra fluid-neighbourhood MLS pass per near-wall particle (a pair-sum of the kind that hides behind latency, and it is what warpSPH-style fluid modules do well). The first moments of `∇W` over the wall (our `Cov`/`A`) are what close the hydrostatic balance exactly at a wall; a uniform `p_b` over the support loses the `∇p·∫ y⊗∇W` term and openMaelstrom compensates by a calibrated offset in the LUT. Whether this keeps our tank profile (RMSE 0.0019 vs mDBC 0.0225) is **open and result-changing**; it is a candidate variant behind a switch, gated on the tank / wedge / dam break, not a replacement.
   * **Surface detection has no wall integral.** The normal is `Σ unit(x_i − x_j)` over *fluid* neighbours only; the free-surface flag is the 30° cone test on fluid neighbours, plus, within `h0` of a wall, `angle(normal, −d direction) ≤ 90°` from the wall distance; then a propagated distance field. So Q3a / Q3b (our exact wall cover vector and cone count, 2.9 ms + the cone area + host code in `_detect_surface`, ≈ 8 ms of the 35.5) have no counterpart there. A distance-based wall rule would cost one `signed_distance` (0.7 ms) but is a different detector from warpSPH's Barecasco-with-wall-particles that the A2 baselines use: again a gated variant.
   * **Friction** (`boundaryFrictionKernel`, `volumeFrictionKernel`): Coulomb-limited tangential impulse at the contact point, limit `μ|F_p|` with `F_p` the wall pressure force (which uses `∇λ(d)`), relative velocity from the body's linear and angular velocity at `x_b`, force/torque on the body from the same impulse. Needs `d`, `n`, `x_b` only, no integrals. It is not our no-slip flux term; for DFSPH-style solvers it is the cheap option and the moving-body bookkeeping pattern (force + torque at the contact point) is the template for the per-body reactions of step 3.
   * **Shifting:** the particle-shifting call is commented out in the integrator and positions are projected onto the planes / SDF; there is no wall tensile term to compare with (so Q2 stays as is, and `W⁵/5` remains the only route).
   **Net for the plan:** (a) the stable plan for `lam, G` could be replaced by an SDF + LUT evaluation wherever the wall is flat to a tolerance (`hκ` small, no vertex within `H`), with the exact edge sum as the fallback near corners: the existing tier-3 hard switch with the polygon fallback (`SdfRep`, `TierPolicy`). This is the cheapest evaluation of all and an independent speed-up of steps 3–5; add it as step 3b once the fused kernel exists, measured as accuracy-vs-cost on the tank, wedge (smooth slope: where the flat model is best) and dam break (corners: where it fails). (b) The contact-point pressure and distance-based detector are two result-changing variants for after phase 3, when the fluid-side MLS is a warpSPH module call.
2. 3D face→edge chain (Part B §9) is independent; the fused-kernel design above must not hard-code 2: term groups and the per-(query, element) geometry are the element-type-specific part, the contraction and accumulation are not.
3. Parallel, not blocking: PLAN.md (tier 3 / oracle), wedge packing, wall-pressure sensor for the sloshing impacts (HANDOFF A5.1, A5.3).

## 7. Open decisions (user)

* ~~Step 2 names~~ Decided: `PairMoments` / `precompute` / `evaluate` stand (user, 2026-10-05).
* ~~Keep `PairMoments` public?~~ Decided: by benchmark (step 6). Evaluation is the default for all solvers; precompute is public only if it measurably wins.
* Step 5: accept a harness margin > 0 from reordered sums, or insist on bit-identity (then no skin, only the sync-free rebuild)?
* Whether `Scene.inside` / `signed_distance` move into the adjacency step (both are brute force over edges; `signed_distance` is 0.7 ms/step now, negligible until large scenes).

## 8. Facts and files

* Probes: `docs/work/refs/review8_build_breakdown_probe.py` (per-build table), `docs/work/refs/review8_latency_probe.py` (pairs-vs-time curve), `review7_edge_cost_probe.py`, `review7_prune_probe.py`.
* The `buildAdjacency` = 9/step pin: `tests/edge/test_wall_data_reuse.py::test_reuse_saves_one_adjacency_per_step`.
* Review record and reviewer lessons: `docs/work/logs/REVIEW-008.md`, `docs/work/REVIEW.md`.
