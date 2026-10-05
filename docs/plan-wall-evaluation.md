# Plan after WORK-008: adjacency = who interacts with whom; integrals are evaluated, not stored

Status 2026-10-05 (evening), `main` = `repo-layout` = `09ff4ca` (layout + cleanup 1/2 merged, not pushed); `paper-audit` = `d97ef10` is one docs/paper commit on top; `local-model` (`0ccfcf4`) and `wall-evaluation` (`2d2a727`) are stale and can be deleted. Supersedes HANDOFF Part A4 "Phase 2" (the candidate list there is folded in below); phases 3–5 of A4 stand. Written for continuing without the local-model protocol: steps are sized for one sitting each, with a verification route and a stop rule, not as work documents.

> **Prerequisite (done, merged to `main`):** `docs/work-item-repo-layout.md` — `src/edgebound/{edge,scene,sim}` + `scripts/`, pyproject, cleanup pass. Paths below use the new layout (`src/edgebound/scene/scene.py`, `edge/warpbc.py`, `sim/deltasph2d.py`, `tests/{scene,sim}`).

## 0. Next steps, in order (resumed 2026-10-05 after the layout landed)

Order differs from §4 in one respect: step 2 (refactor, deterministic) does not depend on step 1 (measurement-gated), so the measurement goes first and costs one sitting; the refactor then proceeds while the kernel decision is made. Work on a new branch off `main` (`wall-eval`); `paper-audit` is merged or not at your call, it touches only `paper/`, `docs/`, `scripts/derivation_checks`.

| # | what | sitting | gate to leave the step |
|---|---|---|---|
| 0 | **Baseline — DONE on an idle GPU (2026-10-05 night; the earlier 35.5 / 44.7 ms/step were taken with the local model resident)**: suite 816 → 827 (see the 2b / 2c rows), harness `check --cases tank,dambreak` PASS (2b); dam break nx 67 **30.8 ms/step**, sloshing nx 100 **22.2 ms/step** (300 / 200 steps after 10 warm-up, `scratchpad/prec_solver.py`); adjacency + precompute 16.7 of 32.6 ms (51 %) with the sync wrappers of `review8_build_breakdown_probe.py` | done | numbers recorded here |
| 1a | **Measure — DONE** (`docs/work/refs/step1a_kernel_vs_host_probe.py`, `scripts/studies/edge_f32_probe.py`; results in §9). The 2 ms is the KERNEL (host enqueue 0.025 ms, event-timed kernel 1.92 ms at 484 pairs, `w2` all channels, monomial f64). **float32 on the Chebyshev plans (the warpSPH precision mechanism, §1): 0.26 ms at 484 pairs (7× the monomial f64 default, 11× the Chebyshev f64), gradient-only channels 0.05 ms, error 4e-6 abs / 3e-5 rel**; still flat in the pair count (0.26 ms at 484 and 4 840 pairs, 1.25 ms at 484 000 = 200× above the per-pair throughput), i.e. **latency-bound in float32 as well → 1b is justified**. Solver in float32 mode (wall kernels only, rest of the step float64 torch): same physics (positions 5e-8, KE 3e-8 – 1e-7 relative over 200 – 300 steps), dam break 30.8 → 23.2 ms/step, sloshing 22.2 → 13.1 | done | decision: 1b yes |
| 2a | **Evaluation schedule — DONE** (`docs/work/refs/schedule_probe.py`, counts only; table in §8). Result: 4 position sets per dam-break step, each with a full query set (all N) and a near-wall subset (272 of 1152), the subset being an *exact row subset* (same positions, same supports; checked bitwise) of the full set; the full-set adjacency of x^n is the cross-step cache. So the target is not "9 → 4" but **9 → 3 full-set builds per dam-break step (2 per sloshing step), the subset consumers (cover, lap, tensile) take a restriction of the full adjacency** (select pairs with `q ∈ near`, remap `q`), which needs a query-subset operation on the adjacency in 2b | done | `test_wall_data_reuse.py::(e)` pin unchanged |
| 2b | **Split adjacency from precompute — DONE** on branch `wall-eval` (suite 821 passed = 816 + 5 new in `tests/scene/test_adjacency_split.py`; `Scene.adjacency` / `precompute` / `pairMoments` / `moments`, `SceneAdjacency.restrict`, `SurfaceTopology`, `PairMoments`; per dam-break step 3 adjacencies + 9 precomputes, was 9 builds; `DeltaSPH2D._wall_state` carries the wall adjacency, cover / lap / tensile take `restrict(near)`). Named test edits: `test_wall_data_reuse.py` ((e) pins now 3 / 9 and 4 / 10, `_disable_cache` patches `_wall_state`, cache tuple index), `test_scene.py` (kernel-guard test uses `moments=` and the new message), `test_viscosity_scene.py` (`counting` takes `channels`) | done | suite + harness |
| 1b | only if 1a says latency-bound: term-parallel prototype | 1–2 | ≥ 2× at 484 pairs, else stop |
| 2c | **`BoxRep` — DONE** (`docs/box-domain-primitive.md` §3b; `scene/box.py`, `BoxRep` in `scene.py`, `domain_scene('box')`, `tests/scene/test_box_rep.py` 6 cases + `box` in the two DFSPH representation tests): exact against the polygon path to 4e-11 (w2 lam) / 1e-7 (w2 g0) / 2e-9 (w4 g0) / 8e-7 (lw2 g0) / 4e-11 (w2p5 g0); `cone` (kinked profile) uses the exact polygon path. On the box domain the real cases match the polygon domain: tank (400 steps) max\|dx\| 7e-9, max\|dv\| 4e-8, KE 3e-7 relative; dam break nx 40 (400 steps) 2e-9 / 4e-8 / 1e-9; sloshing nx 100 (300 steps) 6e-13 / 6e-11 / 9e-11; DFSPH block 2e-10. Cost in eager torch (GPU shared, preliminary): dam break nx 67 122 vs 128 ms/step, sloshing 78 vs 86 — a fused Warp kernel (step 3) is the real speed-up (no pair list, no latency chain). The bit-level harness is NOT applicable (recorded with the polygon domain; default domains are unchanged) | done | suite + physics equivalence above |
| 3 | fused `evaluate` for `SurfaceRep`, output by output (§4 step 3) | 4–6 | each output equal to the old path on the L-shape + tank, harness after each switch-over |

Design constraints that came out of the paper audit and must be in the step-3 spec (not extra work): (a) the output spec names a kernel *and* a moment order — `plan-exact-wall-laplacian.md` needs moments of `∇²W` up to degree 2 in the wall frame per (query, edge) (6 channels per component, E6 there), so `spec` must accept `(kernel, channels, contraction)` where the contraction is a per-query coefficient vector, not a fixed list of the 9 current channels; (b) the 3D face→edge chain must not be hard-coded out (term groups and per-(query, element) geometry are the element-specific part).

What step 2 actually has to handle, from reading `scene.py` (the plan text above says "pair topology only" but the three representation types differ):
* `SurfaceRep`: topology = `(qi, e)` after the segment-distance filter (kernel-independent; depends on `lsup`) plus the indicator candidates `(nz, ind)` (winding number, kernel-independent, only needed when a `lam`/`g1` channel is requested). `rep.pairs` today computes both and the moments in one call: it splits into `rep.topology(lpos, lsup)` and `rep.moments(topology, kernel, channels)`.
* `ImplicitRep` / `SdfRep`: no pairs; the "adjacency" is `(cand, idx = sel, low)` and the tier-3 closed forms are *evaluated* (kernel-dependent: `tier3Table(name)`), plus the `fallback` SurfaceRep pairs for `low` queries. Topology = the selection; precompute = the tier-3 values. Note `sel` depends on `lam != 0`, which is kernel-dependent, so it is *not* pure topology: either keep `low` (tier flag, depends on `lsup` and the SDF smoothness, not on the kernel) as the topology and let a zero `lam` produce a zero contribution, or accept that implicit reps keep a kernel-keyed precompute. Decide in 2b by checking that the second option changes no result.
* `VolumeRep`: `BoundaryAdjacency` already is topology + weights (`weights`, `gradWeights`); the split is the same follow-up as `boundaryOps` (§4 step 2, last bullet). Out of scope for 2b unless the DFSPH tests need it.

## 1. The terminology decision (user, 2026-10-05)

| term | means | holds | lives |
|---|---|---|---|
| **adjacency** | which particles interact with which boundary elements | integer pair topology only: `(query, element)` per body / representation, candidate lists, indicator candidates. Kernel-independent (depends on positions and supports only) | rebuilt or kept (Verlet skin) per position set |
| **precompute** | the per-pair integral values (`lam, g0, m1, g1` = 9 channels) of one kernel | floats, 72 B per pair | explicit, opt-in object (`PairMoments`); public API only if a benchmark shows it faster (DFSPH is the candidate, §4 step 6) |
| **evaluation** | integral + contraction with the per-query data, accumulated into per-query outputs | nothing per pair | a fused kernel over the adjacency |

Why it matters, measured: today `Scene.buildAdjacency` returns `SceneAdjacency` whose entries are `MomentPairs` (the precompute). It is consumed once by cover (3 builds/step), tensile (1), lap (2), and 1–3 times by `_wall_data` / `_surface_state` (lam, G, Cov, A read *disjoint* channels). Consequences already visible in the code: a kernel guard and a channels guard in `sceneOperation` (they exist only because the "adjacency" holds data), `SceneAdjacency.channels`, the pruning of WORK-008, a cache for the cross-step reuse. The same confusion sits in `boundaryOps.BoundaryAdjacency` (its docstring: "an adjacency list that already carries the geometry"; fields `weights`, `gradWeights`). warpSPH's own `AdjacencyListWarp` / `CompactHashMap` are topology only and the operations evaluate kernels on the fly; the boundary side should match.

Memory (illustrative, not measured): 10⁷ near-wall queries × 50 faces/query in 3D = 5·10⁸ pairs × 72 B = 36 GB for *one kernel*. Precompute cannot be the default at scale; in 2D with 484 pairs it is irrelevant, which is why it went unnoticed.

Where precompute might stay (user decision 2026-10-05): **the metric is a benchmark, not a principle.** The default for every solver is evaluation (fused, recompute). δ⁺-SPH reads each wall quantity only a few times per step, so storing it never pays. For fluid–fluid sums the ratio of memory reads to arithmetic is poor enough that recomputing terms hides behind latency, so precompute is not worth it there either. DFSPH reuses wall values over the 10–100 pressure iterations, so it is the one candidate: build its wall contribution as a fused evaluation too, then benchmark fused-evaluation-per-iteration against precompute-and-reuse (§4 step 6). If precompute is legitimately faster at the sizes that matter it becomes a public API (`PairMoments`, documented with its memory cost); if not, it stays an internal reference path for tests. Precision (user, 2026-10-05): most production runs are FP32; large FP64 runs go to H200-class hardware (high FP64 rate) elsewhere. So the benchmark metric is **FP32 first**, FP64 second, and nothing may be tuned only to this card's 1/64 FP64 rate. Consequence for the design: the fused kernels are dtype-generic, and the FP32 path uses the stable (Chebyshev) plans, not the monomial ones (monomial-basis cancellation amplifies float32 error 1e2–2e4×; the Chebyshev compile with `stable=(8,6)` gives f32 value 8e-8 / grad 3e-7, `docs/exactness-and-approximations.md`). **Done (2026-10-05, user: piggyback on warpSPH's mechanism):** the edge kernels are written against `warpSPHCore.type_config.scalar_t` (`src/edgebound/edge/precision.py`: `real`, `vec2_t`, `torch_real`); the precision is chosen once per process exactly as in warpSPH (environment variable `warpSPHCore_PRECISION` or `warpSPHCore_config.configure(precision=…)`, before the first import), edgebound's default when nothing is configured is float64 (`edgebound/__init__.py` calls `ensure_default()`; `tests/conftest.py` imports it first), float32 selects the Chebyshev `stable=(8, 6)` route by default, the finite-element pair kernels (`pair_weights`, `warp2d.moment*`) are float64 only and raise otherwise; the tensors handed back to the scene layer are float64.

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

## 4. Steps (each ends green: `pytest tests` = 816 + new tests, harness `check --physics --cases tank,dambreak`)

**Step 1 — f32 stable-plan edge kernel, then term-parallel (isolated, no API change).** First an f32 kernel on the Chebyshev plans and the pairs-vs-time curve in f32 and f64 (`review8_latency_probe.py` extended). Only if the f32 curve is also latency-bound, prototype: a thread per (pair, term group) writing to a scratch `[P, groups, 9]`, fixed-order reduction (keep it deterministic: atomics would add run-to-run spread to a harness that is bit-level). Measure at 484 / 4 840 / 48 400 pairs for `w2`, `w2p5`, `cone`. Expected (to be measured): `w2` 1.8 → well under 0.5 ms at 484 pairs. If the speed-up is < 2× the latency hypothesis is wrong: stop, re-read the profile before step 3. Acceptance: pruned and full results equal the old kernel to a stated tolerance (the sums change order if terms are reduced differently; with a fixed per-channel term order they stay bit-identical — prefer that), `test_channel_pruning.py` unchanged.

**Step 2 — split adjacency from precompute (refactor, results bit-identical).**
* `Scene.buildAdjacency` → topology only; `Scene.precompute(adjacency, kernel, channels) -> PairMoments`; `sceneOperation(…, adjacency | pair_moments)` (an adjacency alone triggers a precompute for the call, so every current consumer keeps working; consumers that call several operations pass the `PairMoments`).
* One adjacency per position set shared by wall / lap / cover / tensile: 9 builds → 3 (dam break; the cache supplies x^n; sloshing 6 → 2). The near-wall subset consumers (`_detect_surface` → cover / cone, `lap_lambda_scene`, `tensile_vector_scene` on `x[near]`) need `adjacency.restrict(query_index)`: row-subset of the same positions and supports (measured exact), so restriction is a mask on `q` plus a remap, not a rebuild. Saving: the kernel-independent 0.8 ms × 6 ≈ −5 ms/step (estimate).
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

* Evaluation schedule (2026-10-05, `docs/work/refs/schedule_probe.py`, one dam-break step, nx = 40 / sloshing nx = 60; the schedule does not depend on nx). Position sets: x^n (cache hit, `P0`), x^{n+½} (`P2`), x^{n+1} before shifting (`P4`), after shifting (`P6`); `Nsub` = near-wall subset (272 of 1152) at the same positions as the full set before it.

| positions | query set | consumer | kernel, channels | scene ops |
|---|---|---|---|---|
| x^n | full (cached adjacency) | `_wall_data` (first RHS) | w2, all | Gradient (A) only: lam, G come from the cache |
| x^n | sub | `_detect_surface` → cover | cone, (3,4) | Gradient |
| x^n | sub | `lap_lambda_scene` | lw2, all | Density, Covariance |
| x^{n+½} | full | `_wall_data` | w2, all | Density, Gradient (G), Gradient (A) |
| x^{n+½} | sub | cover; lap | cone (3,4); lw2 all | Gradient; Density, Covariance |
| x^{n+1} pre-shift | full | `_surface_state` | w2, all | Density, Gradient, Covariance |
| x^{n+1} pre-shift | sub | cover; tensile | cone (3,4); w2p5 (3,4) | Gradient; Gradient |
| x^{n+1} post-shift | full | `_wall_data` in `no_penetration` | w2, all | Density, Gradient, Gradient (A) (→ cache for the next step) |

Sloshing (no shifting, no `no_penetration` cache, `w4`): two position sets, each full `_wall_data` + sub cover + sub lap = 6 builds, 12 ops. Kernels per position set: w2 (3 channel sets of one kernel), lw2, cone, w2p5 — i.e. step 3's fused pass is one launch family per position set with 4 kernels. Note `lap` needs Density and Covariance of lw2 only on the subset; `_surface_state` needs lam, G, Cov of w2 on the full set.

* Probes: `docs/work/refs/review8_build_breakdown_probe.py` (per-build table), `docs/work/refs/review8_latency_probe.py` (pairs-vs-time curve), `review7_edge_cost_probe.py`, `review7_prune_probe.py`.
* The `buildAdjacency` = 9/step pin: `tests/sim/test_wall_data_reuse.py::test_reuse_saves_one_adjacency_per_step`.
* Review record and reviewer lessons: `docs/work/logs/REVIEW-008.md`, `docs/work/REVIEW.md`.

## 9. Step 1a results (2026-10-05 night, idle GPU: RTX PRO 6000 Blackwell, 0 % utilisation, dam-break wall state nx 67, 484 (query, edge) pairs)

Kernel GPU time alone (Warp events around 50 back-to-back launches) and end-to-end `edge_channels`, ms:

| pairs | monomial f64 `w2` all | end-to-end | monomial f64 `w2` (3,4) | `cone` (3,4) | Chebyshev f64 `w2` all | **Chebyshev f32 `w2` all** | f32 (3,4) | f32 `w2p5` (3,4) | f32 `cone` (3,4) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 484 | 1.92 | 1.99 | 0.28 | 0.10 | 3.2 | **0.26** | 0.048 | 0.087 | 0.041 |
| 4 840 | 1.98 | 2.05 | 0.28 | 0.10 | 3.3 | 0.28 | 0.053 | 0.096 | 0.045 |
| 48 400 | 3.04 | 3.18 | 0.56 | 0.19 | 6.0 | 0.33 | 0.060 | 0.124 | 0.048 |
| 484 000 | 18.1 | 18.2 | 3.06 | 1.06 | 33.9 | 1.25 | 0.23 | 0.53 | 0.17 |

* Host path 0.025 ms of enqueue per launch and ~0.08 ms between the end-to-end and kernel numbers: the time is in the kernel, not in Python / `wp.from_torch` / sync.
* f64 on this card (1/64 FP64 rate): one thread per pair takes 1.9 ms for 21 terms (~150 k cycles per term); the Chebyshev f64 route is 1.7× slower than the monomial one. Not a design target (the production precision is FP32, FP64 runs go to H200-class hardware).
* float32 error against float64 (same Chebyshev plans, units of h, max over 484 pairs and all asked channels): `w2` 4.1e-6 absolute / 2.9e-5 relative (|c| > 1e-3), `w2p5` 2.5e-5 / 5.2e-5, `cone` 2.0e-6 / 1.9e-5. Pair data are body-frame coordinates in float32 (positions O(1) m, h = 0.03 m: 4e-6 h of position rounding is the floor of these numbers; subtracting vertices and positions in float64 before the cast would remove it if needed).
* **Still latency-bound in float32**: the 484 → 4 840 pair curve is flat (0.26 → 0.28 ms) and the saturated per-pair cost (2.6 ns) would give 1.3 µs for 484 pairs, 200× below the measured 0.26 ms. One thread walks 21 terms × (8 nodes × ≤ 14 panel intervals × Clenshaw) serially; a thread per (pair, term) or per (pair, term, side) is the restructuring of step 1b. Expected from the term count: ≥ 10× on top of the float32 gain at 484 pairs (to be measured).
