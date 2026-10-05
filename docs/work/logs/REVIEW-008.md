# REVIEW-008 — WORK-008 (branch `local-model`, `780d150..7f1d1af`)

Reviewer: Claude, 2026-10-05. Verdict: **ACCEPT T8.1, T8.2, T8.3, T8.4**. No rule violation, no correction to code, tests or docs.

## 1. Mechanical gate
* Commits `780d150..7f1d1af`: `T8.1` (`cfa483e`), `T8.2` (`89cbd8d`), `T8.3` (`f8eb1a4`), `T8.4` (`e9f1b2a`), `REPORT-008` (`7f1d1af`); subjects as specified, `git status` clean.
* Files: 13, all on the allowed list. Nothing in `tests/` was modified (only the two new files, +404 lines, no removed line in any later commit, no skip/xfail), nothing in `results/`, the harness, the baseline, `viscosity.py`, `cone_area.py`.
* **The code diff equals the reviewer's prototype** (`refs/review7_work008_proto.diff`) line for line (`diff` of both, `index` lines stripped: identical, 261 lines). So the design decisions (cache key, guard rule, channels) are the reviewed ones.

## 2. Evidence audit
| check | result |
|---|---|
| full suite (reviewer, `.tmp/rev8_tests.log`) | **767 passed** (= 760 + 7), 309 s |
| profile (reviewer, `.tmp/rev8_profile.log`, GPU idle of other jobs) | dam break **35.51** ms/step, sloshing **44.66**; `buildAdjacency` 9.0, `sceneOperation` 10.0 per step (report: 36.48 / 46.49; the report's before/after pair is the back-to-back measurement and is not repeated) |
| report vs `LOG-008` vs docs | harness lines, margins, profile tables and the sloshing line in the three docs equal `.tmp/w008_check.log` / `LOG-008`; the "limit" column of the validation table equals the `[gate]` lines of the log |
| tests | read in full. (a) compares a pruned plan with the *full* plan of the same kernel (independent: different term sets), non-vacuity control `max|full g0| > 0.5`; (c) five guard negative controls each raise; (d) the full-adjacency reference goes through the unpruned path; (e) pins host-call counts 9 / 10; (g) five invalidation cases, hits `torch.equal`, rebuilds with a stated tolerance, each case's stale-cache error is shown to be visible (82× / 89× / 59× over the bound) |
| tolerances | single commit per test file's checks; none edited later |
| the report's two process notes | the "before" profile was re-taken at clean HEAD (right); three fixes were in the model's own new tests only |

## 3. Corrections
None.

## 4. Facts the next work document must assume
* Full suite **767**; five-file solver set 82. Next review base = `7f1d1af` (+ the REVIEW-008 commit).
* **`tests/edge/test_wall_data_reuse.py::test_reuse_saves_one_adjacency_per_step` pins `buildAdjacency` = 9 per step (10 without the cache).** A package that changes the number of builds (sharing pair sets, fusing) must name this test and authorise the edit (WORK-003 lesson).
* Where the 20.4 ms/step of builds goes (reviewer probe `refs/review8_build_breakdown_probe.py`, dam break after 300 steps, per call incl. sync): `_wall_data` kernel `w2` all channels 2 builds × 3.20 ms = 6.39 ms/step; `lap_lambda_scene` kernel `lw2` all channels 2 × 2.79 = 5.58; `_surface_state` `w2` all channels 1 × 3.20; `cover_vector_scene` `cone` (3,4) 3 × 0.97 = 2.90; `tensile_vector_scene` `w2p5` (3,4) 1 × 2.39 (the Chebyshev route). The pruned cover build costs 0.97 ms of which `edge_channels` is 0.18 (WORK-008 probe): **≈ 0.8 ms per build is kernel-independent overhead** (pair list, cell list, `nonzero` syncs, `toWorld`, indicator when not pruned).
* The nine builds sit on four position sets per step (x^n — now cached —, x^{n+½}, x^{n+1} before shifting, x^{n+1} after shifting), so a kernel-independent pair set could be shared by up to 3–4 builds per set.
* Remaining step split (dam break, ms): builds 20.4 (57 %), `_detect_surface` self 5.1, fluid pair sums 9.0 (25 %), `neighbor_pairs` 2.1. warpSPH's dam break is 147 s over the same step count ≈ 7.5 ms/step (implied, not measured here): **the fluid pair sums alone exceed warpSPH's whole step**; parity cannot come from the scene path alone (CUDA graph / warpSPH modules, phase 3).

## 5. Weak spots (none blocking)
* The cache key is positions, poses, supports and kinds: it assumes rigid bodies whose representation geometry is not edited in place. A deforming body (FEM fields, later) needs the key extended; the multi-body case is not tested (the loop over bodies is read, not run).
* The cache hands out the same `lam`/`G` tensors; the read-only contract is enforced by the suite and the harness only. A solver term that later modifies one in place would corrupt the next step silently.
* tank `rmseBulk` margin 0.011 (reviewer ≤ 0.007 earlier): last-digit GPU spread, 90× below the stop threshold.
