# REPORT-006
Status: **BLOCKED**          Branch: `local-model`   HEAD: `1c23e89` (T6.1, last task commit; **this report is committed as the final commit** on the branch)   Date: 2026-10-04

## Tasks

| id | status | commit | one-line result |
|---|---|---|---|
| T6.1 exact cover/cone/tensile only; `wallViscosityForm` | **BLOCKED** | `1c23e89` | all code (deltasph2d items 1–5, detcmp `git rm`, regress docstring) + every authorised test edit + the new `test_deltasph_exact_only.py` are complete; the 20 changed/new tests pass; **blocked** by the single unauthorised test `test_deltasph.py::test_dynamics_independent_of_the_wall_representation` (a `VolumeRep` body vs the SurfaceRep-only exact cover guard); the equivalence run (T6.1 (8), the primary evidence) is not run |
| T6.2 one adjacency per call; kernel guard | not started | — | depends on T6.1 |
| T6.3 gates, baseline, profile | not started | — | depends on T6.1 (the equivalence run / baseline / profile all ride on it) |
| T6.4 documentation | not started | — | depends on T6.1–T6.3 results |

## Acceptance evidence

Starting state (all as required; full captures in `.tmp/log006_starting_full.log`):
```
$ git status
On branch local-model
nothing to commit, working tree clean
$ git log -1 --format=%s
c2ce13d REVIEW-005: WORK-006 revised (pairwise viscosity kept as wallViscosityForm, adjacency kernel guard), WORK-007 (no-slip), adjacency notes, probes
$ python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py tests/edge/test_cover_scene.py tests/edge/test_deltasph_cover.py -q
79 passed, 14 warnings in 107.40s (0:01:47)
$ python -m pytest tests/edge -q
749 passed, 15 warnings in 463.90s (0:07:43)
```

T6.1 — changed + new tests (`1 failed, 20 passed in 19.63s`; the single failure is the blocker, quoted below; all 20 changed/new tests pass):
```
$ python -m pytest tests/edge/test_deltasph_cover.py tests/edge/test_deltasph_cone.py tests/edge/test_deltasph_tensile.py tests/edge/test_deltasph_tensile_c4.py tests/edge/test_deltasph_viscosity.py tests/edge/test_deltasph.py tests/edge/test_deltasph_exact_only.py -q
... (all changed/new tests pass) ...
=========================== short test summary info ============================
FAILED tests/edge/test_deltasph.py::test_dynamics_independent_of_the_wall_representation[cuda:0]
1 failed, 20 passed, 14 warnings in 19.63s
```

T6.1 — full suite (`1 failed, 752 passed` in 426.21 s; capture `.tmp/log006_full_after_t61.log`):
```
$ python -m pytest tests/edge -q
=========================== short test summary info ============================
FAILED tests/edge/test_deltasph.py::test_dynamics_independent_of_the_wall_representation[cuda:0]
1 failed, 752 passed, 15 warnings in 426.21s (0:07:06)
```
(The expected T6.1 count was 753 = 749 + 4 new; 752 pass and 1 fails — the blocker.)

T6.1 (8) — the equivalence run (the primary evidence) — **not run** (the task is blocked at the acceptance tests above; it uses SurfaceRep bodies and would otherwise be the next step once the blocker is resolved).

T6.2 / T6.3 / T6.4 — **not run** (depend on T6.1).

## Numbers

No physics numbers are reported: the equivalence run (T6.1 (8)) and the T6.3 gates / baseline / sloshing / profile were not run because the task is blocked at the T6.1 acceptance tests. The only numbers measured this session are test counts (all from commands run in this session):

| quantity | value | tolerance | margin | file/command |
|---|---|---|---|---|
| starting-state solver tests | 79 passed | = 79 | 0 | `pytest tests/edge/test_deltasph.py test_scene.py test_dfsph.py test_cover_scene.py test_deltasph_cover.py -q` |
| starting-state full suite | 749 passed | = 749 | 0 | `pytest tests/edge -q` (`.tmp/log006_starting_full.log`) |
| after T6.1: changed/new tests | 20 passed / 1 failed | all changed/new pass | 1 failure (the blocker) | `pytest <the 7 changed/new files> -q` |
| after T6.1: full suite | 752 passed / 1 failed | = 753 expected | 1 failure (the blocker) | `pytest tests/edge -q` (`.tmp/log006_full_after_t61.log`) |

(The T6.1 (8) target numbers the equivalence run was to reproduce, from WORK-005 T5.3(1b), are: tank rmseBulk 1.4993e-3 / rmseNear 1.8134e-3 / keLast 7.28438e-7; dam break ke_tstar1/2/25 0.357363651811 / 0.754228393305 / 0.922766940735, maxVelocityMax 6.59792039874, minDensityMin 0.999428350557, maxDensityMax 1.00603117793, arrival 2.47397 (abs 1e-4), KE vs `dambreak_B_nx67` 3.5344e-4 (abs 1e-5) — **not verified this session**.)

## Deviations

* **Tensile test sub-check renumbering (minor, comment-only).** After deleting the effect-on-shift checks (authorised: "(c) the effect on shift (C2 file) and (b) the effect on shift (C4 file) are deleted"), the remaining five-steps / both-kernels checks were renumbered to keep the lettering sequential (C2: (a),(b),(c); C4: (a),(b),(c)). No tolerance or check logic changed; the spec did not specify the letters of the surviving checks.
* **Viscosity test docstrings (minor, comment-only).** The spec authorised only the cfg / toggle / (a) changes to `test_deltasph_viscosity.py` ("all tolerances and checks (b),(c),(d) unchanged"). I also updated the module docstring title and the `d_ex = ...` definition (and the (d) comment / print line) because they referenced the removed `viscosityExact` switch; the updated text describes the new `wallViscosityForm` API. No tolerance or check logic changed.
* **New Laplacian test written as specified.** `test_wall_viscous_term_is_the_wall_laplacian` implements the spec's own `scipy.integrate.quad` of `B = -z ∫ W'/r dx` (divergence theorem on the wall line), the smoke B/fac/pred_y, the physics signs, and the pairwise negative control. It passes.

## Failures and open questions

**Failure (the blocker): `tests/edge/test_deltasph.py::test_dynamics_independent_of_the_wall_representation[cuda:0]`.**

```
$ python -m pytest tests/edge/test_deltasph.py::test_dynamics_independent_of_the_wall_representation -q
... (the test runs sim.step() for domain in ("surface", "volume")) ...
python/edgebound/cover.py:135: NotImplementedError
E   NotImplementedError: cover_vector_scene: SurfaceRep bodies only
=========================== short test summary info ============================
FAILED tests/edge/test_deltasph.py::test_dynamics_independent_of_the_wall_representation[cuda:0]
1 failed in ...
```

Root cause: the test asserts that `domain="surface"` and `domain="volume"` give identical dynamics. `domain_scene("volume", ...)` builds a `VolumeRep` body. Before WORK-006 the free-surface detector's wall part had a polar path (using only `Scene.inside`) that worked with any representation, so the test passed. WORK-006 makes the exact wall operations the **only** path and removes the polar branches, so the detector now always calls `cover_vector_scene`, which is **SurfaceRep-only** (the guard, kept by the out-of-scope note "ImplicitRep/VolumeRep bodies (the guards stay)"). The solver therefore can no longer step a `VolumeRep` (or `ImplicitRep`/`SdfRep`) body, and this test cannot pass. It is the only solver test affected: every other non-SurfaceRep use in `tests/edge` is a scene-layer guard test that calls the exact operation directly and *expects* the `NotImplementedError` (those still pass).

Why I did not resolve it myself:
* The test is **not** in the authorised-edit list ("and nothing else": only `test_deltasph_cover/cone/tensile/tensile_c4/viscosity.py` and `test_deltasph.py::test_wall_viscous_term_matches_the_half_plane_integral` + the one added test). KICKOFF rule 2 forbids editing a test to make it pass, and the test is not one I may fix.
* The alternatives all violate the work document: making the exact operations support non-SurfaceRep bodies is explicitly out of scope ("the guards stay"); re-adding a polar fallback contradicts "the exact operation becomes the only path"; deleting/skipping the test is an unauthorised test edit.
* This is therefore a **design decision the work document does not make** (KICKOFF §5 stop condition).

No tolerance was changed and no test was edited beyond the authorised list. The T6.1 code and every authorised test edit are complete and correct (20/20 changed/new tests pass).

**Question for the reviewer:** what should happen to `test_dynamics_independent_of_the_wall_representation`? (a) delete/skip it (it tests solver stepping of a `VolumeRep` body, which the exact-only path drops); (b) restrict it to `surface` only; or (c) is solver support for non-SurfaceRep bodies an intended behaviour that the exact-only design should preserve (in which case the work document needs to specify a fallback for the free-surface detector)? Once that is decided, T6.1 (8) (the equivalence run), T6.2, T6.3 and T6.4 can proceed.

## Files changed

`git diff --stat c2ce13d..1c23e89` (the T6.1 commit; `docs/work/logs/LOG-006.md` and this report are added by the final commit):
```
 python/edgebound/deltasph2d.py         | 109 ++++++++++++--------
 python/edgebound/deltasph_detcmp.py    |  95 ------ (deleted)
 python/edgebound/deltasph_regress.py   |   2 +-
 tests/edge/test_deltasph.py            |  73 ++++++
 tests/edge/test_deltasph_cone.py       |  17 +--
 tests/edge/test_deltasph_cover.py      |  12 +--
 tests/edge/test_deltasph_exact_only.py |  76 ++++++ (new)
 tests/edge/test_deltasph_tensile.py    |  45 ++----
 tests/edge/test_deltasph_tensile_c4.py |  67 ++-----
 tests/edge/test_deltasph_viscosity.py  |  27 ++--
 10 files changed, 252 insertions(+), 271 deletions(-)
```

## What I did NOT verify

* The T6.1 (8) equivalence run (`check --physics --cases tank,dambreak`, default config) — not run (blocked at the acceptance tests). So the claim "the new defaults reproduce the WORK-005 T5.3(1b) all-switches numbers" is **not verified** this session (the code path is the same exact operations those numbers were measured with, but I did not run the harness).
* T6.2 (the one-adjacency refactor + kernel guard) — not implemented, not tested.
* T6.3 (the pairwise-form gate, the baseline re-record, the sloshing run, the profile) — not run.
* T6.4 (the documentation) — not written.
* The behaviour of the solver on `ImplicitRep`/`SdfRep` bodies specifically (only the `VolumeRep` case is exercised by the failing test; the same SurfaceRep-only guard applies to all non-SurfaceRep representations, but I did not run a solver step on an `ImplicitRep`/`SdfRep` body to confirm).
