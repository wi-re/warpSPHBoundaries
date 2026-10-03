# REPORT-004

Status: **BLOCKED**          Branch: `local-model`   HEAD: `119583b36f9207531819e433615fc9a6e9817ff4`   Date: 2026-10-03

## Tasks

| id | status | commit | one-line result |
|---|---|---|---|
| T4.1 | not started (blocked at starting-state check) | – | – |
| T4.2 | not started | – | – |
| T4.3 | not started | – | – |
| T4.4 | not started | – | – |

No implementation work was done. The work stopped at the mandatory starting-state check (KICKOFF §2 / WORK-004 "Starting state"), which is not met (see below). No WORK-004 task is listed as independent of the starting state, so none was started.

## Acceptance evidence

Starting-state check (all commands run this session, `warp` env; full captures in `.tmp/w004_starting.log`):

```
$ git branch --show-current
local-model
$ git rev-parse HEAD
119583b36f9207531819e433615fc9a6e9817ff4
$ git log -1 --format=%s
WORK-003 REPORT-003: session report (T3.1-T3.5 done)
$ git status --porcelain
 M docs/work/README.md
 M docs/work/REVIEW.md
 M tests/edge/test_tensile_scene.py
?? docs/work/WORK-004.md
?? docs/work/logs/REVIEW-003.md
?? docs/work/refs/slosh_gate_probe.py
?? docs/work/refs/stable_plan_probe.py
?? docs/work/refs/tensile_c4_probe.py
$ git log --all --oneline | grep -i review
9b640a7 REVIEW-002: accept WORK-002; correct T_y factor in q2-conditioning.md; WORK-003 and reviewer probes
54650dd REVIEW-001: accept WORK-001; corrections (cover.py docstring, profile notes); WORK-002 and reviewer probes; HANDOFF Part A
$ python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py tests/edge/test_cover_scene.py tests/edge/test_deltasph_cover.py -q
79 passed, 14 warnings in 137.24s (0:02:17)
$ python -m pytest tests/edge -q
728 passed, 14 warnings in 360.87s (0:06:00)
```

The documented starting state (WORK-004) is: *"`git status` clean; `git log -1 --format=%s` starts with `REVIEW-003`."* Both parts fail (tree is dirty; HEAD is `WORK-003 REPORT-003`, and no `REVIEW-003` commit exists). The pytest counts match (79 / 728), so the code is healthy — the mismatch is purely the uncommitted reviewer state.

## Numbers

| quantity | value | tolerance | margin | file/command |
|---|---|---|---|---|
| solver test set (5 files) | 79 passed | expect 79 | exact | `pytest … -q` → `79 passed … 137.24s` (`w004_starting.log`) |
| full edge suite | 728 passed | expect 728 | exact | `pytest tests/edge -q` → `728 passed … 360.87s` (`w004_starting.log`) |
| uncommitted reviewer files | 8 (3 modified + 5 untracked) | expect 0 (clean) | +8 | `git status --porcelain` |
| `REVIEW-003` commits on any branch | 0 | expect 1 | −1 | `git log --all --oneline | grep -i review` |

(No WORK-004 acceptance numbers exist: no task was run.)

## Deviations

Stopped at the starting-state check instead of proceeding to T4.1. Reason: the documented starting precondition (clean tree at `REVIEW-003`) is not met, and proceeding would force a KICKOFF rule 4/5 violation to reach the Definition-of-done `git status clean` (see below). This is a stop-condition action per KICKOFF §5, not a scope change or a loosened check. No tolerances, tests, baselines, or reference series were touched; nothing was edited outside `docs/work/logs/`.

## Failures and open questions

**Failure (starting state).** The reviewer's `REVIEW-003` (ACCEPT of WORK-003) exists only as the **untracked** file `docs/work/logs/REVIEW-003.md`; it was never committed, unlike `REVIEW-001` (`54650dd`) and `REVIEW-002` (`9b640a7`). The 8 uncommitted files are: `REVIEW-003.md`, `WORK-004.md`, `refs/slosh_gate_probe.py`, `refs/stable_plan_probe.py`, `refs/tensile_c4_probe.py` (untracked) and `README.md`, `REVIEW.md`, `tests/edge/test_tensile_scene.py` (modified). This is a pre-launch step the process requires: `docs/work/README.md` says *"Before launching: commit the reviewer files … as `REVIEW-NNN: ...` so the package starts from a clean tree; **the model never commits them**."

Why this blocks rather than merely warns:
1. **Definition of done is unreachable.** It requires `git status clean` at the end. The 8 files above are outside my allowed-files list; committing them violates KICKOFF rule 4 ("add the files you created or changed by name") and rule 5 ("do not touch files outside the list of allowed files") and the README's "the model never commits them". I cannot reach a clean tree without one of those violations, so I did not proceed.
2. **T4.2 is entangled.** T4.2 is authorised to edit `tests/edge/test_tensile_scene.py`, which already carries the reviewer's **uncommitted** test-(c) correction (the work doc says that correction "is already in `HEAD`; leave it" — it is not in `HEAD`). Committing that file in T4.2 would bundle the reviewer's hunk into my commit and contaminate my `git diff` for the file.
3. **Undefined base for the report.** "`git diff --stat` against the starting commit" needs the (nonexistent) `REVIEW-003` commit as the base; the actual HEAD is `119583b`.

**Question for the reviewer (answer one):**
- **(a) Recommended:** commit the 8 reviewer files as a single `REVIEW-003: ...` commit on `local-model` (restoring the documented starting state: clean tree, HEAD = `REVIEW-003`), then re-launch me for WORK-004. This is the normal path used for WORK-001/002/003.
- **(b) If you intend for me to work on top of the current uncommitted state:** name exactly which of the 8 uncommitted files I may commit to reach a clean `git status`, and confirm that my T4.2 commit should include your uncommitted `test_tensile_scene.py` correction. (This deviates from the README's "the model never commits them".)

## Files changed

`git diff --stat` against the starting commit `119583b` (the only files I created/committed this session):

```
docs/work/logs/LOG-004.md    | ~50 insertions (new)
docs/work/logs/REPORT-004.md | ~100 insertions (new)
```

The 8 uncommitted reviewer files listed above are **not** mine and are left untouched in the working tree (they remain visible in `git status` at the end, by design — I am not permitted to commit or remove them).

## What I did NOT verify

- All of T4.1–T4.4: the Chebyshev-quadrature plan/kernel in `warpbc.py`, `STABLE_KERNELS` / `stable=` semantics, `tensile_vector_scene` for `w4`, the `tensileExact` C4 solver path, and the T4.4 gates (tank/dam-break physics runs and the sloshing run) were **not** run or implemented, because the starting-state check failed before any task.
- The reviewer's uncommitted `REVIEW-003.md` content and the 3 `refs/*_probe.py` were only read, not validated.
- The working tree is **not** clean at the end of this session (the 8 reviewer files remain); this is the direct consequence of the starting-state blocker, not an incomplete task of mine.
