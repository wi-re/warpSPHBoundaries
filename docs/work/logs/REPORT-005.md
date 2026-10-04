# REPORT-005

Status: **BLOCKED**          Branch: `local-model`   HEAD: `7ff0ebed29c8be404cc595391d378258f4ebef6a`   Date: 2026-10-04

(HEAD is the session's starting commit; this report commit lands on top of it. No task commit was made.)

## Tasks

| id | status | commit | one-line result |
|---|---|---|---|
| T5.1 | not started (blocked at starting-state check) | – | – |
| T5.2 | not started | – | – |
| T5.3 | not started | – | – |
| T5.4 | not started | – | – |

No implementation work was done. The work stopped at the mandatory starting-state check (KICKOFF §2 / WORK-005 "Starting state"), which is not met (see below). No WORK-005 task is listed as independent of the starting state, so none was started.

## Acceptance evidence

Starting-state check (all commands run this session, `warp` env; full captures in `.tmp/w005_starting.log`):

```
$ git branch --show-current
local-model
$ git rev-parse HEAD
7ff0ebed29c8be404cc595391d378258f4ebef6a
$ git log -1 --format=%s
WORK-004 REPORT-004: session report (T4.1-T4.4 done)     # expected: starts with REVIEW-004
$ git status --porcelain
 M docs/work/README.md
 M docs/work/REVIEW.md
?? docs/work/WORK-005.md
?? docs/work/logs/REVIEW-004.md
?? docs/work/refs/lap_calib_probe.py
?? docs/work/refs/lap_controls_probe.py
?? docs/work/refs/lap_lshape_probe.py
?? docs/work/refs/lap_scene_probe.py
?? docs/work/refs/lap_solver_probe.py
?? docs/work/refs/review4_c4_indep.py
?? docs/work/refs/slosh_visc_probe.py
?? docs/work/refs/viscosity_switch_proto.diff
$ git log --all --oneline | grep -i "review-004"
(empty — no REVIEW-004 commit on any branch; refs: local-model, main, origin/main)
$ git ls-files docs/work/logs/REVIEW-004.md docs/work/WORK-005.md
(empty — both untracked)
$ python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py tests/edge/test_cover_scene.py tests/edge/test_deltasph_cover.py -q
79 passed, 14 warnings in 109.94s (0:01:49)               # expected 79 — matches
$ python -m pytest tests/edge -q
740 passed, 14 warnings in 365.11s (0:06:05)              # expected 740 — matches
```

The documented starting state (WORK-005) is: *"`git status` clean; `git log -1 --format=%s` starts with `REVIEW-004`."* Both parts fail (tree is dirty with 12 reviewer files; HEAD is `WORK-004 REPORT-004`, and no `REVIEW-004` commit exists on any branch). The pytest counts match (79 / 740), so the code is healthy — the mismatch is purely the uncommitted reviewer state.

## Numbers

| quantity | value | tolerance | margin | file/command |
|---|---|---|---|---|
| solver test set (5 files) | 79 passed | expect 79 | exact | `pytest … -q` → `79 passed … 109.94s` (`w005_starting.log`) |
| full edge suite | 740 passed | expect 740 | exact | `pytest tests/edge -q` → `740 passed … 365.11s` (`w005_starting.log`) |
| uncommitted reviewer files | 12 (2 modified + 10 untracked) | expect 0 (clean) | +12 | `git status --porcelain` (13 lines incl. this session's `LOG-005.md`) |
| `REVIEW-004` commits on any branch | 0 | expect 1 | −1 | `git log --all --oneline \| grep -i "review-004"` (empty) |

(No WORK-005 acceptance numbers exist: no task was run. The reviewer's prototype numbers quoted in `WORK-005.md` Context were not re-measured this session.)

## Deviations

Stopped at the starting-state check instead of proceeding to T5.1. Reason: the documented starting precondition (clean tree at a `REVIEW-004` commit) is not met, and proceeding cannot reach the Definition-of-done `git status clean` (see below). This is a stop-condition action per KICKOFF §5, not a scope change or a loosened check. No tolerances, tests, baselines, or reference series were touched; nothing was edited outside `docs/work/logs/`. No task was started even partially.

## Failures and open questions

**Failure (starting state).** The reviewer's `REVIEW-004` (ACCEPT of WORK-004, `115c16d..7ff0ebe`) exists only as the **untracked** file `docs/work/logs/REVIEW-004.md`; it was never committed, unlike `REVIEW-001` (`54650dd`), `REVIEW-002` (`9b640a7`) and `REVIEW-003` (`115c16d`). `REVIEW-004.md` §4 says explicitly: *"New (uncommitted, **the user commits as `REVIEW-004: …`**): this file, `docs/work/WORK-005.md`, `docs/work/refs/{…}.py`, `docs/work/refs/viscosity_switch_proto.diff`, `docs/work/README.md` + `REVIEW.md` (sequence, lessons)."* The pre-launch commit step documented in `docs/work/README.md` (*"Before launching: commit the reviewer files … as `REVIEW-NNN: ...` so the package starts from a clean tree; **the model never commits them**"*) was not performed before this launch.

The 12 uncommitted reviewer files: `docs/work/README.md`, `docs/work/REVIEW.md` (modified) and `docs/work/WORK-005.md`, `docs/work/logs/REVIEW-004.md`, `docs/work/refs/{lap_calib,lap_controls,lap_lshape,lap_scene,lap_solver}_probe.py`, `docs/work/refs/review4_c4_indep.py`, `docs/work/refs/slosh_visc_probe.py`, `docs/work/refs/viscosity_switch_proto.diff` (untracked). Note: `docs/work/refs/lap_controls_probe.py` is not listed in `REVIEW-004.md` §4 but is in the worktree; it should join the commit.

Precedent: the identical situation at the WORK-004 launch (uncommitted `REVIEW-003`, dirty tree, HEAD at `WORK-003 REPORT-003`) was handled exactly this way — BLOCKED report committed as `00e0e8a`, then the reviewer files were committed as `115c16d REVIEW-003: …` and a fresh session completed WORK-004. `REVIEW-004.md` §3.2 records the resulting process rule: *"for the next package the reviewer files are committed before launch (README rule)"* — this launch is that "next package", launched before the commit.

Why this blocks rather than merely warns:
1. Definition of done requires `git status clean` at the end, but the 12 uncommitted reviewer files are outside the WORK-005 allowed-files list. Committing them would violate KICKOFF rule 4 ("add the files you created or changed by name"), rule 5 ("do not touch files outside the list of allowed files") and the README's "the model never commits them". The model cannot reach a clean tree, so the Definition of done is unattainable by me regardless of task outcomes.
2. The documented starting base (a commit whose subject starts with `REVIEW-004`) does not exist on any branch, so the report's "`git diff --stat` against the starting commit" has no defined base, and WORK-005 commits would land directly on `WORK-004 REPORT-004`, skipping the review-commit boundary the process puts between packages.
3. Proceeding would consume the long GPU gates (T5.3: ≈ 8–10 min × 2 physics runs + ≈ 20–40 min sloshing on a shared GPU) to end at a PARTIAL with a rule conflict — a worse outcome than a clean BLOCKED, for no information gain (the code is already proven healthy by 79/740 green).

**Question for the reviewer/operator:** commit the 12 reviewer files on `local-model` as e.g. `REVIEW-004: accept WORK-004; WORK-005 and reviewer probes` (including `docs/work/refs/lap_controls_probe.py`, unlisted in §4), then re-launch WORK-005 with the same prompt. Nothing else is needed: both pytest starting-state counts are green (79 / 740), no code or test is in question, and no task of WORK-005 has been touched.

## Files changed

Against the starting commit `7ff0ebe`, this session adds exactly two files, both in `docs/work/logs/` (the report commit's `git show --stat`):

```
 docs/work/logs/LOG-005.md    | <insertions>
 docs/work/logs/REPORT-005.md | <insertions>
```

Nothing else: no source file, test, baseline, or `results/` file was read-modified or created; the 12 reviewer files remain exactly as found (uncommitted); scratch is `.tmp/w005_starting.log` (git-ignored). `git status` after the report commit: the 12 reviewer files still dirty — that residual dirt **is** the reported blocker, not a session artifact.

## What I did NOT verify

* Nothing in T5.1–T5.4: `python/edgebound/viscosity.py` and the two new test files do not exist; no gate, sloshing, or ms/step measurement was run; the reviewer's prototype numbers in `WORK-005.md` Context (flat-wall `B(z)` table, L-body brute-force comparisons, gate/sloshing results) were not independently re-measured this session.
* I did not verify that the uncommitted reviewer files are byte-identical to what the reviewer intended (I only read `REVIEW-004.md` and the `README.md`/`REVIEW.md` diffs); that is the operator's commit step.
* `HANDOFF.md` Part A, the reviewer's `lap_*` probes, and `viscosity_switch_proto.diff` were not executed (they are read-only references for the tasks that did not start).
