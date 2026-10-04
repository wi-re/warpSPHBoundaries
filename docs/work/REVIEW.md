# Reviewing a local-model session (for Claude, or anyone)

Input: branch `local-model` (commits since the tag `reviewed-(NNN-1)`, or since `main` for the first review), `docs/work/logs/REPORT-NNN.md`, `LOG-NNN.md`, the work document `WORK-NNN.md`. Goal: decide cheaply whether to trust the output, correct it, and write `WORK-(NNN+1).md`. Do the steps in order and stop early on a hard failure.

## 1. Mechanical gate (no judgement, ~5 min)
```
git log --oneline <base>..local-model            # one commit per task + the report, subjects "WORK-NNN Tn.m: ..."
git diff --stat <base>..local-model              # only the files WORK-NNN allows? anything else = violation
git diff <base>..local-model -- tests/ results/ python/edgebound/deltasph2d.py python/edgebound/scene.py python/edgebound/warpbc.py   # existing tests / baselines / solver untouched unless allowed
git status --short                                 # clean
python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py -q    # 72 passed (GPU, ~2 min)
python -m pytest <the new test files> -q
```
Any violation of the KICKOFF hard rules (edited tests/baselines/tolerances, files outside the allowed list, commits on main, fabricated numbers) → reject the session's affected commits, say why in the next work document.

## 2. Evidence audit (the part that catches model errors)
* Re-run **two** acceptance commands from the report yourself (pick the most important and one random one) and compare the output with the quoted lines. A mismatch beyond the stated determinism = fabrication or environment drift: investigate before going on.
* For every number in the report's table: is the command that produced it in the log? Is the tolerance older than the result (check the commit order: tolerance in the test file committed with the test, not changed later: `git log -p -- <test file>`)?
* Independent check of each new formula/function: the reviewer re-derives one value by hand or with a throw-away script that shares no code with the new module (as in WORK-001, T1.1's smoke reference). Check signs and units statements.
* Read the *negative controls* (a check that cannot fail proves nothing) and the *Deviations / Failures* sections first; weak spots are usually listed there, or missing from there.
* Look for the usual local-model failure modes: tests that compare a function with a copy of itself; tolerances widened after a failure; tests that are skipped or `xfail`ed; try/except that swallows errors; hard-coded expected values taken from the code's own output; silently reduced problem sizes; docs that claim more than the evidence.

## 3. Verdict and next step
Write `docs/work/logs/REVIEW-NNN.md`: verdict per task (ACCEPT / ACCEPT WITH CORRECTIONS / REJECT), the corrections made (by the reviewer, in commits `REVIEW-NNN: ...`), and the facts that the next work document must assume. Merge `local-model` into `main` only after the verdict (user decides when) and tag the reviewed head `reviewed-NNN` (that tag is `<base>` of the next review). Then write `WORK-(NNN+1).md` from the template below; each work document is bounded to ~3–5 tasks with a hard stop, because a local model drifts on long task lists.

## Work-document template
Title, branch, log/report names · context (≤ 10 lines) · starting state with the exact commands and expected outputs · allowed files (new / modified) · tasks, each with: exact spec, independent verification route, tolerance **with reason**, negative control, acceptance commands, commit subject · definition of done · out of scope. Put every design decision into the document; the model must not need to make one (KICKOFF §5 stops it if it does).

## Planned sequence (HANDOFF.md Part A, adjusted after each review)
* WORK-001 (regression harness, step profile, exact cover vector formula): **done, accepted** (`logs/REVIEW-001.md`).
* WORK-002 (Q3a into the solver behind `cfg.coverExact`, physics gate + `--cfg` in the harness, detector comparison, Q2 conditioning study): **done, accepted** (`logs/REVIEW-002.md`; one wrong number in `docs/q2-conditioning.md` corrected).
* WORK-003: Q3b closed-form cone area (`cone_area.py`, scene wrapper, `cfg.coneExact`) and the exact tensile term for Wendland C2 (`tensile.py`, `cfg.tensileExact`), both behind switches, tests + detector comparison + physics gate runs: **done, accepted** (`logs/REVIEW-003.md`; one vacuous negative control corrected).
* WORK-004: Chebyshev-quadrature edge plan in `warpbc.py` (opt-in through `STABLE_KERNELS` / `stable=`), `tensile_vector_scene` for C2 and C4 on it, `tensileExact` for C4, gates incl. a truncated sloshing run: **done, accepted** (`logs/REVIEW-004.md`; no correction; 740 tests).
* WORK-005: Q1, the exact wall Laplacian `Δλ` (`viscosity.py`: `2λ[L] − tr Cov[L]` of the kernel `L = W′/r` through the unmodified `Density` / `Covariance` scene operations; registering `∇²W` directly fails on the indicator pseudo-pair), `cfg.viscosityExact` (free-slip mirror, `ν_eff = α c0 H/(8ξ)` from a moment identity — a change of operator, not a quadrature replacement), gates incl. a truncated sloshing run, derivation doc: **done, accepted** (`logs/REVIEW-005.md`; three doc/docstring corrections: a broken table row, "one operator" wording, related-work §8 on Chiron 2019; 749 tests).
* WORK-006: the exact wall operations become the **only** path: delete the polar quadrature, `_solid_samples` and the four switches (`near` comes from `λ`), the pairwise wall viscosity goes (the accepted tests that pin removed behaviour are named and their edits authorised; the polar reference moves to `tests/edge/polar_reference.py`), `lap_lambda_scene` with one adjacency, re-record the bit-level baseline, sloshing with the new defaults, profile after. The primary evidence is that the new defaults reproduce the WORK-005 all-switches gate numbers. **Written, ready to launch** — the exact path is currently *slower* than the polar one (43 → 79 ms/step dam break) until `_solid_samples` is gone; the rest of the cost is per-kernel pair-channel evaluation inside `buildAdjacency` (phase 2).
* then phase 2 (adjacency: 4 builds ≈ 4.2 ms each is the largest item once `Scene.inside` is gone), driven by `docs/deltasph-profile.md`.

## Lessons for writing work documents (from REVIEW-001)
* Verify every "must fail" claim and every smoke number with a throw-away script **before** issuing the document (WORK-001's negative control was mis-specified; the model caught it, but a model that does not would have forced a pass or stalled). Keep such probes in `docs/work/refs/`.
* Say what a gate is *sensitive to*: the bit-level harness fails for any physics change, the physics gate (5 % KE, 0.05 t*) does not see a shifting switch; give a negative control that is valid for the gate in question.
* When a task says "compare against the old quadrature", state the quadrature's own error beforehand (measured), otherwise the tolerance is either vacuous or false.

## Lessons for writing work documents (from REVIEW-002)
* A conversion factor that appears in both the implementation and its test proves nothing: add an **absolute** check (a plain-numpy grid integral at a geometry with a known answer). WORK-002's doc had a wrong factor (1e7 off) that its own tests could not have caught.
* Probe the degenerate geometry (tangent `z == H`, particle on an edge / at a vertex) before issuing a spec for a piecewise formula; the reviewer's own probe had a tangent-case bug that a midpoint sanity check exposed.
* A physics gate with a loose band is a *do-no-harm* check; say in the work document what carries the correctness evidence instead (tests, detector comparison), and which gate items are insensitive (`steps`).

## Lessons for writing work documents (from REVIEW-003)
* When the risky part of a package is new device code, **prototype it** (reviewer probe in `refs/`) and measure every number the document quotes (accuracy, controls at coarser resolutions, which of two routes is right where they disagree) — WORK-004's design decisions (resolution `(16, 8)`, tolerance 5e-9 against the monomial plan) all come from probe measurements, one of which (a 1.3e-10 disagreement) turned out to be the *old* route's error.
* A negative control that compares a result with a scaled copy of itself (`1.01·T` vs `T`) is arithmetic, not a check: the control must be compared with the *independent* value. Say so in the document (WORK-003 T3.4 (c) had it; the model deviated silently from "the grid value").
* A work document that deliberately removes behaviour pinned by an accepted test must name the test and authorise the edit; otherwise KICKOFF rule 2 forces a stop.
* Check the cost of a validation case (`ms/step × steps`) before putting it into a package; give a truncation (here T = 1.5 s) and a hard timeout, and say that a timeout is *not* a blocker.

## Lessons for writing work documents (from REVIEW-004)
* **Probe the scene machinery's hidden assumptions, not only the formula.** The obvious Q1 route (register `∇²W` as a kernel) is mathematically exact and *silently wrong* inside a body: the scene's indicator pseudo-pair assumes `∫K = 1`. It only showed up because the probe compared against an independent brute force at points on both sides of the body and at the tangent case. A solver gate would never have caught it (fluid particles are never inside a solid). Put the "inside the body / support fully inside" case into every spec of a new scene operation.
* **A calibration the HANDOFF asks for may have a closed form.** `ν_eff` for the wall Laplacian is a moment identity of the bulk pairwise term (`fac/(2(d+2))`), pinned by an absolute quadrature test; the wall-damping comparison (a table in `z/H`) is documentation of a *change of operator*, not something to tune. When a "calibration" turns out to be a change of operator, say so in the porting notes and in the work document, and report the physical effect as a finding.
* **Keep a prototype switch as a diff in `refs/`** (`viscosity_switch_proto.diff`) so the gate numbers quoted in the work document are reproducible, and revert the tree afterwards; run the long gates in the background while writing the document (add placeholders and fill them in).

## Lessons for writing work documents (from REVIEW-005)
* **Measure the speed claim of a refactor before promising it.** The plan said "delete the polar grid → faster"; the reviewer's probe showed the exact path is 63–82 % *slower* until the grid is gone, and where the rest goes (per-operation table). Probe: run the candidate configuration for ~100 steps and attribute the cost per call (`refs/review5_cost_probe.py`) before writing a package whose point is performance; say in the document what is information and what is a gate.
* **A cache/adjacency may hold kernel-specific data.** `buildAdjacency` stores the pair channels of its own kernel; reusing one across kernels silently gives wrong numbers (off by 30–60 % of the scale) while the same-kernel reuse is bit-identical. Probe the reuse with an independent comparison before specifying it.
* **A test of a constant that the code and the test share is not a test of the constant.** `ν_eff = fac/8` was "pinned" by arithmetic on the package's own quadrature; the real check is the solver's own bulk term (probe: 0.239/0.250 and 0.757/0.750 — which also showed the bulk operator carries `2∇(∇·v)`, so "wall and bulk are one operator" was wrong). Ask: does the check touch the solver's number, or only the document's derivation of it?
* **Set up the replacement of a removed behaviour's test before issuing the package** (the pairwise wall-viscosity test): compute the new expected value by a route that shares nothing with the solver (the divergence theorem on the wall line), run the solver against it (8.8e-16), and give the old value as the negative control (factor 9.87).
* **GPU summation is not bit-reproducible:** counts of threshold-selected particles and ratios of small numbers vary run to run; acceptance lines use tolerances, never counts or last digits.
* When a package removes *options*, the equivalence run (new default == old all-on numbers, to the harness's measured spread) is the primary evidence; list the numbers in the document and say what a deviation means (stop, do not adjust).
