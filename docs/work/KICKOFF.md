# Kickoff: how a local-model session works in this repo

You are an autonomous coding agent working on `curvatureBoundaries` (exact SPH boundary integrals; a 2D δ⁺-SPH solver on analytic walls that is being prepared for a port into warpSPH). A reviewer (Claude) will check everything you do afterwards, so **evidence matters more than speed**.

**Your instructions are this file plus one work document** (`docs/work/WORK-NNN.md`, named in the message that started you). Read both completely before doing anything. Then work through the tasks of the work document **in order, without asking questions**, until its *Definition of done* is met or you are blocked (section 5). Then stop and write the report (section 6). Do not start work that is not in the work document, even if it looks obviously useful.

## 1. Hard rules (breaking any of these invalidates the whole session)
1. **Never fabricate a number.** Every number in your log or report comes from a command you ran in this session; quote the command and the relevant output lines. If you did not run it, write "not run".
2. **Never edit a test, a tolerance, a baseline file or a reference series to make something pass.** If a check fails, report that it fails, with the output. You may add new tests; you may fix a test only if you can show it is wrong independently of your code (explain how in the log) — otherwise report it.
3. **Never modify anything outside this repository.** In particular `~/dev/warpSPH`, `~/dev/warpSPHCore`, `~/dev/warpSPHIntegrators`, `~/dev/omniSPH`, `~/dev/openMaelstrom` are read-only for you (reading is fine, importing is fine).
4. **Git:** work on the single long-lived branch `local-model` (create it from the current HEAD if it does not exist, otherwise check it out and continue; the reviewer merges it into `main` after each review). One commit per finished task, message `WORK-NNN Tn.m: <what>` and nothing else in the subject. Never commit to `main` (the branch is shared by all work packages, so keep the commit subjects prefixed with the work number), never push, never rewrite history, never `git add -A` (add the files you created or changed by name; `git status` before every commit; scratch files go to `.tmp/`, which is git-ignored).
5. **Do not touch** files outside the list of *allowed files* of the work document (new files in the listed directories are allowed).
6. **One GPU job at a time.** Check `nvidia-smi` before starting a long run. Never start a second long run while another one of yours is alive.
7. **No silent scope changes.** If a task cannot be done as written, stop that task, log why, and go to section 5.

## 2. Environment
```
cd /home/lu26029/dev/curvatureBoundaries
export PATH=/home/lu26029/miniconda3/envs/warp/bin:$PATH     # python with torch, warp, mpmath, pytest, warpSPH*
python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py -q     # ~ 2 min (72 passed on 2026-10-03), GPU; must stay green
cd python && python -m edgebound.<module> ...                  # solver runners (see docs/deltasph-resume.md "Reproduce")
```
Scratch space: `.tmp/` (ignored). Long commands (> 2 minutes): run in the background with output to a log file (`nohup ... > .tmp/<name>.log 2>&1 &`), then poll the log; never block on one call for more than ~10 minutes; give every command a sensible `timeout`. The solver is deterministic only up to GPU reduction order (`index_add_`), so identical commands can differ in the last digits; the work documents say how to treat that.
Background reading (only what a task points to): `HANDOFF.md` Part A (plan and glossary of the quadrature items Q1–Q3), `docs/deltasph-porting-notes.md` (term map, change log), `docs/deltasph-validation.md` (results), `docs/scene-architecture.md`.

## 3. Loop (per task)
1. Write the plan for the task in the log (3–10 lines: files to touch, how you will check it).
2. Implement the smallest thing that satisfies the task's *acceptance* section. Match the surrounding code style (dense, docstring-light modules, same naming; read two neighbouring files first).
3. Run the task's acceptance commands **and** the regression command listed in the work document. Paste the exact commands and the decisive output lines into the log.
4. If green: update `docs/deltasph-porting-notes.md` change log only if the work document says so; commit (rule 4); append `T-n.m DONE <commit hash>` to the log; continue.
5. If red: debug at most **3 distinct attempts** per failure (each attempt = a changed hypothesis, logged in 2–3 lines). Then section 5.

## 4. Verification discipline
* Prefer an **independent** check over a self-consistent one: compare against `python/edgebound/oracle.py` (mpmath polar quadrature, shares no code with the closed forms), against a brute-force numpy loop, or against stored series in `results/deltasph/`. A test that compares a function with a copy of itself proves nothing.
* State the **tolerance before** you look at the result, and give the reason for it (round-off scale, determinism measurement, ...). Never widen it afterwards.
* When a number is "close", say by how much relative to the tolerance. Report the worst case over the cases you tried, not the mean.
* Check units and signs explicitly in the log for every new formula (one sentence: what a positive value means).
* If two of your own results disagree, that is a finding, not a nuisance: log it and report it.

## 5. Blocked / stop conditions
Stop the current task and write the report (status `BLOCKED`) when any of these happens: a test or regression check you cannot make pass in 3 attempts; the task needs a design decision the work document does not make; a required file/function does not exist as described; a command would violate rule 1–7; you have used more than **150 tool calls on one task**; a GPU job fails twice for non-code reasons (out of memory, driver). Do **not** work around a blocker by changing scope, loosening a check or skipping a task. Finish the report with what you tried, what you observed, and the question the reviewer should answer. Tasks that do not depend on the blocked one may still be done if the work document lists them as independent.

## 6. Files you write
* `docs/work/logs/LOG-NNN.md` — append-only, one entry per task step (plan, commands + output excerpts, decisions, `DONE`/`BLOCKED`). Create it at the start. Keep entries factual and short.
* `docs/work/logs/REPORT-NNN.md` — written **last**, in exactly this structure (the reviewer reads only this first):
  ```
  # REPORT-NNN
  Status: DONE | PARTIAL | BLOCKED          Branch: <name>   HEAD: <hash>   Date: <date>
  ## Tasks            (one row per task: id | status | commit | one-line result)
  ## Acceptance evidence   (per task: the exact command and the decisive output lines, copied)
  ## Numbers          (table: quantity | value | tolerance | margin | file/command)
  ## Deviations       (anything you did differently from the work document, and why; "none" if none)
  ## Failures and open questions   (everything that did not work, with output; questions for the reviewer)
  ## Files changed   (git diff --stat against the starting commit)
  ## What I did NOT verify
  ```
  A report that says "all fine" without the evidence sections is a failed report.
* When finished: `git status` must be clean except for ignored scratch files; the last commit contains the report. Then stop. Do not begin the next task list, do not refactor, do not tidy.
