# docs/work — local-model work packages

| file | purpose |
|---|---|
| `KICKOFF.md` | standing protocol for the local model (rules, loop, stop conditions, report format) |
| `WORK-NNN.md` | one bounded work document (3–5 tasks, definition of done, hard stop) |
| `REVIEW.md` | checklist for the reviewer (Claude) and the work-document template |
| `logs/LOG-NNN.md`, `logs/REPORT-NNN.md` | written by the local model; `logs/REVIEW-NNN.md` by the reviewer |

Launch prompt for the local model (paste as the first message; change only the number):
```
Read /home/lu26029/dev/curvatureBoundaries/docs/work/KICKOFF.md completely, then
/home/lu26029/dev/curvatureBoundaries/docs/work/WORK-001.md completely. You are the agent
described there. Carry out WORK-001 task by task, in order, following KICKOFF.md exactly
(branch, log, one commit per task, evidence for every number, stop conditions). Do not ask
me questions; if you are blocked, write the BLOCKED report as KICKOFF.md section 5/6 says.
Work until the Definition of done of WORK-001 is met, then write REPORT-001.md, commit it, and stop.
```
Single shared branch `local-model` for all packages. Before launching: commit the reviewer files (`HANDOFF.md`, `docs/README.md`, `docs/work/`, reviewer corrections) on `local-model` as `REVIEW-NNN: ...` so the package starts from a clean tree; the model never commits them. `docs/work/refs/` holds the reviewer's throw-away probes (read-only for the model). Current package: WORK-006 (change the number in the launch prompt; WORK-001…005 reviewed and accepted).

## Repository layout changed after WORK-008 (2026-10-05) — command and path map for the older documents
The work documents, logs and reports above are history and keep the old names. Mapping: package `python/edgebound/X.py` is now `src/edgebound/{edge,scene,sim}/X.py` (`import edgebound.X` → `edgebound.<edge|scene|sim>.X`); runners are files in `scripts/` (no `cd python`, no `-m`); the `curvbound` package is `src/curvbound`; tests moved to `tests/{edge,scene,sim,curvbound}/`. Install once: `pip install -e . --no-deps` in the `warp` env; run tests with `python -m pytest tests -q -n 4`.

| old | new |
|---|---|
| `python -m edgebound.deltasph_regress check …` | `python scripts/deltasph_regress.py check …` |
| `python -m edgebound.deltasph_profile` | `python scripts/deltasph_profile.py` |
| `python -m edgebound.deltasph_validation …` | `python scripts/deltasph_validation.py …` (runners: `edgebound.sim.validation`) |
| `python -m edgebound.dfsph_validation / dfsph_runcase / dfsph_video / deltasph_snap / deltasph_compare` | `python scripts/<same name>.py` |
| `python -m edgebound.np2d_study / q2_conditioning / precision_probe` | `python scripts/studies/<name>.py` |
| `python -m edgebound.fem_study` | `python -m edgebound.edge.fem_study` |
| `python -m edgebound.warp_bench / scene_bench / boundary_bench` | `python scripts/bench/<name>.py` |
| `python -m edgebound.fixtures` / `fem_fixtures` | `python scripts/make_fixtures.py edge` / `fem` |
| `edgebound.q2_conditioning.terms / ref_coeffs` | `edgebound.edge.kernels.power_terms / power_monomials` |
| `tests/edge/test_{scene,cover*,cone_area*,tensile_scene*,viscosity_scene,implicit_bodies,channel_pruning}.py` | `tests/scene/` |
| `tests/edge/test_{deltasph*,dfsph,wall_data_reuse}.py` | `tests/sim/` |
| `tests/test_{kernels,planar,sphere}.py` | `tests/curvbound/` |
