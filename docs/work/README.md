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
Single shared branch `local-model` for all packages. Before launching: commit the reviewer files (`HANDOFF.md`, `docs/README.md`, `docs/work/`, reviewer corrections) on `local-model` as `REVIEW-NNN: ...` so the package starts from a clean tree; the model never commits them. `docs/work/refs/` holds the reviewer's throw-away probes (read-only for the model). Current package: WORK-005 (change the number in the launch prompt; WORK-001…004 reviewed and accepted).
