---
name: scheduler-status
description: Build the abi CLI and smoke-test the one-shot scheduler surface — run `abi scheduler status`, assert the probe task ran to completion (completed=1 total_tasks=1 failed=0), and assert the always-on Prometheus telemetry block. Use to run/smoke-test the abi scheduler, check scheduler stats/status output, or verify the one-shot task-submission accounting after touching src/core/scheduler.zig or telemetry. Fully local; self-terminating one-shot.
---

# scheduler-status — smoke the one-shot scheduler surface

Driver: **`.agents/skills/scheduler-status/status.sh`** (paths relative to repo root).
Builds the CLI, runs `abi scheduler status`, and asserts the internal probe task
ran to completion plus the Prometheus telemetry block. Evidence is the `RESULT:`
line. Fully local — no state, no network.

## Run (agent path)
```bash
.agents/skills/scheduler-status/status.sh
```
Prints `RESULT: PASS` (exit 0) or `RESULT: FAIL` with the missing assertion (exit 1).

## Gotchas
- ⚠️ **The only valid subcommand is `status`.** There is no `stats` and no `info` —
  those are the *MCP tool names* (`scheduler_stats`, `scheduler_info`), not the CLI
  word. `abi scheduler stats` / `info` / bare `abi scheduler` all exit **2** with
  `usage: abi scheduler status`. The driver asserts this negative case.
- **Output is on stderr**, not stdout (it's `std.debug.print`). Capture `2>&1` —
  `2>/dev/null` blanks the whole thing.
- **No setup, no state, no network.** It's a self-contained one-shot: creates its
  own `Scheduler` + `MemoryTracker`, submits one probe task, runs it, prints, exits 0.
- **Don't use `timeout`** — macOS doesn't ship it, and the command self-terminates
  so no guard is needed (`gtimeout` from coreutils only if you must).
- **The build is near-silent** — `./build.sh cli` prints ~2 info lines and exits;
  confirm with `ls zig-out/bin/abi`, not stdout.
- Healthy output = `running=0 pending=0 completed=1 failed=0 total_tasks=1`,
  `memory_tracker=attached` (zeroed usage is normal — the probe allocates nothing),
  and a Prometheus block with `scheduler_tasks_submitted 1` / `scheduler_tasks_completed 1`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Check `zig version` (see `/zig-pin`), then `./build.sh check`. |
| empty output | You captured stdout only — scheduler prints to stderr; use `2>&1`. |
| `usage: abi scheduler status` | You used `stats`/`info`/bare `scheduler`; the CLI word is `status`. |
| `completed=0` / `total_tasks=0` | The probe task didn't run — inspect `src/core/scheduler.zig`; use the `scheduler-memory-auditor` subagent to audit submission/completion accounting. |

Historical verification: **PASS** on Zig master `0.17.0-dev.1099` — one probe task
submitted and completed, telemetry block emitted, `stats`/`info` correctly rejected.
