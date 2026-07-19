---
name: agent-status-reporter
description: Report current agent/session status and system health — skill-loop run state, scheduler/memory counters, and active context. Use when the user runs /status in the abi REPL or asks what the agent is doing or for system health.
---

# agent-status-reporter

Backs the `/status` slash command in the `agent tui` REPL
(`opencode.json` slash-command entry `status`). Reports a compact, read-only
snapshot:

## What to report
1. **Skill-loop run state** — read the last entry of
   `.skill-telemetry/runs.jsonl` and `.skill-telemetry/.sessions/` for the
   active session id + status. Summarize the latest run's status + counts; do
   not dump the file.
2. **Scheduler / memory** — run `abi scheduler status` (one-shot
   self-terminating probe): report counters + attached MemoryTracker stats +
   the always-on telemetry block. The probe is a no-op, so memory counters read
   0 by design — say so; do not fabricate load.
3. **Context** — give one line: loaded files + whether SEA mode is on (from
   `sea-learning-controller`). Defer the full context view to
   `context-state-reporter` (`/context`).

## Rules
- Read-only. Never edit source, telemetry, or session files.
- No performance / accuracy claims without a repo benchmark.
- If a telemetry file is missing, report "not initialized" — do not invent.
