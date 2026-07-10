---
name: agent-plan-train
description: Build the abi CLI and drive the non-interactive agent surfaces — `agent plan` (dry-run planning via the profile router) and `agent train <profile>` (real scheduler-backed training that records metadata in WDBX). Use to smoke-test agent planning/training after touching src/features/ai/ profiles, routing, or the scheduler wiring.
---

# agent-plan-train — drive abi's agent plan + train surfaces

Driver: **`.agents/skills/agent-plan-train/agent.sh`** (paths relative to repo root).
Builds the CLI and drives the two non-interactive `agent` subcommands. Evidence
is the `RESULT:` line. Fully local, no network.

## Run (agent path)
```bash
.agents/skills/agent-plan-train/agent.sh                                  # default plan, profile=abbey
.agents/skills/agent-plan-train/agent.sh "draft a release note" all       # custom plan, train all profiles
```
- `agent plan "<text>"` → asserts `agent=cli-agent`, `mode=dry-run`,
  `selected_profile=`, `response=` (routes through the persona router; dry-run
  never executes).
- `agent train <abbey|aviva|abi|all>` → asserts `training executed via
  scheduler`, `recorded in wdbx` (the printed message varies: "training metadata recorded in wdbx..." for a single profile, "known agent profiles recorded in wdbx" for `all`).

Prints `RESULT: PASS` (exit 0) or a FAIL count.

Historical verification: **PASS** on Zig master `0.17.0-dev.1099` — plan selects a
profile and returns a dry-run response; `agent train abbey` runs real scheduler
tasks and records training metadata in WDBX.

## Gotchas
- `agent train` does **real** work (scheduler tasks) and appends training
  metadata to the default WDBX store — not a demo no-op. `agent train all`
  trains all three profiles and is slower.
- Skipped by design: `agent tui` (interactive REPL; non-TTY stdin uses line-mode
  fallback and can be smoke-tested separately) and `agent os` (command-execution policy — covered by the
  `os-control-dryrun` skill, which never executes).
- `agent plan` `mode=dry-run` / `review_required=false` means it plans only; it
  does not execute the described operation.
- For source-level reasoning about the Abbey/Aviva/Abi router weights and the
  constitution, use the `ai-constitution-reviewer` subagent.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Check `zig version` (see `/zig-pin`), then `./build.sh check`. |
| `training executed via scheduler` missing | Scheduler/training wiring drift — check the `agent` handler in `src/cli/handlers/` and `src/features/ai/`. |
