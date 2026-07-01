---
name: os-control-dryrun
description: Build the abi CLI and exercise the OS-control policy in dry-run (planning) mode only, and verify the execute path refuses without --confirm. Use when working on src/features/os_control/ or the `abi agent os` handler, or to safely demo OS-command planning. Never executes a command.
---

# os-control-dryrun — safely drive abi's OS-control planning

Driver: **`.claude/skills/os-control-dryrun/dryrun.sh`** (paths relative to repo root).
Read-only-effect CLI check — evidence is the `RESULT:` line. **Nothing is executed.**

## Run (agent path)
```bash
.claude/skills/os-control-dryrun/dryrun.sh                       # default plan
.claude/skills/os-control-dryrun/dryrun.sh "restart the cache"   # custom plan text
```
Builds the CLI, runs `abi agent os dry-run "<plan>"` (asserts the `dry-run:`
marker), then asserts `abi agent os execute` **without** `--confirm` is refused
with usage (exit 2). Prints `RESULT: PASS` (exit 0) or a FAIL count.

Verified this session: **PASS** on Zig master `0.17.0-dev.1099` — dry-run emits
the plan; execute-without-confirm returns exit 2.

## Gotchas
- ⚠️ **This skill never runs `agent os execute --confirm`** — that actually runs
  the planned OS command and is deliberately out of scope. Only `dry-run` (plan)
  and the negative confirm-gate check are exercised.
- `feat-os-control` is on by default; the policy lives in `src/features/os_control/`.
  Use the `os-control-policy-reviewer` subagent to audit the validation/whitelist.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | `/zig-build-doctor` or `./build.sh check`. |
| execute-without-confirm didn't return 2 | the confirm gate regressed — check `src/cli/handlers/agent.zig` agent-os path. |
