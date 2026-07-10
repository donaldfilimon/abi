---
name: agent-plan-train
description: Build the abi CLI and drive non-interactive agent surfaces — plan, train, multi, spawn, and browser orchestration. Use after touching src/features/ai/ profiles, routing, orchestration.zig, or agent CLI handlers.
---

# agent-plan-train — drive abi agent surfaces

Driver: **`.agents/skills/agent-plan-train/agent.sh`** (paths relative to repo root).
Builds the CLI and exercises non-interactive `agent` subcommands. Evidence is
the `RESULT:` line. Fully local, no network.

## Run (agent path)
```bash
.agents/skills/agent-plan-train/agent.sh                                  # default plan, profile=abbey
.agents/skills/agent-plan-train/agent.sh "draft a release note" all       # custom plan, train all
```

What it asserts:

| Command | Markers |
|---------|---------|
| `agent plan` | `agent=cli-agent`, `mode=dry-run`, `selected_profile=`, `response=` |
| `agent train` | `training executed via scheduler`, `recorded in wdbx` |
| `agent multi` | `MULTI-AGENT RESULTS` (Abbey/Aviva/Abi trio) |
| `agent spawn` | `CUSTOM MULTI-AGENT RESULTS` |
| `agent browser` | `embedded_browser=false`, `delegation_hint=external-mcp-playwright` |
| `agent browser --execute` without `--confirm` | exit **2** |

Prints `RESULT: PASS` (exit 0) or a FAIL count.

## Gotchas
- `agent train` does **real** scheduler work and appends training metadata to WDBX.
- `agent multi` / `spawn` / `browser` are local orchestration only — no embedded browser, no distributed agents, no new MCP tools.
- Skipped: `agent tui` (interactive) and `agent os` (use `os-control-dryrun`).
- For router/constitution internals, use the `ai-constitution-reviewer` subagent.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Check `zig version` (see `/zig-pin`), then `./build.sh check`. |
| multi/spawn FAIL | Check `src/features/ai/orchestration.zig` and `src/cli/handlers/agent.zig`. |
| browser missing `embedded_browser=false` | Claim-honesty regression in `planBrowserOrchestration`. |
