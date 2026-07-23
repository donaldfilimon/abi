---
name: abi-superpower-agent-orchestration
description: Local multi-agent orchestration superpower. Fixed trio, custom workers, browser planning — all scheduler-backed, no distributed agents, no embedded browser.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["multi", "spawn", "browser", "plan"]
      description: "Orchestration action"
    - name: "input"
      type: "string"
      description: "Task input"
    - name: "workers"
      type: "string"
      description: "Custom workers (name|instructions|hints; semicolon-separated)"
    - name: "background"
      type: "boolean"
      description: "Print task IDs before runAll (synchronous completion)"
    - name: "url"
      type: "string"
      description: "URL for browser task"
    - name: "execute"
      type: "boolean"
      description: "Require --confirm for browser execute"
---

# ABI Superpower: Agent Orchestration

Exposes local multi-agent orchestration as a superpower. **Honest scope**: Scheduler-backed local workers over existing AI router/constitution paths. No distributed agents, no embedded browser, no new MCP tools.

## Actions

### multi
Run fixed Abbey/Aviva/Abi trio via scheduler:
```
/abi-superpower-agent-orchestration multi --input "analyze the architecture"
```

### spawn
Run custom workers parsed from `name|instructions|hints` (semicolon-separated):
```
/abi-superpower-agent-orchestration spawn --input "refactor module" --workers "reviewer|find bugs|plan;optimizer|improve performance|explore"
```
Options:
- `--background` — prints task IDs before `runAll()` (synchronous completion, NOT a detached daemon)

### browser
Emit claim-honest local plan (`embedded_browser=false`, `delegation_hint=external-mcp-playwright`):
```
/abi-superpower-agent-orchestration browser --input "navigate to github.com" --url "https://github.com"
```
Options:
- `--execute --confirm` — REQUIRES literal `--confirm` token; local planner only runs dry-run; real navigation = external MCP step

### plan
Single-agent dry-run planning via the scheduler (maps to `abi agent plan`, not a multi-worker dry-run):
```
/abi-superpower-agent-orchestration plan --input "inspect WDBX persistence"
```
For multi-worker execution use `multi` or `spawn`. For claim-honest browser planning without navigation use `browser` (still runs a local dry-run planner worker after printing the plan).

## Architecture

| Component | Source | Role |
|-----------|--------|------|
| Orchestration | `src/features/ai/orchestration.zig` | `parseWorkerSpecs`, `defaultTrioSpecs`, `runCustomMultiAgentWithScheduler`, `submitAgentsBackground`, `planBrowserOrchestration` |
| AI module | `src/features/ai/mod.zig` | `runMultiAgentWithScheduler` (fixed trio entrypoint), `runAgent` / `submitAgentTask`, re-exports |
| Scheduler | `src/core/scheduler.zig` | Local task coordination, memory tracker integration (unconditional core — not a feature flag) |
| Router | `src/features/ai/router.zig` | Abbey/Aviva/Abi profile selection |
| Constitution | `src/features/ai/constitution.zig` | 6-principle response audit |

## Worker Model

- **Fixed trio**: Abbey (primary empathetic polymath), Aviva (direct expert), ABI (orchestration/governance) — `abi agent multi`
- **Custom**: Up to 32 workers (`max_worker_count = 32`) — `abi agent spawn`
- **Browser**: Local planner worker only; real browser = external MCP integration
- **Execution**: All workers submit through scheduler; `--background` prints task IDs then blocks on `runAll()`

## CLI Surface

| Command | Description |
|---------|-------------|
| `abi agent multi <input>` | Fixed trio orchestration |
| `abi agent spawn [--background] [--workers <spec>] <input>` | Custom workers |
| `abi agent browser [--url <url>] [--execute --confirm] <task>` | Browser planning |
| `abi agent plan <input>` | Single-agent planning |

## Feature Gates

Requires `feat-ai=true` (default). When AI is disabled, stubs preserve public symbols and return explicit degraded behavior. The scheduler lives in unconditional `src/core/` — there is no `feat-scheduler` flag.

## Claim Boundary

Per `docs/contracts/external-claims-audit.mdx` and `docs/spec/abi-refactor-design.mdx` §5.4:
- ✅ Local scheduler-backed multi-worker orchestration
- ✅ Fixed trio + custom workers (≤32)
- ✅ Browser planning with honest `embedded_browser=false`
- ❌ NOT distributed agents
- ❌ NOT embedded browser (real nav = external MCP)
- ❌ NOT additional MCP tools (frozen 12-tool surface)
- ❌ NOT autonomous daemon (background = synchronous `runAll()`)

## Implementation Notes

- Worker fan-out capped at 32
- Background submission is failure-transactional
- Feature-off stubs preserve type ownership
- CLI runtime smoke covers new surface
- MCP HTTP has transport-level wrong-bearer + oversized-body regression tests
