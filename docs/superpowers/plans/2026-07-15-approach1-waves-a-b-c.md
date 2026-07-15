# Approach-1 Waves A–C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the Zig 0.17-dev WIP, ship token streaming into agent TUI / local bridge, then finish the remaining honest parity slices without fake-completing non-goals.

**Architecture:** Wave pipeline on the frozen CLI (13 cmds) and MCP (12 tools). Source and tests override prose. Disclosed stubs (accelerator/shaders/mlir/mobile/ANE/production FHE) stay disclosed.

**Tech Stack:** Zig 0.17.0-dev (`.zigversion`), WDBX, SEA, connectors (`local_bridge` + SSE), TUI REPL, `./build.sh check`.

---

## Scope map (C1–C6)

| ID | Item | Wave | Status target |
|----|------|------|---------------|
| C1 | Stabilize & land dirty WIP | **A** | Conventional commits; `./build.sh check` green |
| C2 | Streaming parity (SSE + TUI + CLI) | **B** | Bridge SSE callback fixed; agent tui + `complete --stream` wired |
| C3 | TUI multi-turn / pane remainder | **C** | Multi-turn history already partial; pane split optional follow-on |
| C4 | WDBX REST rate limit + TLS config honesty | **C** | Land `rate_limiter.zig` / `tls_config.zig` with rest wiring; still loopback-only |
| C5 | Doc/todo/skill hygiene | **C** | Board matches source; skill inventory refs |
| C6 | Zig 0.17 hygiene only on real debt | **C** | No big-bang rewrite |

### Explicit non-goals (never mark Done without real native proof)

- ANE / CoreML dispatch
- Fake native GPU/CUDA/TPU stubs
- Production multi-host sharding / audited FHE / SOTA compression
- New top-level CLI commands or MCP tools beyond frozen surfaces

---

## Wave A — Land WIP

- [x] Inventory dirty tree (~91 paths) and baseline `./build.sh check`
- [x] Confirm agent surfaces via `.agents/skills/agent-plan-train/agent.sh`
- [ ] Commit by subsystem (conventional commits):
  - `feat(connectors|ai|tui|cli|wdbx|plugins):` source
  - `chore(skills):` skill pack
  - `docs:` instruction + todo board
- [ ] Exclude or leave untracked: `modernized/`, large `datasets/`, scratch `.superpower-verification.json` unless intentional
- [ ] Re-run `./build.sh check` after commits

## Wave B — Streaming token path

**Files:**

- Modify: `src/connectors/http.zig` (SSE parse — real `callback_ctx`, accumulate tokens, `[DONE]`)
- Modify: `src/connectors/local_bridge.zig` (`completeLiveStreaming`)
- Modify: `src/features/tui/repl.zig` (bridge model → SSE stream → print deltas)
- Modify: `src/cli/handlers/train.zig` (`complete --stream` on bridge models)

- [x] Fix SSE `processSseEvent` / `parseSseStream` to pass real callback context
- [x] Accumulate full streamed text for history/pushTurn
- [x] Unit tests: parseSseEvent + multi-token parseSseStream
- [x] Agent TUI: if `isLocalBridgeModel`, health-check → `completeLiveStreaming` with sanitize print
- [x] CLI: `handleLocalBridgeComplete(..., stream)`
- [ ] Prove with gate: `zig build test -Dtest-filter=parseSse` + `./build.sh check`
- [ ] Optional live smoke: local llama-server + `abi complete --stream --model llama-cpp/…` (requires server)

## Wave C — Remaining Approach-1 slices

### C3 TUI multi-turn (partial already)

- [x] Ring buffer `MAX_TURN_HISTORY` + history injection in `completePrompt`
- [ ] Optional: pane split chat+diff — **defer** unless product priority (no frozen CLI change)

### C4 WDBX REST partial hardening

- [x] `rate_limiter.zig` + `tls_config.zig` present and imported from `mod.zig` / `rest.zig`
- [ ] Ensure untracked files are committed with tests
- [ ] Keep claims: loopback + optional bearer + rate limit env; native TLS not linked

### C5 Hygiene

- [x] `tasks/todo.md` north-star honest status
- [ ] Skill tooling: drop or implement missing `scripts/abi_inventory.py` / `current-goals.md` references
- [ ] `/abi-skills`: sync-clis after skill land

### C6 Zig 0.17

- [x] Pin `0.17.0-dev.1275+59a628c6d` green
- [ ] Only fix compile debt surfaced by `./build.sh check` / master nightly skill

---

## Validation ladder (every wave)

```bash
zig version   # must match .zigversion
zig build check-parity --summary all   # if public API
./build.sh check
.agents/skills/agent-plan-train/agent.sh
# optional:
.agents/skills/complete-base/complete.sh
.agents/skills/mcp-smoke/smoke.sh   # if present
```

---

## Execution order (user-approved)

1. **A** — stabilize + commit WIP  
2. **B** — streaming (in progress / largely implemented this session)  
3. **C** — plan remainder + REST/skills hygiene  

---

## Self-review

| Spec item | Task coverage |
|-----------|---------------|
| C1 land WIP | Wave A commits |
| C2 streaming | Wave B files + tests |
| C3 multi-turn | Landed partial; pane split deferred |
| C4 REST | Land rate_limiter/tls_config |
| C5 docs/skills | todo + sync-clis |
| C6 zig | pin only, no rewrite |
| Non-goals | Listed; not implemented as Done |
