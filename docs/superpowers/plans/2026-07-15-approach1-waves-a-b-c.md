# Approach-1 Waves A–C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status (2026-07-16):** Waves **A–C source work largely complete**. Remaining open items are **optional/deferred** product follow-ons or environment-dependent smokes — not Approach-1 code gates. Do **not** claim ANE / native GPU / production FHE done.

**Goal:** Land the Zig 0.17-dev WIP, ship token streaming into agent TUI / local bridge, then finish the remaining honest parity slices without fake-completing non-goals.

**Architecture:** Wave pipeline on the frozen CLI (13 cmds) and MCP (12 tools). Source and tests override prose. Disclosed stubs (accelerator/shaders/mlir/mobile/ANE/production FHE) stay disclosed.

**Tech Stack:** Zig 0.17.0-dev (`.zigversion`), WDBX, SEA, connectors (`local_bridge` + SSE), TUI REPL, `./build.sh check`.

---

## Scope map (C1–C6)

| ID | Item | Wave | Status target |
|----|------|------|---------------|
| C1 | Stabilize & land dirty WIP | **A** | ✅ Done — conventional commits; `./build.sh check` green |
| C2 | Streaming parity (SSE + TUI + CLI) | **B** | ✅ Done in source — optional live smoke deferred |
| C3 | TUI multi-turn / pane remainder | **C** | ✅ Multi-turn + `/pane` split landed (`repl_pane.zig`) |
| C4 | WDBX REST rate limit + TLS config honesty | **C** | ✅ Done — committed + wired; claim boundary held |
| C5 | Doc/todo/skill hygiene | **C** | ✅ Board/skills hygiene; sync-clis operator-optional |
| C6 | Zig 0.17 hygiene only on real debt | **C** | ✅ Standing policy; pin green per `tasks/todo.md` |

### Explicit non-goals (never mark Done without real native proof)

- ANE / CoreML dispatch
- Fake native GPU/CUDA/TPU stubs
- Production multi-host sharding / audited FHE / SOTA compression
- New top-level CLI commands or MCP tools beyond frozen surfaces

---

## Wave A — Land WIP

- [x] Inventory dirty tree (~91 paths) and baseline `./build.sh check`
- [x] Confirm agent surfaces via `.agents/skills/agent-plan-train/agent.sh`
- [x] Commit by subsystem (conventional commits):
  - `d69ee6cd` feat: stream bridge SSE, agent TUI multi-turn, WDBX rate/TLS config
  - `654cc951` chore(skills): expand agent skills pack and Approach-1 wave plan
- [x] Exclude untracked: `modernized/`, `datasets/`, `.superpower-verification.json`, `test_build_flags.sh`
- [x] Re-run `./build.sh check` after implementation (exit 0; parity green)

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
- [x] Prove with gate: `zig build test -Dtest-filter=parseSse` + `./build.sh check` (exit 0)
- [ ] Optional live smoke (DEFERRED — requires local server; not a code gate): local llama-server + `abi complete --stream --model llama-cpp/…`

## Wave C — Remaining Approach-1 slices

### C3 TUI multi-turn (partial already)

- [x] Ring buffer `MAX_TURN_HISTORY` + history injection in `completePrompt`
- [x] Optional pane split (`/pane` via `repl_pane.zig` — chat left, `git diff --stat` right)

### C4 WDBX REST partial hardening

- [x] `rate_limiter.zig` + `tls_config.zig` present and imported from `mod.zig` / `rest.zig`
- [x] Committed in git with tests (`src/features/wdbx/rate_limiter.zig`, `tls_config.zig`; wired from `mod.zig` / `rest.zig`; rest unit coverage)
- [x] Claim boundary held: loopback + optional bearer + rate-limit env; native TLS not linked (see AGENTS.md / REST docs)

### C5 Hygiene

- [x] `tasks/todo.md` north-star honest status
- [x] Skill tooling: 9 new superpower skills created from docs/specs (agent-orchestration, constitution, wdbx-cluster, wdbx-compute, wdbx-secure, claims-validator, wdbx-persistence, mcp-transport, plugin-system)
- [x] `/abi-skills` sync-clis: skills already under `.agents/skills/`; full sync-clis is an optional operator step (copies to local `.claude`/`.grok` trees — not an Approach-1 code gate)

### C6 Zig 0.17

- [x] Pin green (historical land at `0.17.0-dev.1275+59a628c6d`; current pin `0.17.0-dev.1398+cb5635714` per `.zigversion` / `tasks/todo.md`)
- [x] Standing policy: only fix compile debt surfaced by `./build.sh check` / master nightly skill — no big-bang rewrite; pin currently green per `tasks/todo.md`

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

1. **A** — stabilize + commit WIP ✅  
2. **B** — streaming ✅ (source + unit gates; live llama-server smoke deferred)  
3. **C** — REST/skills hygiene ✅ (`/pane` split landed)  

---

## Self-review

| Spec item | Task coverage |
|-----------|---------------|
| C1 land WIP | Wave A commits ✅ |
| C2 streaming | Wave B files + tests ✅; live smoke deferred |
| C3 multi-turn | Landed + `/pane` split ✅ |
| C4 REST | rate_limiter/tls_config committed + wired; claim boundary held ✅ |
| C5 docs/skills | todo + skills under `.agents/skills/`; sync-clis operator-optional ✅ |
| C6 zig | standing policy; pin green; no rewrite ✅ |
| Non-goals | Listed; not implemented as Done |
