# Plan: Priority A G1–G5 — commit, review, land

**Date:** 2026-07-14  
**Branch:** `feat/priority-a-g1-g5` (from `a2e36b28`)  
**Status:** Implementation already present in working tree (verified `./build.sh check` 39/39). SDD executes **scoped verify + atomic commits + task reviews + final branch review**.

## Goal kind
code-change (package and quality-gate existing implementation)

## Global Constraints
- Zig pin `.zigversion` → `0.17.0-dev.1252+e4b325c19`; prefer `./build.sh …` on macOS.
- **Do not** expand frozen CLI top-level commands or the **12** MCP tools.
- No unproven claims (sharding, production FHE, multi-host, QPS). G5 is store-dir `0700` only — not multi-host security.
- Public feature API name changes → mod + stub + `check-parity` (this work should not need public API renames).
- Conventional Commits; one logical slice per commit.
- Preserve unrelated dirty work outside each task’s file list.
- Tests must exercise **shipped** code (no reimplementation in tests).
- Windows ACL/keychain remain **disclosed gaps** (do not fake-complete).

## Non-goals
- Priority B (G6–G9), Priority C north-star partials, Windows runtime verification.
- Keyed WAL HMAC (plan allowed perms **or** HMAC; implementation chose `0700`).
- Expanding MCP/CLI surfaces.

## Task 1: Commit G1 — REPL line editor

**Files only:**
- `src/features/tui/line_editor.zig` (new)
- `src/features/tui/repl.zig`
- `tools/run_tui_smoke.sh`

**Done when:**
1. Focused tests pass: `zig build test -Dtest-filter="line editor"` exit 0.
2. `tools/run_tui_smoke.sh` exits 0 (or documents tmux skip for PTY-only path).
3. Single commit: `feat(tui): add pure REPL line editor with CSI decode and history`.
4. Report lists SHA + test evidence.

## Task 2: Commit G2 — MCP JSON depth bound

**Files only:**
- `src/mcp/protocol.zig`
- `src/mcp/rpc.zig`
- `src/mcp/stdio_transport.zig`
- `src/mcp/handlers.zig` (only if errorMessage mapping for JsonTooDeep)
- `src/mcp/middleware.zig` (only if comment-only / no surface change)

**Done when:**
1. `zig build test -Dtest-filter="protocol:"` and nested/oversize/bearer-related filters exit 0.
2. No new MCP tool names.
3. Commit: `feat(mcp): reject over-nested JSON-RPC requests at shared parse boundary`.

## Task 3: Commit G3 — HTTPS live URLs + no-echo signin

**Files only:**
- `src/connectors/http.zig`
- `src/connectors/connector.zig`
- `src/cli/handlers/auth.zig`

**Done when:**
1. HTTPS require tests pass; secret-entry no-echo POSIX test passes.
2. Windows gap disclosed in signin help.
3. Commit: `feat(security): require HTTPS live connectors and no-echo auth signin`.

## Task 4: Commit G4 — ai_train path sandbox

**Files only:**
- `src/features/ai/training_support.zig`
- `src/features/ai/training.zig`
- `src/mcp/handlers.zig` / `middleware.zig` only if still uncommitted path-related bits

**Done when:**
1. `confineTrainingPath` tests pass (accept under root, reject abs outside, symlink escape).
2. Commit: `feat(ai): confine training dataset and artifact paths under data root`.

## Task 5: Commit G5 — store dir 0700 + board note

**Files only:**
- `src/features/wdbx/durable_store.zig`
- `tasks/todo.md`

**Done when:**
1. Owner-only POSIX test passes.
2. `tasks/todo.md` Priority A rows present and marked done.
3. Commit: `feat(wdbx): create durable store parent dirs as owner-only on POSIX`.
4. Optional second commit if board-only: `docs(tasks): mark Priority A G1–G5 complete` — prefer **one** commit with both if already co-edited.

## Task 6: Full gate re-run (after Tasks 1–5 commits)

**Done when:**
1. `./build.sh check` exit 0.
2. Dual `abi help` OK; MCP tools/list still 12 tools.
3. No uncommitted Priority A files remain (except intentional scratch).

## Final whole-branch review
- Range: `a2e36b28` .. `HEAD` on `feat/priority-a-g1-g5`
- Spec: acceptance criteria G1–G5 + global constraints
- Then finishing-a-development-branch options
