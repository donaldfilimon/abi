---
name: abi-goal-orchestrator
description: Plan and execute ABI repo work from the TODO board, roadmap/spec docs, Zig 0.17 constraints, and validation gates. Use when the user asks to find ABI todos/roadmaps, organize a large implementation from tasks/todo.md, or decide the next safe work slice. Distinct from the goals skill — that skill owns the persistent ledger tasks/goals.md; this skill plans from the active board and north-star specs.
---

# ABI Goal Orchestrator

Use this skill to turn ABI's active board and long-horizon specs into a concrete implementation plan grounded in current repo files.

**Boundary:** `goals` owns `tasks/goals.md` (user intentions / goal-to-slice ledger). This skill plans from `tasks/todo.md` + `docs/spec/wdbx-north-star.mdx`. When the user names a durable goal, capture/update it via the `goals` skill; use this skill for board/spec-driven slice selection.

## Workflow

1. Start in `/Users/donaldfilimon/abi` unless the user gives another ABI checkout.
2. Inspect `git status --short --branch` before edits; preserve unrelated dirty work.
3. Read `AGENTS.md`, `tasks/todo.md`, and `tasks/lessons.md`. Optionally skim `tasks/goals.md` for in-progress user intentions (do not treat it as the working list).
4. Optional: refresh the inventory with `abi_inventory.py --repo /Users/donaldfilimon/abi` if available (ships with the codex `abi-mega` plugin, not this repo); skip when absent.
5. Optional: load `references/current-goals.md` if present alongside this skill (codex plugin copies carry it; the canonical repo copy does not) — otherwise derive the source map from `tasks/todo.md` + `docs/spec/wdbx-north-star.mdx`.
6. Derive a small executable slice. Prefer changes that make one TODO, roadmap gap, doc mismatch, or validation gap measurably more true.
7. Keep claims honest: source/build/tests override prose. Do not fake-complete stubs.
8. Verify with the narrow command that proves the slice, then the broader gate when the blast radius justifies it.

## Goal Rules

- Treat `tasks/todo.md` as the active board and `docs/spec/wdbx-north-star.mdx` as the Current/Partial/Proposed map. Leave coarse user intentions in `tasks/goals.md` (via the `goals` skill).
- Do not convert disclosed stubs into fake completions. Native dispatch, production clustering, production FHE, and learned-compression claims need real source/tests/artifacts.
- Do not add legacy CLI names. Preserve the frozen top-level command set and MCP 12-tool contract.
- When changing public feature APIs, update real and stub modules and run `./build.sh check-parity`.
- When changing docs, run `.agents/skills/docs-validate/validate.sh` in addition to code gates.

## Useful Commands

```bash
abi_inventory.py --repo /Users/donaldfilimon/abi   # optional; codex abi-mega plugin only
zig version
./build.sh check-parity
./build.sh check
```

Use `references/current-goals.md` (when present — see Workflow step 5) for the current source inventory and validation ladder.
