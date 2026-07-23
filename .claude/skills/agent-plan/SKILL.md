---
name: agent-plan
priority: high
description: Execute ABI repo work from current TODOs, roadmap/spec docs, Zig 0.17 constraints, and validation gates. Use when user asks to find all ABI todos/roadmaps, compile ABI goals/specs, organize a large ABI implementation goal, or decide the next safe work slice in ~/abi.
---

# agent-plan

## Workflow

1. Start in `/Users/donaldfilimon/abi` unless the user gives another ABI checkout.
2. Inspect `git status --short --branch` before edits; preserve unrelated dirty work.
3. Read `AGENTS.md`, `tasks/todo.md`, and `tasks/lessons.md`.
4. Optional: refresh the inventory with `abi_inventory.py --repo /Users/donaldfilimon/abi` if available (ships with the codex `abi-mega` plugin, not this repo); skip when absent.
5. Optional: load a sibling `current-goals.md` under this skill's `references/` dir if present (codex plugin copies carry it; the canonical repo copy does not) — otherwise derive the source map from `tasks/todo.md` + `docs/spec/wdbx-north-star.mdx`.
6. Derive a small executable slice. Prefer changes that make one TODO, roadmap gap, doc mismatch, or validation gap measurably more true.
7. Keep claims honest: source/build/tests override prose.
8. Verify with the narrow command that proves the slice, then the broader gate when the blast radius justifies it.

## Skill Use Cases

- Plan from `tasks/todo.md` and `docs/spec/wdbx-north-star.mdx` sections
- Analyze spec gaps and validation failures
- Generate deliverable slices for complex refactoring
- Validate feature parity with `zig build check-parity`
