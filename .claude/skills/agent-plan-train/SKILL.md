---
name: agent-plan-train
description: Plan and execute ABI repo work from current TODOs, roadmap/spec docs, Zig 0.17 constraints, and validation gates. Use when user asks to find all ABI todos/roadmaps, compile ABI goals/specs, organize a large ABI implementation goal, or decide the next safe work slice in ~/abi.

## Workflow

1. Start in `/Users/donaldfilimon/abi` unless the user gives another ABI checkout.
2. Inspect `git status --short --branch` before edits; preserve unrelated dirty work.
3. Read `AGENTS.md`, `tasks/todo.md`, and `tasks/lessons.md`.
4. Generate or refresh the inventory with `../../scripts/abi_inventory.py --repo /Users/donaldfilimon/abi`.
5. Load `references/current-goals.md` for the source map and goal taxonomy.
6. Derive a small executable slice. Prefer changes that make one TODO, roadmap gap, doc mismatch, or validation gap measurably more true.
7. Keep claims honest: source/build/tests override prose.
8. Verify with the narrow command that proves the slice, then the broader gate when the blast radius justifies it.

## Goal Rules

- Treat `tasks/todo.md` as the active board and `docs/spec/wdbx-north-star.mdx` as the Current/Partial/Proposed map.
- Do not convert disclosed stubs into fake completions. Native dispatch, production clustering, production FHE, and learned-compression claims need real source/tests.
- Do not add legacy CLI names. Preserve the frozen top-level command set and the MCP 12-tool contract.
- When changing public feature APIs, update both the real and stub modules and run `zig build check-parity`.
- When changing docs, run `.agents/skills/docs-validate/validate.sh` in addition to code gates.

## Useful Commands

```bash
../../scripts/abi_inventory.py --repo /Users/donaldfilimon/abi
zig version
zig build check-parity --summary all
./build.sh check
```

Use `references/current-goals.md` for the current source inventory and validation ladder.