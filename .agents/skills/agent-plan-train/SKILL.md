---
name: agent-plan-train
description: Plan and execute ABI repo work from current TODOs, roadmap/spec docs, nightly Rust constraints, and validation gates. Use when user asks to find all ABI todos/roadmaps, compile ABI goals/specs, organize a large ABI implementation goal, or decide the next safe work slice in ~/abi.
---

# agent-plan-train

## Workflow

1. Start in `/Users/donaldfilimon/abi` unless the user gives another ABI checkout.
2. Inspect `git status --short --branch` before edits; preserve unrelated dirty work.
3. Read `AGENTS.md`, `tasks/todo.md`, and `tasks/lessons.md`.
4. Optional: refresh the inventory with the abi-mega `abi_inventory` Python helper (`--repo /Users/donaldfilimon/abi`) if available (ships with the codex `abi-mega` plugin, not this repo); skip when absent.
5. Optional: load a sibling goals markdown named `current-goals` under this skill's `references/` dir if present (codex plugin copies carry it; the canonical repo copy does not) — otherwise derive the source map from `tasks/todo.md` + `docs/spec/wdbx-north-star.mdx`.
6. Derive a small executable slice. Prefer changes that make one TODO, roadmap gap, doc mismatch, or validation gap measurably more true.
7. Keep claims honest: source/build/tests override prose.
8. Verify with the narrow command that proves the slice, then the broader gate when the blast radius justifies it.

## Goal Rules

- Treat `tasks/todo.md` as the active board and `docs/spec/wdbx-north-star.mdx` as the Current/Partial/Proposed map.
- Do not convert disclosed stubs into fake completions. Native dispatch, production clustering, production FHE, and learned-compression claims need real source/tests.
- Do not add legacy CLI names. Preserve the frozen top-level command set and the MCP 12-tool contract.
- When changing public feature APIs, update both the real and stub modules and run `./tools/check.sh`.
- When changing docs, run `.agents/skills/docs-validate/validate.sh` in addition to code gates.

## Useful Commands

```bash
# optional; codex abi-mega plugin only:
#   abi_inventory --repo /Users/donaldfilimon/abi
./tools/cargo.sh --version
./tools/check.sh
```

Use the optional sibling goals markdown named `current-goals` under this skill's `references/` dir (when present — see Workflow step 5) for the current source inventory and validation ladder.