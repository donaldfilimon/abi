---
name: goals
description: Capture, track, and execute goals for the abi Zig project. Use when the user wants to record a goal, see outstanding goals, update goal status, or break a goal into a verified work slice. Distinct from abi-goal-orchestrator (which plans from the TODO board) — this skill owns a persistent goal ledger and drives goal-to-slice execution.
---

# Goals

A lightweight goal ledger for the abi project plus a loop that turns a goal into
a verifiable work slice and records the outcome. Goals persist in
`tasks/goals.md` so they survive sessions; the active board (`tasks/todo.md`)
remains the granular working list.

## Ledger

The ledger is `tasks/goals.md`, a markdown file with one `## <Goal>` section per
goal. Each section carries a status line and a short body:

```
## Ship file-aware agent context
status: done
- Wired @file resolution into agent plan/multi + TUI repl
- Added ContextBudget path-escape rejection + 5 resolution tests
```

Valid `status:` values: `todo`, `in_progress`, `blocked`, `done`.

If `tasks/goals.md` does not exist, create it with a `# Goals` header. Treat the
file as the source of truth; never delete a closed goal — move a one-line note
to the bottom or mark `done`.

## Workflow

1. Start in `/Users/donaldfilimon/abi` unless the user gives another checkout.
2. Inspect `git status --short --branch` before edits; preserve unrelated dirty work.
3. Read `AGENTS.md`, `tasks/todo.md`, and `tasks/lessons.md` for conventions and the
   active board.
4. **Capture**: if the user states a goal, ensure it has a `## <Goal>` section in
   `tasks/goals.md` with `status: todo` (or `in_progress` if started this turn).
5. **Track**: `list` goals by reading `tasks/goals.md` and reporting each goal with
   its status. `update <goal> <status>` rewrites that goal's `status:` line.
6. **Execute**: derive the smallest slice that moves the goal forward. Prefer a
   change that makes one TODO, roadmap gap, doc mismatch, or validation gap
   measurably more true. Implement it, then verify.
7. Keep claims honest: source/build/tests override prose. Update the goal's status
   and append a one-line outcome note when the slice lands.

## Goal Rules

- Do not convert disclosed stubs into fake completions. Native dispatch, production
  clustering, production FHE, and learned-compression claims need real source/tests.
- Do not add legacy CLI names. Preserve the frozen top-level command set and the
  MCP 12-tool contract.
- When changing public feature APIs, update both the real and stub modules and run
  `./build.sh check-parity`.
- When changing docs, run `.agents/skills/docs-validate/validate.sh` in addition to
  code gates.
- Goals are user intentions, not tasks — keep `tasks/goals.md` coarse and let
  `tasks/todo.md` hold the granular steps.

## Verification Ladder

Prove the slice with the narrow command first, then broaden when the blast radius
justifies it:

```bash
./build.sh check-parity
./build.sh check
```


Feature-off parity (each must compile cleanly):

```bash
zig build cli -Dfeat-accelerator=false
zig build cli -Dfeat-ai=false
zig build cli -Dfeat-gpu=false
zig build cli -Dfeat-hash=false
zig build cli -Dfeat-metrics=false
zig build cli -Dfeat-mlir=false
zig build cli -Dfeat-mobile=false
zig build cli -Dfeat-nn=false
```

## Notes

- This skill is project-scoped (lives under `.agents/skills/goals/`); it does not
  change the CLI surface or MCP tools.
- Mirrors the taxonomy and honesty constraints of `abi-goal-orchestrator` but owns
  the persistent ledger rather than reading the TODO board only.
