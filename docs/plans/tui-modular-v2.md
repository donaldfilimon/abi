---
title: TUI Modular Extraction v2
description: Generated implementation plan
---

# TUI Modular Extraction v2
## Status
- Status: **In Progress**
- Owner: **Abbey**
- Canonical metadata source: `src/services/tasks/roadmap_catalog.zig`
- Active execution tracker: `tasks/todo.md`

## Scope
Wave 2 active lane: complete modular extraction, enforce layout/input correctness, and expand regression tests.

## Success Criteria
- Launcher and dashboard flows use shared module boundaries without behavior drift.
- Resize, navigation, and input handling stay correct across small and full terminal layouts.
- TUI layout and hit-testing regressions are covered by deterministic tests.


## Validation Gates
- zig build cli-tests
- zig build tui-tests
- zig build run -- ui launch --help
- zig build run -- ui gpu --help


## Milestones
- Wave 2A: complete launcher/dashboard extraction onto shared render/layout primitives.
- Wave 2B: close input routing and focus-state correctness gaps.
- Wave 2C: expand unit and integration-style TUI tests for layout and hit-testing.


## Related Roadmap Items

| ID | Item | Track | Horizon | Status | Gate |
| -- | --- | --- | --- | --- | --- |
| RM-004 | Finish TUI modular extraction | CLI/TUI | Now | In Progress | zig build cli-tests ; zig build tui-tests ; zig build run -- ui launch --help ; zig build run -- ui gpu --help |

Roadmap guide: [../roadmap/](../roadmap/)



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` on supported hosts. On Darwin 25+ / 26+, use `zig fmt --check ...` plus `./tools/scripts/run_build.sh <step>`. For docs generation, use `zig build gendocs` or `./tools/scripts/run_build.sh gendocs` on Darwin.
