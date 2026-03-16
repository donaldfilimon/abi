---
title: TUI Modular Extraction v2
description: Generated implementation plan
---

# TUI Modular Extraction v2
## Status
- Status: **In Progress (Steady-State)**
- Owner: **Abbey**
- Last updated: 2026-03-16
- Canonical metadata source: `src/services/tasks/roadmap_catalog.zig`
- Active execution tracker: `tasks/todo.md`

### Progress Summary (2026-03-16)
Completed:
- Layout engine extraction onto shared primitives
- Launcher extraction complete
- Dashboard extraction complete

Remaining:
- Regression test coverage expansion (Wave 2C)
- Input routing and focus-state correctness gaps (Wave 2B)

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
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
