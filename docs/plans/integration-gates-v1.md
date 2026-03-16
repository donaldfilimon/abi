---
title: Integration Gates v1
description: Generated implementation plan
---

# Integration Gates v1
## Status
- Status: **Blocked**
- Owner: **Abbey**
- Canonical metadata source: `src/services/tasks/roadmap_catalog.zig`
- Active execution tracker: `tasks/todo.md`

## Scope
Wave 4 blocked lane: restore exhaustive integration gates after explicit unblock criteria are met while keeping interim gate policy green.

## Success Criteria
- cli-tests-full is deterministic and isolated across command matrices.
- Preflight diagnostics clearly identify environment, tool, and network blockers.
- Interim cli-tests/tui-tests/launcher-smoke policy remains required until unblock completion.


## Validation Gates
- zig build cli-tests-full
- zig build cli-tests
- zig build tui-tests
- zig build run -- ui launch --help
- zig build run -- ui gpu --help
- zig build verify-all


## Milestones
- Unblock criterion A: complete matrix manifest and PTY timeout policy hardening.
- Unblock criterion B: deliver actionable preflight blocked-report diagnostics.
- Unblock criterion C: document and validate required integration environment contract.
- Policy guard: keep interim cli-tests/tui-tests/launcher smoke checks passing while blocked.


## Related Roadmap Items

| ID | Item | Track | Horizon | Status | Gate |
| -- | --- | --- | --- | --- | --- |
| RM-007 | Complete exhaustive CLI integration gate | Infrastructure | Next | Blocked | zig build cli-tests-full ; zig build cli-tests ; zig build tui-tests ; zig build run -- ui launch --help ; zig build run -- ui gpu --help |

Roadmap guide: [../roadmap/](../roadmap/)



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` on supported hosts. On Darwin 25+ / 26+, use `zig fmt --check ...` plus `./tools/scripts/run_build.sh <step>`. For docs generation, use `zig build gendocs` or `./tools/scripts/run_build.sh gendocs` on Darwin.
