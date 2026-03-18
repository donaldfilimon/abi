---
title: Integration Gates v1
description: Generated implementation plan
---

# Integration Gates v1
## Status
- Status: **Complete**
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
- Unblock criterion A: **Complete** — matrix manifest with `safeVectors()` export, smoke runner timeout enforcement (5s quick / 30s CLI tiers).
- Unblock criterion B: **Complete** — preflight enhanced with `--json` output mode, `run_build.sh` availability check, distinct exit codes (0=OK, 1=blocked, 2=degraded).
- Unblock criterion C: **Complete** — environment contract documented in `docs/guides/integration-environment.md`.
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
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
