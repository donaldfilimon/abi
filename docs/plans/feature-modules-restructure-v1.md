---
title: Feature Modules Restructure v1
description: Generated implementation plan
---

# Feature Modules Restructure v1
## Status
- Status: **Complete**
- Owner: **Abbey**
- Last updated: 2026-03-16
- Canonical metadata source: `src/services/tasks/roadmap_catalog.zig`
- Active execution tracker: `tasks/todo.md`

### Completion Summary (2026-03-16)
All 18 phases of restructuring are complete (commit `04002b79`). Key deliverables:
- persona to profile rename across all modules
- Bridge retirement and legacy facade removal
- Import compliance (single `abi` module, relative imports within `src/`)
- Zero mod/stub signature drift
- Feature gating validated across all 54 flag combinations

No remaining work items for this plan.

## Scope
Wave 5 active lane: remove legacy facades, finalize module boundaries, and consolidate shared primitives.

## Success Criteria
- No stale imports remain against removed facade modules.
- Module boundaries are explicit and stable for feature enable/disable permutations.
- Shared primitives are centralized and reused without duplicate local forks.


## Validation Gates
- zig build validate-flags
- zig build full-check


## Milestones
- Wave 5A: finish AI/service boundary cleanup and remove obsolete facade surfaces.
- Wave 5B: consolidate shared primitives into canonical modules.
- Wave 5C: update integration roots and tests to the final module topology.


## Related Roadmap Items

| ID | Item | Track | Horizon | Status | Gate |
| -- | --- | --- | --- | --- | --- |
| RM-009 | Complete feature module hierarchy cleanup | Platform | Now | In Progress | zig build validate-flags ; zig build full-check |
| RM-012 | Expand cloud function adapters | Platform | Later | Planned | zig build full-check ; zig build verify-all |

Roadmap guide: [../roadmap/](../roadmap/)



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
