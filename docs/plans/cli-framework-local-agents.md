---
title: CLI Framework + Local-Agent Fallback
description: Generated implementation plan
---

# CLI Framework + Local-Agent Fallback
## Status
- Status: **In Progress (Steady-State)**
- Owner: **Abbey**
- Last updated: 2026-03-16
- Canonical metadata source: `src/services/tasks/roadmap_catalog.zig`
- Active execution tracker: `tasks/todo.md`

### Progress Summary (2026-03-16)
Completed:
- Descriptor-driven CLI framework is working
- Provider health checks implemented
- Local-first fallback order is correct

Remaining:
- Strict-mode routing hardening (Wave 1B)
- Help/completion/assertion drift cleanup (Wave 1C)
- Regression test stabilization for llm run/session/providers/plugins flows (Wave 1D)

## Scope
Wave 1 active lane: descriptor/runtime parity, local-first provider/plugin hardening, and command help/assertion drift cleanup.

## Success Criteria
- Descriptor metadata and runtime dispatch remain parity-locked for command families.
- Provider/plugin selection and fallback chains remain deterministic in strict and fallback modes.
- CLI help and assertions stay in sync with descriptor definitions.


## Validation Gates
- zig build cli-tests
- zig build feature-tests
- zig build verify-all


## Milestones
- Wave 1A: close remaining descriptor/runtime parity gaps.
- Wave 1B: harden providers/plugins health checks and strict-mode routing.
- Wave 1C: remove stale help/completion/assertion drift across command families.
- Wave 1D: stabilize regression tests for llm run/session/providers/plugins flows.


## Related Roadmap Items

| ID | Item | Track | Horizon | Status | Gate |
| -- | --- | --- | --- | --- | --- |
| RM-003 | Finalize CLI descriptor framework cutover | CLI/TUI | Now | In Progress | zig build cli-tests ; zig build verify-all |
| RM-008 | Harden local-agent provider plugins | AI | Now | In Progress | zig build feature-tests ; zig build cli-tests |

Roadmap guide: [../roadmap/](../roadmap/)



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
