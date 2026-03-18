---
title: Docs + Assistant Canonical Sync
description: Generated implementation plan
---

# Docs + Assistant Canonical Sync

> **Status Update (2026-03-18):** Superseded by 18-phase restructuring. Most milestones completed in commit 04002b79.

## Status
- Status: **Complete**
- Owner: **Abbey**
- Canonical metadata source: `src/services/tasks/roadmap_catalog.zig`
- Active execution tracker: `tasks/todo.md`

## Scope
Canonical docs wave: align AGENTS.md, zig-master, tasks/todo.md, root status docs, and generated outputs around one workflow and Zig validation contract.

## Success Criteria
- Workflow ownership is consistent: AGENTS.md for repo policy, zig-master for Zig validation, tasks/todo.md for active execution, and tasks/lessons.md for corrections.
- Generated docs pick up the updated workflow and zig-master wording from canonical gendocs sources in one regeneration wave.
- TODO.md, PLAN.md, and ROADMAP.md act as summary/index surfaces instead of competing live task trackers.


## Validation Gates
- zig build gendocs -- --no-wasm --untracked-md
- zig build gendocs -- --check --no-wasm --untracked-md
- zig build check-docs
- zig build verify-all
- zig build check-workflow-orchestration-strict --summary all


## Milestones
- [x] Update roadmap catalog metadata first, then regenerate docs/plans outputs with --no-wasm --untracked-md.
- [x] Collapse active execution tracking into tasks/todo.md and archive or demote overlapping root/task status files.
- [x] Rewrite assistant-facing docs so CLAUDE.md and GEMINI.md become wrappers around AGENTS.md plus zig-master.
- [x] Close the wave with docs drift checks, strict workflow orchestration checks, and the zig-master verification sequence.


## Related Roadmap Items

| ID | Item | Track | Horizon | Status | Gate |
| -- | --- | --- | --- | --- | --- |
| RM-001 | Complete canonical docs and assistant contract sync | Docs | Now | Complete | zig build gendocs -- --check --no-wasm --untracked-md ; zig build check-docs |
| RM-005 | Docs v3 pipeline baseline established | Docs | Now | Done | zig build check-docs |
| RM-011 | Launch developer education track | Docs | Later | Planned | zig build check-docs |

Roadmap guide: [../roadmap/](../roadmap/)



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
