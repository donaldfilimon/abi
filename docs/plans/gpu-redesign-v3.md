---
title: GPU Redesign v3
description: Generated implementation plan
---

# GPU Redesign v3
## Status
- Status: **In Progress**
- Owner: **Abbey**
- Canonical metadata source: `src/services/tasks/roadmap_catalog.zig`
- Active execution tracker: `tasks/todo.md`

## Scope
Wave 3 active lane: enforce strict backend policy, pool lifecycle safety, and cross-target policy verification.

## Success Criteria
- Explicit backend requests fail fast instead of silently falling back.
- Pool lifecycle transitions remain safe under mixed-backend execution.
- Cross-target policy checks stay deterministic for Linux, Windows, and macOS targets.


## Validation Gates
- zig build typecheck
- zig build -Dtarget=x86_64-linux-gnu -Dgpu-backend=auto typecheck
- zig build -Dtarget=x86_64-windows-gnu -Dgpu-backend=auto typecheck
- zig build verify-all


## Milestones
- Wave 3A: finalize strict backend request handling across creation paths.
- Wave 3B: harden pool deinit/ownership rules for mixed backend graphs.
- Wave 3C: close remaining cross-target policy parity gaps and lock tests.


## Related Roadmap Items

| ID | Item | Track | Horizon | Status | Gate |
| -- | --- | --- | --- | --- | --- |
| RM-002 | Close GPU strictness and pool lifecycle gaps | GPU | Now | In Progress | zig build typecheck ; zig build verify-all |
| RM-006 | Automate cross-target GPU policy verification | Platform | Next | Planned | zig build -Dtarget=x86_64-linux-gnu -Dgpu-backend=auto typecheck ; zig build -Dtarget=x86_64-windows-gnu -Dgpu-backend=auto typecheck ; zig build -Dtarget=aarch64-macos -Dgpu-backend=auto typecheck |
| RM-010 | Hardware acceleration research track | Infrastructure | Later | Planned | zig build verify-all |

Roadmap guide: [../roadmap/](../roadmap/)



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use the `$zig-master` Codex skill for ABI Zig validation, docs generation, and build-wiring changes.
