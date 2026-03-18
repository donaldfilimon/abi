---
title: GPU Redesign v3
description: Generated implementation plan
---

# GPU Redesign v3

> **Status Update (2026-03-18):** Superseded by 18-phase restructuring. Most milestones completed in commit 04002b79.

## Status
- Status: **Complete**
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
- [x] Wave 3A: finalize strict backend request handling across creation paths.
- [x] Wave 3B: harden pool deinit/ownership rules for mixed backend graphs.
- [x] Wave 3C: close remaining cross-target policy parity gaps and lock tests.


## Related Roadmap Items

| ID | Item | Track | Horizon | Status | Gate |
| -- | --- | --- | --- | --- | --- |
| RM-002 | Close GPU strictness and pool lifecycle gaps | GPU | Now | Complete | zig build typecheck ; zig build verify-all |
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
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
