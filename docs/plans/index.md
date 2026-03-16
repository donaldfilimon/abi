---
title: Plans
description: Generated active execution plans
---

# Plans
## Summary
Active generated plans: **6**. This index is generated from the canonical roadmap catalog and kept in sync with task import metadata. Last updated: 2026-03-16.

## Active Plans
| Plan | Status | Owner | Scope |
| --- | --- | --- | --- |
| [Feature Modules Restructure v1](./feature-modules-restructure-v1.md) | Complete | Abbey | 18-phase restructuring finished: persona to profile rename, bridge retirement, import compliance, zero mod/stub drift. |
| [Docs + Assistant Canonical Sync](./docs-roadmap-sync-v2.md) | Substantially Complete | Abbey | AGENTS.md, CLAUDE.md, tasks/todo.md, tasks/lessons.md all canonical. Remaining: CLAUDE.md/GEMINI.md wrapper consolidation. |
| [CLI Framework + Local-Agent Fallback](./cli-framework-local-agents.md) | In Progress (Steady-State) | Abbey | Descriptor-driven CLI working, provider health checks done. Remaining: strict-mode routing, help drift cleanup. |
| [GPU Redesign v3](./gpu-redesign-v3.md) | In Progress (Steady-State) | Abbey | Backend policy module complete, factory pattern with fallback. Remaining: strict fail-fast, pool lifecycle, cross-target tests. |
| [TUI Modular Extraction v2](./tui-modular-v2.md) | In Progress (Steady-State) | Abbey | Layout engine, launcher, dashboard extracted. Remaining: regression test coverage expansion. |
| [Integration Gates v1](./integration-gates-v1.md) | Blocked | Abbey | Blocked pending: matrix manifest, PTY timeout policy hardening, preflight diagnostics. |


## Roadmap Horizons

- Now: **7** item(s)
- Next: **2** item(s)
- Later: **3** item(s)

Roadmap guide: [../roadmap/](../roadmap/)


---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
