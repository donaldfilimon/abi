---
title: Plans
description: Generated active execution plans
---

# Plans
## Summary
Active generated plans: **4**. This index is generated from the canonical roadmap catalog and kept in sync with task import metadata.

## Active Plans
| Plan | Status | Owner | Scope |
| --- | --- | --- | --- |
| [CLI Framework + Local-Agent Fallback](./cli-framework-local-agents.md) | In Progress | Abbey | Wave 1 active lane: descriptor/runtime parity, local-first provider/plugin hardening, and command help/assertion drift cleanup. |
| [Docs + Assistant Canonical Sync](./docs-roadmap-sync-v2.md) | In Progress | Abbey | Canonical docs wave: align AGENTS.md, zig-master, tasks/todo.md, root status docs, and generated outputs around one workflow and Zig validation contract. |
| [GPU Redesign v3](./gpu-redesign-v3.md) | In Progress | Abbey | Wave 3 active lane: enforce strict backend policy, pool lifecycle safety, and cross-target policy verification. |
| [TUI Modular Extraction v2](./tui-modular-v2.md) | In Progress | Abbey | Wave 2 active lane: complete modular extraction, enforce layout/input correctness, and expand regression tests. |


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
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ensure pinned Zig matching `.zigversion` is on PATH. Format checks (`zig fmt --check ...`) always work as a fallback.
