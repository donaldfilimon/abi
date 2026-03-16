---
title: Plans
description: Generated active execution plans
---

# Plans
## Summary
Active generated plans: **6**. This index is generated from the canonical roadmap catalog and kept in sync with task import metadata.

## Active Plans
| Plan | Status | Owner | Scope |
| --- | --- | --- | --- |
| [CLI Framework + Local-Agent Fallback](./cli-framework-local-agents.md) | In Progress | Abbey | Wave 1 active lane: descriptor/runtime parity, local-first provider/plugin hardening, and command help/assertion drift cleanup. |
| [Docs + Assistant Canonical Sync](./docs-roadmap-sync-v2.md) | In Progress | Abbey | Canonical docs wave: align AGENTS.md, zig-master, tasks/todo.md, root status docs, and generated outputs around one workflow and Zig validation contract. |
| [Feature Modules Restructure v1](./feature-modules-restructure-v1.md) | In Progress | Abbey | Wave 5 active lane: remove legacy facades, finalize module boundaries, and consolidate shared primitives. |
| [GPU Redesign v3](./gpu-redesign-v3.md) | In Progress | Abbey | Wave 3 active lane: enforce strict backend policy, pool lifecycle safety, and cross-target policy verification. |
| [TUI Modular Extraction v2](./tui-modular-v2.md) | In Progress | Abbey | Wave 2 active lane: complete modular extraction, enforce layout/input correctness, and expand regression tests. |
| [Integration Gates v1](./integration-gates-v1.md) | Blocked | Abbey | Wave 4 blocked lane: restore exhaustive integration gates after explicit unblock criteria are met while keeping interim gate policy green. |


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
Use `zig build full-check` on supported hosts. On Darwin 25+ / 26+, use `zig fmt --check ...` plus `./tools/scripts/run_build.sh <step>`. For docs generation, use `zig build gendocs` or `./tools/scripts/run_build.sh gendocs` on Darwin.
