---
title: Docs + Roadmap Canonical Sync
description: Generated implementation plan
---

# Docs + Roadmap Canonical Sync
## Status
- Status: **In Progress**
- Owner: **Abbey**

## Scope
Canonical roadmap catalog, generated roadmap docs, generated plan docs, and task import synchronization.

## Success Criteria
- Roadmap and plans are generated from one catalog source.
- Task roadmap import reads canonical entries and skips done items.
- check-docs fails on roadmap/plans drift.


## Validation Gates
- zig build gendocs
- zig build check-docs
- zig build verify-all


## Milestones
- Introduce roadmap_catalog.zig with typed entries.
- Wire gendocs roadmap + plans renderers to canonical source.
- Enable plans drift checks and archive handling.


## Related Roadmap Items

| ID | Item | Track | Horizon | Status | Gate |
| -- | --- | --- | --- | --- | --- |
| RM-001 | Complete canonical roadmap/plans sync | Docs | Now | In Progress | zig build gendocs ; zig build check-docs |
| RM-005 | Docs v3 pipeline baseline established | Docs | Now | Done | zig build check-docs |
| RM-011 | Launch developer education track | Docs | Later | Planned | zig build check-docs |

Roadmap guide: [../roadmap/](../roadmap/)



---

*Generated automatically by `zig build gendocs`*


## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
