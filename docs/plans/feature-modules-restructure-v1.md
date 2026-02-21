---
title: Feature Modules Restructure v1
description: Generated implementation plan
---

# Feature Modules Restructure v1
## Status
- Status: **Planned**
- Owner: **Abbey**

## Scope
Consolidate feature layout, remove obsolete facades, and align mod/stub parity under new module boundaries.

## Success Criteria
- No stale imports to removed facade modules.
- Feature enable/disable builds pass with parity intact.
- Shared primitives are centralized under services/shared.


## Validation Gates
- zig build validate-flags
- zig build full-check


## Milestones
- Finish AI hierarchy consolidation.
- Complete shared resilience extraction.
- Update integration imports and feature test roots.


## Related Roadmap Items

| ID | Item | Track | Horizon | Status | Gate |
| -- | --- | --- | --- | --- | --- |
| RM-009 | Complete feature module hierarchy cleanup | Platform | Next | Planned | zig build validate-flags ; zig build full-check |
| RM-012 | Expand cloud function adapters | Platform | Later | Planned | zig build full-check ; zig build verify-all |

Roadmap guide: [../roadmap/](../roadmap/)



---

*Generated automatically by `zig build gendocs`*


## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
