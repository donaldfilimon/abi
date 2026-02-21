---
title: GPU Redesign v3
description: Generated implementation plan
---

# GPU Redesign v3
## Status
- Status: **In Progress**
- Owner: **Abbey**

## Scope
Metal/Vulkan policy hardening, GL family consolidation, strict backend creation, and mixed-backend stability.

## Success Criteria
- Explicit backend requests stay strict.
- Auto policy is deterministic per target.
- Mixed-backend execution remains stable under pool lifecycle checks.


## Validation Gates
- zig build typecheck
- zig build -Dtarget=x86_64-linux-gnu -Dgpu-backend=auto typecheck
- zig build -Dtarget=x86_64-windows-gnu -Dgpu-backend=auto typecheck
- zig build verify-all


## Milestones
- Complete backend registry/pool strictness enforcement.
- Finalize GL profile wrappers over shared runtime.
- Close cross-target compile and policy consistency gaps.


## Related Roadmap Items

| ID | Item | Track | Horizon | Status | Gate |
| -- | --- | --- | --- | --- | --- |
| RM-002 | Close GPU strictness and pool lifecycle gaps | GPU | Now | In Progress | zig build typecheck ; zig build verify-all |
| RM-006 | Automate cross-target GPU policy verification | Platform | Next | Planned | zig build -Dtarget=x86_64-linux-gnu -Dgpu-backend=auto typecheck ; zig build -Dtarget=x86_64-windows-gnu -Dgpu-backend=auto typecheck ; zig build -Dtarget=aarch64-macos -Dgpu-backend=auto typecheck |
| RM-010 | Hardware acceleration research track | Infrastructure | Later | Planned | zig build verify-all |

Roadmap guide: [../roadmap/](../roadmap/)



---

*Generated automatically by `zig build gendocs`*


## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
