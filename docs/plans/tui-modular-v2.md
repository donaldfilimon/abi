---
title: TUI Modular Extraction v2
description: Generated implementation plan
---

# TUI Modular Extraction v2
## Status
- Status: **In Progress**
- Owner: **Abbey**

## Scope
Split launcher/dashboard rendering into reusable modules with responsive layout and shared async loop behavior.

## Success Criteria
- Launcher execution path is unified across enter/search/mouse.
- Resize behavior is immediate and stable across panels.
- Small terminal fallback rendering remains readable.


## Validation Gates
- zig build cli-tests
- zig build run -- ui launch --help
- zig build run -- ui gpu --help


## Milestones
- Finalize launcher split modules and helpers.
- Migrate dashboards to shared layout/render primitives.
- Expand TUI unit tests for layout and hit-testing.


## Related Roadmap Items

| ID | Item | Track | Horizon | Status | Gate |
| -- | --- | --- | --- | --- | --- |
| RM-004 | Finish TUI modular extraction | CLI/TUI | Now | In Progress | zig build cli-tests ; zig build run -- ui launch --help |

Roadmap guide: [../roadmap/](../roadmap/)



---

*Generated automatically by `zig build gendocs`*


## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
