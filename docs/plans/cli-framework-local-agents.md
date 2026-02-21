---
title: CLI Framework + Local-Agent Fallback
description: Generated implementation plan
---

# CLI Framework + Local-Agent Fallback
## Status
- Status: **In Progress**
- Owner: **Abbey**

## Scope
Descriptor-driven CLI framework and local-first LLM provider routing with plugin support.

## Success Criteria
- LLM command family runs through provider router.
- Fallback chain is deterministic and configurable.
- CLI command metadata and runtime dispatch share one source.


## Validation Gates
- zig build cli-tests
- zig build feature-tests
- zig build verify-all


## Milestones
- Use canonical command catalog as metadata source for descriptors/spec/matrix.
- Finalize llm run/session/providers/plugins command tree.
- Harden provider health checks and strict backend mode.
- Align TUI command preview with descriptor graph.


## Related Roadmap Items

| ID | Item | Track | Horizon | Status | Gate |
| -- | --- | --- | --- | --- | --- |
| RM-003 | Finalize CLI descriptor framework cutover | CLI/TUI | Now | In Progress | zig build cli-tests ; zig build verify-all |
| RM-008 | Harden local-agent provider plugins | AI | Next | Planned | zig build feature-tests ; zig build cli-tests |

Roadmap guide: [../roadmap/](../roadmap/)



---

*Generated automatically by `zig build gendocs`*


## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
