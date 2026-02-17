---
title: Roadmap
description: ABI development milestones and quality metrics
---

# Roadmap

For AI assistants: high-level milestones only. Phased execution and agent roles: [plans/plan.md](../plans/plan.md). Skills, plans, and agents index: [CLAUDE.md — Skills, Plans, and Agents](../CLAUDE.md#skills-plans-and-agents-full-index).

![Tests-1270%2F1275_(5_skip)](https://img.shields.io/badge/Tests-1270%2F1275_(5_skip)-brightgreen)
![Feature_Tests-1535%2F1535](https://img.shields.io/badge/Feature_Tests-1535%2F1535-brightgreen)

## Current (v0.4.0)

Built with Zig 0.16.0-dev.2611+f996d2866 (pinned in `.zigversion`).

- 21 feature modules with comptime gating
- 10 GPU backends (vtable abstraction)
- 9 LLM provider connectors
- 28 CLI commands + 8 aliases
- MCP/ACP server infrastructure
- vNext migration surface (staged)

## Next

- Complete vNext API (`.feature()`, `.has()`, `.state()` methods)
- GPU/database Zig 0.16 migration pass (37 backend compile errors)
- Programmatic test count validation in build.zig
