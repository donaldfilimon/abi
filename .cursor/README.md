# Cursor configuration (root)

This directory holds Cursor-specific config for the ABI repo. Use it from **repo root**.

## Contents

| Path | Purpose |
|------|---------|
| **agents/** | Expert agents (e.g. Metal/CoreML GPU/NPU). See [agents/README.md](agents/README.md). |
| **plans/** | Cursor plan files (e.g. refactor plans). |

## Quick links

- **Full index (skills, plans, agents):** [CLAUDE.md — Skills, Plans, and Agents](../CLAUDE.md#skills-plans-and-agents-full-index)
- **Execution plans (phases, roles):** [plans/plan.md](../plans/plan.md)
- **Main assistant guide:** [CLAUDE.md](../CLAUDE.md)

## Reuse in other projects

To reuse in another repo:

1. Copy `.cursor/agents/` if you use Cursor agents; add one `.md` per agent with front matter (`name`, `description`).
2. Point agents and plans to your project’s main assistant doc (e.g. CLAUDE.md or README) for context.
3. Optionally add `.cursor/plans/` for Cursor-specific plan files.
