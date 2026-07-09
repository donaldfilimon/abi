# modern-refactor

Portable agent toolkit for high-quality, "from scratch" style codebase refactors. Applies clean-slate modern design thinking to legacy code while preserving correct behavior.

Usable from Claude Code, OpenCode, Grok/Codex-style agent hosts, or any workflow that loads `agents/` + `skills/` markdown. In this repo, align with `AGENTS.md` / `CLAUDE.md` / `GEMINI.md` and always run `./build.sh check` before and after substantive changes.

## Purpose

Traditional refactors often patch on top of legacy decisions. This toolkit helps you plan and execute as if writing the module for the first time with today's idioms — without inventing capabilities the source does not prove.

## Features

- **5 specialized skills** for the full modernization workflow (each with `references/` and, where useful, `examples/`)
- **2 autonomous agents** for deep planning and execution
- **Repo-native gates** — prefer `./build.sh check`, contract tests, and claims discipline over ad-hoc checklists alone
- Follows progressive-disclosure skill layout (`SKILL.md` + optional references/examples)

**Packaging notes:**

- PreToolUse hooks live in host config (e.g. root `.claude/` / `.codex/`), not under `modern-refactor/`
- Optional host settings template: `.claude/modern-refactor.local.md.example` (copy into the host project; not auto-loaded from the plugin package)

## Installation (local testing)

```bash
# Claude Code example
claude --plugin-dir ./modern-refactor
```

Or point your agent host at this directory / copy skills into your skills path.

## Quick Start

> **In this repo**: Use for `@docs`, `@tools`, `@modern-refactor`, or any `src/features/*` / plugin. Always run `./build.sh check` before/after. Keep instruction-file siblings in sync when conventions change.

### Trigger Skills

- "Plan a clean-slate refactor of docs/ using modern-refactor"
- "Analyze tools/build.sh for modernization opportunities"
- "Apply modern patterns to modern-refactor/skills/..."
- "Validate this modernization against the checklist"

### Use Agents

- `refactor-planner` for roadmap + milestones
- `modern-refactorer` for execution

### Optional host settings

See `.claude/modern-refactor.local.md.example`. Copy to the host project path your agent supports (for example `.claude/modern-refactor.local.md`) and add `*.local.md` to project `.gitignore` if those files hold local preferences.

## Component Overview

| Component Type | Count | Purpose |
|----------------|-------|---------|
| Skills         | 5     | refactor-strategy, modern-patterns, codebase-analysis, refactor-implementation, refactor-validation |
| Agents         | 2     | refactor-planner, modern-refactorer |
| Hooks          | 0 in-tree | Use host hooks if desired; not advertised as packaged here |
| MCP            | 0     | Not required for core workflow |
| Settings       | Host-side optional | Template in Quick Start above |

## Skills Details

- **refactor-strategy**: Clean-slate planning, strategy selection (direct / phased / parallel), risk assessment.
- **modern-patterns**: Concrete before/after modern idioms for errors, types, modularity.
- **codebase-analysis**: Systematic discovery of legacy patterns and high-value targets (`references/analysis-checklist.md`).
- **refactor-implementation**: Safe transformation techniques (`references/implementation-playbook.md`).
- **refactor-validation**: Parity + modern quality criteria (`references/validation-checklist.md`).

## Agents Details

See the individual `.md` files in `agents/` for full system prompts and triggering examples.

## Validation Performed

- Valid JSON manifest (`.claude-plugin/plugin.json`)
- Skills have frontmatter (name + third-person description with triggers)
- Skill `Resources` paths resolve to real files under `skills/*/references/` (and examples where referenced)
- Agents follow structured prompts under `agents/`
- Layout matches what this README claims (no missing advertised assets)

## License

MIT
