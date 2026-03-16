---
title: AGENTS.md — Workflow Contract
purpose: Defines workflow contract for human and automated contributors
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# AGENTS.md — Contributor Workflow Contract

All contributors (human and automated) must follow this contract.

For build commands, architecture details, and API patterns see [CLAUDE.md](CLAUDE.md).
For codebase patterns and conventions see [docs/PATTERNS.md](docs/PATTERNS.md).
For directory layout see [docs/STRUCTURE.md](docs/STRUCTURE.md).

## Before You Start

1. Read `tasks/lessons.md` — prior corrections that prevent repeat mistakes
2. Check `tasks/todo.md` — current work in progress and planned items
3. Skim [CLAUDE.md](CLAUDE.md) for build commands and the feature flag list
4. If touching a feature module, review both `mod.zig` and `stub.zig` to understand the contract

## Coding Style

- `zig fmt` only — never manual alignment, never `zig fmt .` from repo root
- `lower_snake_case` for files/functions, `PascalCase` for types/error sets
- Relative imports within feature modules, `@import("abi")` for framework API
- Explicit `.zig` extensions on all path imports (Zig 0.16 requirement)
- Explicit error sets, propagate with `try`, never silently swallow

## Feature Module Contract

Every `src/features/<name>/` follows the **mod/stub/types** pattern:

- `mod.zig` — real implementation (feature enabled)
- `stub.zig` — API-compatible no-ops (feature disabled)
- `types.zig` — shared types imported by both

When changing `mod.zig` public signatures, update `stub.zig` immediately.
Sub-module stubs are not required.

## Commits

- Conventional commits: `fix:`, `feat:`, `docs:`, `chore:`, `style:`, `refactor:`
- Atomic scope — one logical change per commit
- PR descriptions must include validation results

## Workflow

1. Review `tasks/lessons.md` before starting work
2. Plan multi-file changes in `tasks/todo.md` before editing
3. Validate before completing (see verification gates below)
4. Verify mod/stub parity for any changed feature module
5. Update `tasks/lessons.md` after fixing any mistake that could recur

## Verification Gates

Run the strongest gate your environment supports:

| Gate | Command | When |
|------|---------|------|
| Format check | `zig fmt --check build.zig build/ src/ tools/` | Every change (always works) |
| Full check | `zig build full-check` | Before completing (non-Darwin) |
| Darwin fallback | `./tools/scripts/run_build.sh typecheck --summary all` | Darwin 25+ |
| Full release | `zig build verify-all` | Release prep |

On Darwin 25+, `zig build` fails at the linker stage. Use format checks and
`run_build.sh` for typecheck-only validation. See [CLAUDE.md](CLAUDE.md) for details.

## Documentation Changes

The `.gitignore` uses a markdown allowlist — `*.md` is ignored globally, with
explicit `!/path.md` entries for tracked files. When adding new documentation:

1. Create the file
2. Add `!/path/to/file.md` to the `.gitignore` allowlist
3. Verify `git status` shows the file as untracked (not ignored)

## Acceptance Criteria

A task is complete only when:

1. The strongest available verification gate passes
2. All touched `stub.zig` files match their `mod.zig` counterparts
3. `tasks/todo.md` is updated with completion evidence
4. No duplicative content introduced across governance docs
5. Any new `.md` files are added to the `.gitignore` allowlist
