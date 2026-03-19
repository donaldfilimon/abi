---
title: AGENTS.md — Workflow Contract
purpose: Defines workflow contract for human and automated contributors
last_updated: 2026-03-18
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# AGENTS.md — Contributor Workflow Contract

All contributors (human and automated) must follow this contract.

For build commands, architecture, import rules, API patterns, and troubleshooting see [CLAUDE.md](CLAUDE.md).
For codebase patterns and conventions see [docs/PATTERNS.md](docs/PATTERNS.md).
For directory layout see [docs/STRUCTURE.md](docs/STRUCTURE.md).

## Before You Start

1. Read `tasks/lessons.md` — prior corrections that prevent repeat mistakes
2. Check `tasks/todo.md` — current work in progress and planned items
3. Skim [CLAUDE.md](CLAUDE.md) for build commands and the feature flag list
4. If touching a feature module, review both `mod.zig` and `stub.zig` to understand the contract

## Coding Style

See [CLAUDE.md — Conventions](CLAUDE.md#conventions) for the full list. Key rules:

- `zig fmt` only — never manual alignment, never `zig fmt .` from repo root
- `lower_snake_case` for files/functions, `PascalCase` for types/error sets
- Relative imports within `src/`, `@import("abi")` from external code (CLI, tests)
- Explicit `.zig` extensions on all path imports (Zig 0.16 requirement)
- Explicit error sets, propagate with `try`, never silently swallow

## Feature Module Contract

See [CLAUDE.md — mod/stub contract](CLAUDE.md#modstub-contract) for details. Summary:

- Every `src/features/<name>/` has `mod.zig` + `stub.zig` + `types.zig`
- When changing `mod.zig` public signatures, update `stub.zig` immediately
- CLI-accessed sub-modules must be re-exported from both mod and stub

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
6. Update `src/core/feature_catalog.zig` and regenerate artifacts before updating feature count references elsewhere
7. Version pin changes: update `.zigversion`, `build.zig.zon`, `baseline.zig`, `README.md`, CI config atomically

## Verification Gates

Run the strongest gate your environment supports — see [CLAUDE.md — Workflow](CLAUDE.md#workflow) for the gate table and Darwin 25+ details.

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
6. Feature catalog (`src/core/feature_catalog.zig`) is in sync with the actual feature set
