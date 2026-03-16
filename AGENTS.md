# AGENTS.md — Workflow Contract

All contributors (human and automated) must follow this contract. See [CLAUDE.md](CLAUDE.md) for build commands and architecture details.

## Coding Style

- `zig fmt` only — never manual alignment, never `zig fmt .` from repo root
- `lower_snake_case` for files/functions, `PascalCase` for types/error sets
- Relative imports within feature modules, `@import("abi")` for framework API
- Explicit error sets, propagate with `try`, never silently swallow
- Every `src/features/<name>/mod.zig` must have a matching `stub.zig` with identical public signatures

## Commits

- Conventional commits: `fix:`, `feat:`, `docs:`, `chore:`, `style:`
- Atomic scope — one logical change per commit
- PR descriptions must include `zig build full-check` results

## Workflow

1. Review `tasks/lessons.md` before starting work
2. Plan multi-file changes in `tasks/todo.md` before editing
3. Validate before completing — `zig build full-check` (or `zig fmt --check` on Darwin 25+)
4. Verify mod/stub parity for any changed feature module
5. Update `tasks/lessons.md` after fixing any mistake that could recur

## Acceptance Criteria

A task is complete only when:
1. `zig build full-check` passes (or equivalent Darwin fallback)
2. All touched `stub.zig` files match their `mod.zig` counterparts
3. `tasks/todo.md` is updated with completion evidence
4. No duplicative content introduced across governance docs
