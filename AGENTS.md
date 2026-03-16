# AGENTS.md -- Contributor Workflow Contract

This file is the workflow contract for all contributors (human and AI).
For codebase conventions, architecture, and API details, see [`CLAUDE.md`](CLAUDE.md).
For design patterns reference, see [`docs/PATTERNS.md`](docs/PATTERNS.md).
For directory layout, see [`docs/STRUCTURE.md`](docs/STRUCTURE.md).

---

## Before You Start

Complete this checklist before making any changes:

- [ ] Read [`tasks/lessons.md`](tasks/lessons.md) for recent corrections and pitfalls
- [ ] Check [`tasks/todo.md`](tasks/todo.md) for active work, blockers, and claimed tasks
- [ ] Understand the mod/stub contract: `stub.zig` must match `mod.zig` public signatures;
      shared types live in `types.zig` (see `CLAUDE.md` and `docs/PATTERNS.md`)
- [ ] Know your build command:
  - **Darwin 25+**: `./tools/scripts/run_build.sh <args>` (linker workaround)
  - **Linux / CI**: `zig build <args>`
  - **Format check (all platforms)**: `zig fmt --check build.zig build/ src/ tools/`

---

## Development Workflow

### 1. Plan

- Record multi-file changes in `tasks/todo.md` before editing code.
- Claim your task to avoid conflicts with parallel contributors.

### 2. Implement

- Follow coding conventions defined in `CLAUDE.md` (naming, imports, error handling).
- Use conventional commits with atomic scope:
  `fix:`, `feat:`, `refactor:`, `docs:`, `chore:`, `style:`, `test:`
- One logical change per commit. PR descriptions must include verification results.

### 3. Keep mod/stub Parity

- When changing any `src/features/<name>/mod.zig` public signature, update `stub.zig` to match.
- Verify: code must compile with both `feat_X=true` and `feat_X=false`.
- See `CLAUDE.md` for contract details (types.zig, StubFeature helpers).

### 4. Format and Validate

- Run `zig fmt --check build.zig build/ src/ tools/` before every commit.
- Never run `zig fmt .` from repo root (walks vendored fixtures).

### 5. Record Learnings

- After fixing any mistake that could recur, add an entry to `tasks/lessons.md`.
- Update `tasks/todo.md` with completion evidence.

---

## Verification Gates

All gates must pass before merging. On Darwin 25+, prefix build commands
with `./tools/scripts/run_build.sh` instead of `zig build`.

| Gate | Command | What it checks |
|------|---------|----------------|
| Formatting | `zig fmt --check build.zig build/ src/ tools/` | Style compliance |
| Flag combos | `zig build validate-flags` | 42 feature-flag combinations |
| CLI smoke | `zig build cli-tests` | CLI command registry and execution |
| Doc compile | `zig build gendocs` | Generated documentation builds |
| Doc validation | `zig build check-docs` | Doc content and link integrity |
| Full check | `zig build full-check` | Pre-commit aggregate gate |

For CI-only gates (Linux), `zig build full-check` and `zig build verify-all` are authoritative.

---

## Version Pin Discipline

When repinning Zig, update **all** of the following atomically in a single commit:

1. `.zigversion`
2. `build.zig.zon`
3. `baseline.zig`
4. `README.md`
5. CI configuration

Validate version/commit pairs against `ziglang.org/builds` artifact metadata,
not GitHub master HEAD. See `tasks/lessons.md` for past repinning issues.

---

## Feature Flag Rules

For flag inventory and syntax, see `CLAUDE.md`. Contributor-specific rules:

- When adding a flag, add it to **all** existing no-X validation entries in `build/flags.zig`.
- Required matrix shape: 2 baseline + N solo + N no-X.

---

## Acceptance Criteria

A task is complete only when:

1. All verification gates pass (or Darwin-equivalent fallback).
2. All touched `stub.zig` files match their `mod.zig` counterparts.
3. `tasks/todo.md` is updated with completion evidence.
4. `tasks/lessons.md` is updated if any correctable mistake occurred.
5. No content duplicated from `CLAUDE.md` or other governance docs.

---

## References

| Document | Purpose |
|----------|---------|
| [`CLAUDE.md`](CLAUDE.md) | Build commands, architecture, conventions, API changes |
| [`docs/PATTERNS.md`](docs/PATTERNS.md) | Design patterns (mod/stub, foundation module, build system) |
| [`docs/STRUCTURE.md`](docs/STRUCTURE.md) | Directory layout and module ownership |
| [`tasks/todo.md`](tasks/todo.md) | Active work queue and blockers |
| [`tasks/lessons.md`](tasks/lessons.md) | Correction log and pitfall reference |
