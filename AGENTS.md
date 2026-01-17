# Repository Guidelines

## Project Structure & Module Organization

- `src/` holds the core framework and public API. Key areas include `src/abi.zig`, `src/core/`, `src/compute/`, `src/features/`, `src/framework/`, `src/shared/`, and `src/tests/`.
- `tools/cli/` contains the CLI implementation.
- `benchmarks/` contains performance suites.
- `docs/` holds architecture and user documentation (e.g., `docs/intro.md`, `API_REFERENCE.md`).

## Build, Test, and Development Commands

- `zig build` builds the project.
- `zig build test --summary all` runs the full test suite with detailed output.
- `zig build run -- --help` runs the CLI and prints help.
- `zig fmt .` formats the codebase; `zig fmt --check .` verifies formatting without edits.
- Feature flags are passed via `-Denable-<feature>=true` (for example, `zig build -Denable-ai=true`).

## Coding Style & Naming Conventions

- Indentation: 4 spaces, no tabs; max line length 100 characters.
- Zig 0.16 conventions: use `std.Io`, `std.ArrayListUnmanaged`, explicit allocators, `{t}` formatting for enums/errors.
- No `anyerror`; define specific error sets.
- Resource cleanup: always use `defer`/`errdefer`.
- Naming: types in `PascalCase`, functions in `snake_case`, constants in `UPPER_SNAKE_CASE`, variables in `snake_case`. Struct fields put `allocator` first.

## Testing Guidelines

- Primary test command: `zig build test` or `zig build test --summary all`.
- Single file: `zig test src/compute/runtime/engine.zig`.
- Filtered tests: `zig test src/compute/runtime/engine.zig --test-filter "pattern"`.
- Keep tests aligned with feature flags; enable required features explicitly.

## Commit & Pull Request Guidelines

- Recent commits use short, imperative summaries (example: "Add GPU, SIMD, and database performance optimizations"). Follow that style.
- PRs should include a clear description, linked issues (if applicable), and the exact commands run for validation (e.g., `zig build test --summary all`).
- Update docs/examples when APIs or behavior change.

## Agent-Specific Instructions

- Keep changes minimal and consistent with existing patterns; avoid breaking public APIs unless requested.
- Preserve feature gating: stub modules must mirror real APIs and return `error.*Disabled`.
