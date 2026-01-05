# Repository Guidelines

## Project Structure & Module Organization

- `src/` holds the library; key areas include `src/compute/`, `src/features/`,
  `src/framework/`, `src/shared/`, and `src/core/`.
- Public API entrypoints live in `src/abi.zig` and `src/root.zig`; the CLI
  entrypoint is `tools/cli/main.zig` (fallback: `src/main.zig`).
- `tests/` contains unit, integration, and property tests (see `tests/mod.zig`).
- `examples/`, `benchmarks/`, and `docs/` provide demos, performance runs, and
  documentation.
- Build metadata is in `build.zig` and `build.zig.zon`.

## Build, Test, and Development Commands

Zig 0.16.x is required.

```bash
zig build                                 # Build all modules
zig build run -- --help                   # Run the CLI
zig build test                            # Run the full test suite
zig build test --summary all              # Detailed test output
zig test src/compute/runtime/engine.zig   # Single-file tests
zig test --test-filter "engine init"      # Filtered test names
zig build benchmark                       # Benchmarks
zig fmt .                                 # Format code
zig fmt --check .                         # Format check
zig build -Denable-gpu=false -Denable-network=true  # Feature flags
```

## Coding Style & Naming Conventions

- 4 spaces, no tabs, max 100 chars, one blank line between functions.
- `//!` module docs, `///` public API docs with `@param`/`@return`.
- Types: PascalCase; functions/variables: snake_case; constants: UPPER_SNAKE_CASE.
- Allocator is the first field/arg when needed; prefer `std.ArrayListUnmanaged`
  for struct fields.
- Use explicit imports only; never `usingnamespace`. Prefer `defer`/`errdefer`
  for cleanup.

## Testing Guidelines

- Tests live in `tests/` and inline `test "..."` blocks in modules.
- Name tests descriptively, and add coverage for new features or note why not.
- Use feature flags to gate hardware-specific tests (e.g., `-Denable-gpu=true`).

## Commit & Pull Request Guidelines

- History favors short, imperative subjects; doc-only commits often use `docs:`.
- Required format: `<type>: <imperative summary>` with `feat`, `fix`, `docs`,
  `refactor`, `test`, `chore`, or `build`; keep summaries <= 72 chars.
- Keep commits focused and update docs when public APIs change.
- PRs should explain intent, link related issues if any, and list commands run
  (e.g., `zig build`, `zig build test`, `zig fmt .`).

## Architecture References

- System overview: `docs/intro.md`.
- API surface: `API_REFERENCE.md`.

## Configuration Notes

- Feature flags use `-Denable-*` and GPU backends use `-Dgpu-*` (see `README.md`).
- Connector credentials are provided via environment variables listed in
  `README.md`.
