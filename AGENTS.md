# Repository Guidelines

## Project Structure & Module Organization
ABI is a Zig framework with `src/abi.zig` as the public API root.
- Core code: `src/api/`, `src/core/`, `src/services/`, `src/features/`
- Feature modules: `src/features/{ai,analytics,auth,cache,cloud,database,gpu,messaging,mobile,network,observability,search,storage,web,...}`
- Shared runtime and tests: `src/services/tests/`
- Additional tests: `tests/`
- Examples and docs: `examples/`, `benchmarks/`, `docs/api/`

## Build, Test, and Development Commands
- `zig build` – build with default feature flags.
- `zig build run -- --help` – show CLI entry help.
- `zig build run -- plugins list` – list available plugins.
- `zig build test --summary all` – run the full test suite.
- `zig test src/path/to/file.zig --test-filter "pattern"` – run focused tests.
- `zig build validate-flags` – validates feature-flag combinations.
- `zig build cli-tests` – run CLI smoke tests.
- `zig build full-check` – run local gate (format + tests + flags + CLI smoke).
- `zig fmt .` – format source before opening PR.
- `zig build lint` – formatting/lint check used in CI.

## Coding Style & Naming Conventions
- Use Zig `0.16.0-dev.2535+b5bd49460` or newer.
- Indentation: 4 spaces, no tabs.
- Naming: `PascalCase` for types, `camelCase` for funcs/vars, `*Config` suffix for config structs.
- Prefer explicit imports (`@import(...)`) over `usingnamespace`.
- Prefer `defer`/`errdefer` for cleanup and specific error sets.
- Use `std.ArrayListUnmanaged(T).empty` style in code that follows this pattern.

## Testing Guidelines
- Unit tests belong beside implementation in `*_test.zig` files.
- Integration/stress/parity suites go under `src/services/tests/`.
- Hardware-dependent tests must skip cleanly with `error.SkipZigTest`.
- Naming should reflect intent (e.g., `test "Feature ..."`).
- Minimum PR check: at least run relevant focused tests and ideally `zig build test --summary all`.

## Commit & Pull Request Guidelines
- Use conventional commits (e.g., `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- PRs should include: clear summary, linked issue (when applicable), test commands run, and docs updates for API/flag changes.

## Security & Configuration Notes
- Do not hardcode secrets; pass credentials via environment variables.
- When touching feature-gated modules, validate both paths:
  - `zig build -Denable-<feature>=true`
  - `zig build -Denable-<feature>=false`
