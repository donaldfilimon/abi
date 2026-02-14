# Repository Guidelines

## Project Structure & Module Organization

ABI is a Zig framework with `src/abi.zig` as the public API entrypoint. Core implementation is split across:

- `src/api/`, `src/core/`, `src/services/`, and `src/features/` for primary code.
- Feature areas: `src/features/{ai,analytics,auth,cache,cloud,database,gateway,gpu,messaging,mobile,network,observability,search,storage,web,...}`.
- Shared runtime and integration tests: `src/services/tests/`.
- Additional tests: `tests/`.
- Examples/docs: `examples/`, `benchmarks/`, `docs/api/`.

## Build, Test, and Development Commands

- `zig build`: build with default feature flags.
- `zig build run -- --help`: show CLI help.
- `zig build run -- plugins list`: list available plugins.
- `zig build test --summary all`: run the full test suite.
- `zig test src/path/to/file.zig --test-filter "pattern"`: run focused tests.
- `zig build validate-flags`: validate feature-flag combinations.
- `zig build cli-tests`: run CLI smoke tests (top-level and nested commands).
- `zig build full-check`: local quality gate (format + tests + flags + CLI smoke).
- `zig fmt .`: format sources before PR review.
- `zig build lint`: formatting/lint check used in CI.

## Coding Style & Naming Conventions

- Use Zig `0.16.0-dev.2535+b5bd49460` or newer.
- Indentation: 4 spaces, no tabs.
- Naming: `PascalCase` types, `camelCase` functions/variables.
- Prefer explicit imports (`@import(...)`) and avoid `usingnamespace`.
- Prefer `defer`/`errdefer` for deterministic cleanup.
- Prefer `std.ArrayListUnmanaged(T).empty` patterns where appropriate.

## Testing Guidelines

- Unit tests should live next to implementation in `*_test.zig`.
- Integration/stress/parity tests go in `src/services/tests/`.
- Hardware-dependent tests must skip with `error.SkipZigTest`.
- Test names should be descriptive (e.g., `test "Feature ..."`).
- Run focused tests for changed areas first, then broaden as time permits.

## Commit & Pull Request Guidelines

- Use conventional commit style (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- PRs should include:
  - clear summary of behavior changes,
  - linked issue/PR context when applicable,
  - commands run and outcomes (especially tests),
  - docs updates for API or feature-flag changes.

## Security & Configuration Notes

- Do not hardcode secrets; pass credentials via environment variables.
- When editing feature-gated modules, validate both paths:
  - `zig build -Denable-<feature>=true`
  - `zig build -Denable-<feature>=false`
