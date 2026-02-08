# Repository Guidelines

For deeper architecture notes, see `CLAUDE.md`. For vulnerability reporting and security
practices, see `SECURITY.md`.

**AI agents**: Claude Code (Cursor) and Codex share `.cursorrules` and these docs for
consistent behavior.

## Project Structure & Module Organization
ABI is a Zig 0.16 framework with `src/abi.zig` as the public API root. Core layout:
- `src/api/`: executable entry points (`main.zig`).
- `src/core/`: framework orchestration and config.
- `src/features/`: feature modules (`ai`, `gpu`, `database`, `network`, `web`, etc.).
- `src/services/`: shared runtime/platform infrastructure and test suites.
- `src/services/tests/`: integration, parity, and stress tests.
- `tests/`: additional top-level test harnesses.
- `examples/`, `benchmarks/`, `bindings/`, `docs/`: samples, performance, language bindings, and docs.

Import public APIs via `@import("abi")` rather than deep file paths. For feature-gated modules, keep `mod.zig` and `stub.zig` signatures aligned.

## v2 Module Integration Notes
- Shared v2 utilities are rooted at `src/services/shared/utils/`:
  `v2_primitives.zig`, `structured_error.zig`, `swiss_map.zig`,
  `abix_serialize.zig`, `profiler.zig`, `benchmark.zig`.
- Shared v2 memory primitives are rooted at `src/services/shared/utils/memory/`:
  `arena_pool.zig`, `combinators.zig`.
- Runtime v2 primitives are rooted at:
  - `src/services/runtime/concurrency/channel.zig`
  - `src/services/runtime/scheduling/thread_pool.zig`
  - `src/services/runtime/scheduling/dag_pipeline.zig`
- Prefer re-exported public paths when possible:
  - `abi.shared.utils.v2_primitives`, `abi.shared.utils.swiss_map`
  - `abi.runtime.Channel`, `abi.runtime.ThreadPool`, `abi.runtime.DagPipeline`
- Avoid cross-module deep imports from feature code into `src/services/**`.
  Depend on `@import("abi")` namespaced exports at module boundaries.

## Build, Test, and Development Commands
- `zig build`: build with default feature flags.
- `zig build run -- plugins list`: list ABI plugins.
- `zig build run -- --help`: run CLI entry point.
- `zig build test --summary all`: run full test suite.
- `zig test src/path/to/file.zig --test-filter "pattern"`: run focused tests.
- `zig build validate-flags`: verify feature-flag combinations compile.
- `zig build cli-tests`: CLI smoke tests.
- `zig build full-check`: full local gate (format + tests + flag validation + CLI smoke tests).
- `zig fmt .`: format source.
- `zig build lint`: CI formatting check.

## Coding Style & Naming Conventions
- Use Zig `0.16.0-dev.2471+e9eadee00` or newer.
- Indentation: 4 spaces, no tabs; keep lines under 100 chars.
- Naming: `PascalCase` for types, `camelCase` for functions/variables, `*Config` for config structs.
- Prefer explicit imports (no `usingnamespace`), specific error sets, and `defer`/`errdefer` for cleanup.
- Prefer `std.ArrayListUnmanaged(T).empty` patterns used across this codebase.

## Testing Guidelines
- Unit tests should live alongside implementation files as `*_test.zig`.
- Integration/stress/parity suites live under `src/services/tests/`.
- Hardware-gated tests should skip cleanly with `error.SkipZigTest` when prerequisites are unavailable.
- Before opening a PR, run at minimum `zig fmt .` and `zig build test --summary all`.

## Commit & Pull Request Guidelines
- Follow Conventional Commit prefixes seen in history: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Keep commits focused and avoid mixing refactors with behavior changes.
- PRs should include a clear summary, linked issue (if applicable), tests run, and documentation updates for API/flag changes.

## Security & Configuration Tips
- Never hardcode secrets; pass credentials via environment variables.
- When touching feature-gated code, validate both paths:
  - `zig build -Denable-<feature>=true`
  - `zig build -Denable-<feature>=false`
