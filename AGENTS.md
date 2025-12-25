# Repository Guidelines

## Project Structure & Module Organization
- `src/abi.zig` is the public API surface; `src/root.zig` is the root module.
- `src/features/` holds feature stacks (`ai`, `gpu`, `database`, `web`, `monitoring`,
  `connectors`, `network`).
- `src/compute/` contains runtime (`runtime/`), concurrency (`concurrency/`), and memory
  (`memory/`) modules.
- `src/shared/` provides shared logging, observability, platform, and utilities.
- `tools/cli/main.zig` is the primary CLI entrypoint (fallback: `src/main.zig`).
- `tests/` contains integration/unit tests; `docs/` holds documentation.
- `build.zig` and `build.zig.zon` define the build graph and options.

## Build, Test, and Development Commands
- `zig build` builds the library and CLI.
- `zig build test` runs the full test suite.
- `zig test tests/mod.zig` runs smoke tests directly.
- `zig test src/compute/runtime/engine.zig` runs a single file's tests.
- `zig test --test-filter="pattern"` runs matching tests.
- `zig build run -- --help` runs the CLI help.
- `zig build benchmark` runs benchmarks (if present).
- `zig fmt .` formats code; `zig fmt --check .` verifies formatting.

## Coding Style & Naming Conventions
- Zig 0.16.x; 4-space indentation, 100-character lines; run `zig fmt`.
- Naming: `PascalCase` for types, `snake_case` for functions/vars, `UPPER_SNAKE_CASE`
  for constants.
- Imports: `std` first, then internal; no `usingnamespace`, prefer qualified access.
- Errors & memory: return `!` with specific error sets; use `try`/`errdefer`. Use the
  stable allocator for long-lived data and worker arenas only for scratch; reset arenas,
  don't destroy mid-session.
- Docs: module docs use `//!`, function docs use `///`, examples in ```zig blocks.

## Testing Guidelines
- Use Zig `test` blocks at file end; co-locate `*_test.zig` and re-export via
  `tests/mod.zig`.
- Prefer `std.testing.allocator`.
- Add tests for new behavior and run `zig build test` before opening a PR.

## Configuration & Feature Flags
- Defaults: `-Denable-gpu=true`, `-Denable-ai=true`, `-Denable-web=true`,
  `-Denable-database=true`, `-Denable-network=false`, `-Denable-profiling=false`.
- GPU backends: `-Dgpu-cuda`, `-Dgpu-vulkan`, `-Dgpu-metal`, `-Dgpu-webgpu`.
- Additional backends: `-Dgpu-opengl`, `-Dgpu-opengles`, `-Dgpu-webgl2`.
- Example: `zig build -Denable-network=true -Dgpu-vulkan=true`.

## Commit & Pull Request Guidelines
- Use short, imperative subjects; optional conventional scopes (e.g., `refactor(cli): ...`).
- PRs should include: a concise summary, test command(s) run, and linked issues if any.
  Add sample CLI output when behavior changes.
