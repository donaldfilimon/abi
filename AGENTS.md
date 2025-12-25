# Repository Guidelines

## Project Structure & Module Organization
Core sources live in `src/`. Public API is `src/abi.zig` and the root module is
`src/root.zig`. Major subsystems include `src/core/`, `src/compute/`,
`src/features/`, `src/framework/`, and `src/shared/`. A CLI-style entrypoint
exists at `src/main.zig`, but `zig build run` is only enabled when
`tools/cli/main.zig` is present. Tests live in `tests/` (`tests/mod.zig` for
smoke, `tests/phase5_integration.zig` for integration). Docs live in `docs/`.
Build config is `build.zig` and `build.zig.zon`, with outputs in `zig-out/` and
cache in `.zig-cache/`.

## Build, Test, and Development Commands
- `zig build` - Build the library and any configured executables.
- `zig build run -- --help` - Run the CLI if `tools/cli/main.zig` exists.
- `zig build test` - Run the full test suite.
- `zig build benchmark` - Run benchmarks when `src/compute/runtime/benchmark.zig` exists.
- `zig test tests/mod.zig` - Run smoke tests directly.
- `zig test src/compute/runtime/engine.zig` - Run tests from a single file.
- `zig test --test-filter="pattern"` - Run matching tests.
- `zig fmt .` or `zig fmt --check .` - Format or verify formatting.

## Coding Style & Naming Conventions
Use 4 spaces (no tabs) and keep lines under 100 characters; run `zig fmt .`
before commits. Types use PascalCase, functions and variables use snake_case,
and constants use UPPER_SNAKE_CASE. Import `std` first, then internal modules;
avoid `usingnamespace`. Error handling uses `!` return types with specific enums,
`try` for propagation, and `errdefer` for cleanup. Memory: use a stable
allocator (GPA) for long-lived data and worker arenas for scratch; never return
arena-backed results. Zig 0.16 APIs: use `cmpxchgStrong`/`cmpxchgWeak`,
`std.atomic.spinLoopHint()`, and `std.Thread.spawn(.{}, ...)`.

## Testing Guidelines
Place `test` blocks at file ends, use `testing.allocator`, and co-locate
`*_test.zig` where helpful. Update `tests/mod.zig` when adding new integration
coverage. New features should include tests or a short rationale in the PR.

## Commit & Pull Request Guidelines
Commit messages are short and imperative with optional scopes, for example
`feat(compute): ...`, `refactor(cli): ...`, or `docs: ...`. Keep commits focused.
PRs should describe the change, list tests run (for example `zig build test`),
and mention doc updates for public API changes.

## Configuration & Security Notes
Feature flags use `-Denable-*` options (AI, GPU, Web, Database, Network,
Profiling). Example:
```sh
zig build -Denable-ai=true -Denable-gpu=false -Denable-web=true
```
GPU backends: `-Dgpu-cuda`, `-Dgpu-vulkan`, `-Dgpu-metal`, `-Dgpu-webgpu`.
Keep secrets out of the repo; see `README.md` for connector environment
variables and `SECURITY.md` for reporting.
