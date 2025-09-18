# Repository Guidelines

This repo is a Zig project with a Makefile wrapper. Source lives in `src/`, examples in `examples/`, benchmarks in `benchmarks/`, tests in `tests/`, and build artifacts in `zig-out/`. Build configuration is defined in `build.zig` and dependencies in `build.zig.zon`.

## Project Structure & Module Organization
- `src/` — library and app code; keep modules small and cohesive (e.g., `src/net/`, `src/util/`).
- `tests/` — unit and integration tests mirror `src/` layout (e.g., `tests/net/…`).
- `examples/` — minimal runnable samples demonstrating public APIs.
- `benchmarks/` — micro/throughput benchmarks; isolate external effects.
- `tools/` — helper scripts; keep cross‑platform where possible.

## Build, Test, and Development Commands
- `zig version` — confirm toolchain (also see `.zigversion`).
- `zig build` — default build; produces artifacts under `zig-out/`.
- `zig build test` — compile and run all tests.
- `zig build run` — run the default executable (if defined in `build.zig`).
- `zig build -Doptimize=ReleaseFast` — optimized build for benchmarks.
- `make` / `make test` — convenience wrappers (see `Makefile`).

## Coding Style & Naming Conventions
- Indentation: 4 spaces; no tabs. Line length ~100 chars.
- Zig style: prefer explicit types at public boundaries; use `const` where possible.
- Naming: `PascalCase` for types, `camelCase` for functions/vars, `SCREAMING_SNAKE_CASE` for compile‑time constants.
- Errors: return typed error sets; avoid `catch |e|` that masks context.
- Formatting: run `zig fmt .` before committing.

## Testing Guidelines
- Framework: Zig’s built‑in test runner (`test "name" { … }`).
- Layout: place tests beside code with `test` blocks or under `tests/` mirroring `src/`.
- Naming: describe behavior, e.g., `test "parser handles empty input" {}`.
- Coverage: add tests for new features and bug fixes; include edge cases and error paths.
- Run: `zig build test` (CI expects zero failures).

## Commit & Pull Request Guidelines
- Commits: present‑tense, scope-first messages, e.g., `net: fix timeout handling`.
- Keep changes focused; include rationale in the body when non‑obvious.
- PRs: include summary, linked issues, usage notes, and before/after benchmarks when performance‑related. Add repro steps for fixes and update `examples/` when APIs change.

## Security & Configuration Tips
- Avoid unchecked `@ptrCast`/`@intCast`; validate sizes and alignment.
- Prefer bounded operations; assert invariants in debug builds.
- Respect platform differences; gate OS‑specific code via `std.builtin.os`.

