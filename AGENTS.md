# Repository Guidelines

## Project Structure & Module Organization
`src/` hosts the runtime (`framework/`), feature modules (including `features/gpu/` kernels), and CLI entrypoints. `tests/` mirrors those packages with focused suites; integration matrices for OS targets live under `tests/cross-platform/`. Static documentation and generated API references resolve to `docs/`, while `docs/api/` is overwritten by the generators. Automation lives in `tools/` (Zig helpers) and `scripts/` (bash/PowerShell) and should be updated alongside new targets.

## Build, Test, and Development Commands
Run `zig build` for a debug binary in `zig-out/bin/abi`. `zig build test` executes the curated suite configured in `build.zig`. `zig build docs` orchestrates both docs generators and refreshes `docs/`. For incremental API docs during authoring, call `zig run tools/docs_generator.zig`. Performance checks land behind `zig build bench-all`; pass `-Doptimize=ReleaseFast` when validating release builds. Always honor `.zigversion` (Zig 0.16.0-dev); CI mirrors that configuration via `.github/workflows/ci.yml`.

## Coding Style & Naming Conventions
Source files are formatted with `zig fmt src tests` (four-space indentation). Modules and files prefer snake_case, public types stay PascalCase, and compile-time constants use UPPER_SNAKE. Keep GPU kernels under `src/features/gpu/` with suffix `_kernels.zig` and pair helpers in `shared/`. Avoid introducing lint tools outside the Zig toolchain without discussion.

## Testing Guidelines
Extend or add tests under `tests/` using `std.testing`. Name new files `test_<domain>.zig` and ensure exported test blocks describe behavior, e.g., `test "gpu matmul handles strides"`. Top-level execution stays `zig build test`; for GPU smoke runs, set `ABI_GPU_SMOKE=1 zig build test --test-filter "gpu_ai_acceleration"`. Include regression cases whenever touching `framework/` allocators or GPU memory paths.

## Commit & Pull Request Guidelines
Commit subjects follow concise Title Case (see history: "Align docs workflow Zig version"). Reference tickets in the body when applicable and group related changes per commit. Pull requests must list validation commands, note whether docs were regenerated, and link screenshots or profiles for UI or performance shifts. Update documentation or comments whenever public APIs change.

## Documentation & Pages
Docs in `docs/` must stay static assets compatible with GitHub Pages. Before pushing, run `zig build docs` and spot-check `docs/index.html` plus key `.js` bundles in `docs/assets/js/`. The `deploy_docs.yml` workflow publishes from `docs/`, so never commit partially generated artifacts.
