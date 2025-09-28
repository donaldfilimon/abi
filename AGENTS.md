# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/` (runtime in `src/framework/`; feature modules under `src/features/` such as `src/features/gpu/`).
- GPU kernels stay in `src/features/gpu/` with `_kernels.zig` suffix; shared helpers in `src/features/gpu/shared/`.
- CLI entrypoints are under `src/`.
- Tests mirror modules in `tests/`; cross‑platform matrices live in `tests/cross-platform/`.
- Static docs live in `docs/`; `docs/api/` is generated and should not be edited.
- Automation: `tools/` (Zig helpers) and `scripts/` (bash/PowerShell).

## Build, Test, and Development Commands
- Use Zig `0.16.0-dev` as pinned by `.zigversion` (verify with `zig version`).
- `zig build` → builds debug binary at `zig-out/bin/abi`.
- `zig build test` → runs the curated suite from `build.zig`.
- `zig build docs` → regenerates static site into `docs/`.
- `zig run tools/docs_generator.zig` → incremental API docs while authoring.
- `zig build bench-all -Doptimize=ReleaseFast` → performance checks in release mode.

## Coding Style & Naming Conventions
- Format with `zig fmt src tests` (four-space indentation) before committing.
- Filenames/modules: snake_case. Public types: PascalCase. Compile‑time constants: UPPER_SNAKE.
- Keep GPU kernels under `src/features/gpu/` and use `_kernels.zig` suffix.
- Avoid non-Zig linters; rely on Zig tooling.

## Testing Guidelines
- Write tests with `std.testing`; mirror source layout under `tests/`.
- Name files `test_<domain>.zig`; describe behavior, e.g., `test "gpu matmul handles strides"`.
- GPU smoke: `ABI_GPU_SMOKE=1 zig build test --test-filter "gpu_ai_acceleration"`.
- Add regression tests whenever touching `framework/` allocators or GPU memory paths.

## Commit & Pull Request Guidelines
- Commit subject: concise Title Case (e.g., "Align Docs Workflow Zig Version").
- Group related changes; reference tickets in bodies.
- PRs must list validation commands, note if docs were regenerated, and include screenshots or profiles for UI/perf changes.

## Documentation & Pages
- Run `zig build docs` before pushing; spot‑check `docs/index.html` and `docs/assets/js/`.
- `deploy_docs.yml` publishes from `docs/`; never commit partially generated artifacts.
