# Lessons Learned

## 2026-03-06 - Aggressive-5 reprioritization must be canonical-first
- Root cause: Plan state drift appears when tasks/docs are edited before canonical roadmap catalog values are updated.
- Prevention rule: In one wave, update `roadmap_catalog.zig`, regenerate roadmap/plans data and markdown artifacts, then update `tasks/todo.md` and `tasks/lessons.md` before running close-out gates.

## 2026-03-06 - Pin + planning sync must move together
- Root cause: Zig pin and planning artifacts can drift when version updates are applied without the full contract set.
- Prevention rule: When repinning Zig, update `.zigversion`, `build.zig.zon`, `tools/scripts/baseline.zig`, `README.md`, and planning/generated artifacts in one wave.

## 2026-03-01 - Zig 0.16 ZON parsing ownership
- Root cause: `std.zon.parse.fromSliceAlloc` allocations were treated like wrapper-owned values instead of direct struct-owned slices.
- Prevention rule: Use arena-backed parsing for complex ZON inputs and deinit the arena at scope end.

## 2026-03-01 - Registry/docs extraction coupling
- Root cause: Tooling assumed direct imports and regex-based ZON rewrites after metadata moved to generated registry snapshots.
- Prevention rule: Resolve generated registry artifacts explicitly and keep deterministic parser paths for generated ZON.

## 2026-03-01 - Tool boundary discipline
- Root cause: Patch flow was attempted through generic shell execution instead of dedicated patch tooling.
- Prevention rule: Use dedicated patch/edit tools for file mutations and reserve shell for non-mutating inspection or command execution.

## 2026-03-06 - Workflow contract must be applied before implementation
- Root cause: Mandatory workflow rules were applied only after implementation work had already started, which created avoidable drift in consensus, task tracking, and review discipline.
- Prevention rule: For any non-trivial ABI task, review `tasks/lessons.md`, run the required multi-CLI consensus with a real prompt packet, and refresh `tasks/todo.md` before making repo-tracked edits.

## 2026-03-06 - Zig 0.16 API Breakages in Build System
- **Root cause**: Attempting to use older Zig 0.15/0.14 patterns in `build.zig` (e.g., `addOptions(options)`, `addTest(.{ .root_source_file = ... })`, `.path` in `LazyPath`).
- **Prevention rule**: 
  - For `addOptions`: Use manual field iteration or `createModule()` from an options step.
  - For `addTest` / `addExecutable`: Use the `root_module` field instead of top-level `root_source_file`.
  - For `LazyPath`: Use `.cwd_relative` or `.src_path` instead of the removed `.path` field for absolute or relative paths.
  - For Environment: Prefer `b.env_map.get()` (if initialized) or `std.process.getEnvVarOwned()` but verify the latest signature (e.g., `Init.Minimal` required for `main`).
  - For Darwin SDK: Force `b.sysroot` globally when building on macOS 26+ to bypass toolchain-internal linker failures.

## 2026-03-06 - Emergency Bootstrapping when Zig is fundamentally broken
- **Root cause**: Pre-built Zig toolchains can fail to link any binary (even the build runner) on futuristic Darwin environments (macOS 26+).
- **Prevention rule**: Provide a standalone C-based bootstrapper (`tools/scripts/emergency_bootstrap.c`) that can be compiled with `clang` to build a native Zig toolchain from source, bypassing the broken pre-built binary entirely.

## 2026-03-06 - Build runner links first; build.zig workarounds cannot fix it
- **Root cause**: When `zig build` fails with undefined symbols (e.g. `__availability_version_check`, `_arc4random_buf`) the first binary being linked is the **build runner** (the program that runs `build.zig`). That link happens before `build.zig` runs, so `use_llvm` / `use_lld` and other options set in `build.zig` do not apply to it.
- **Prevention rule**: If the failure is in the build runner, the only fix is to use a Zig built from source on the same host (e.g. `zig-bootstrap-emergency/./build aarch64-macos-none baseline`). Document this in CLAUDE.md and `docs/ZIG_MACOS_LINKER_RESEARCH.md`; set `use_llvm`/`use_lld` in build.zig for macOS 26+ anyway so that once the build runner links, our artifacts use the LLVM path.

## 2026-03-06 - Async Event Loop Polling in Zig 0.16
- **Root cause**: Busy-wait loops using `std.time.sleep` in TUI event handlers consume unnecessary CPU and degrade responsiveness compared to blocking I/O waits.
- **Prevention rule**: Use `std.posix.poll` (or equivalent platform-native non-blocking I/O multiplexing) on `std.posix.STDIN_FILENO` instead of `std.time.sleep` when waiting for input in asynchronous event loops.

## 2026-03-09 - Test manifest files must compile standalone
- **Root cause**: `src/services/lsp/client.zig` used cross-directory `@import("../../core/config/mod.zig")` and `@import("../shared/utils/zig_toolchain.zig")` which fail in standalone `zig test -fno-emit-bin` because the import paths go above the module root.
- **Prevention rule**: Files listed in `build/test_discovery.zig` must be self-contained. For small dependencies, inline the needed types/functions with a comment pointing to the canonical source. For larger dependencies, use build-system-provided modules via `addImport`. The Zig 0.16 `std.Io.Dir` API has no `makeDirAbsolute*` — use `createDirPath(.cwd(), io, path)` for recursive directory creation, `deleteTree(.cwd(), io, path)` for cleanup, and `file.writeStreamingAll(io, data)` instead of the removed `File.writeAll`.

## 2026-03-08 - CEL toolchain integration must be build-system native
- **Root cause**: The .cel toolchain was only accessible via shell scripts, making it invisible to the Zig build system and requiring manual PATH manipulation. Diagnostics were scattered.
- **Prevention rule**: When adding a toolchain variant or platform workaround, integrate it into `build.zig` as a first-class module (`build/cel.zig`) with dedicated build steps. Provide a diagnostics script (`cel_doctor.zig`) that follows the same patterns as `toolchain_doctor.zig`. Keep version consistency checks unified — `check_zig_version_consistency.zig` should validate all version sources including `.cel/config.sh`.

## 2026-03-09 - Zig nightly pins must come from artifact metadata, not GitHub master
- **Root cause**: Treating the current `ziglang/zig` `master` commit as interchangeable with the pinned nightly version drifted CEL to a source snapshot that did not match ABI's expected Zig 0.16-dev API level.
- **Prevention rule**: When repinning Zig nightlies, validate the version/commit pair against the actual `ziglang.org/builds` artifact metadata first. Only then update `.zigversion`, `.cel/config.sh`, `build.zig.zon`, `tools/scripts/baseline.zig`, and docs together.

## 2026-03-09 - Zig 0.16 removed usingnamespace; file splits need parameter-passing
- **Root cause**: Splitting large Zig files (e.g. GPU unified.zig, metal.zig) into submodules fails if submodules try to `@import` parent types circularly. Zig 0.16 removed `usingnamespace`.
- **Prevention rule**: When splitting large files into submodules, pass parent context as parameters to submodule init functions rather than circular imports. Keep the parent file as the orchestrator.

## 2026-03-09 - macOS linker prevents full verification on Darwin 25+
- **Root cause**: `zig build lint` and other build steps fail with undefined symbol errors (`_malloc_size`, `_nanosleep`, etc.) on macOS 25+ due to upstream Zig linker incompatibility.
- **Prevention rule**: On affected macOS versions, use `zig fmt --check` directly for format validation, or use the CEL toolchain. Don't block commits on `zig build lint` if the failure is the known linker issue.

## 2026-03-09 - lib.sh DRY pattern for shell scripts
- **Root cause**: Multiple CEL shell scripts (`.cel/build.sh`, `tools/scripts/cel_migrate.sh`, `tools/scripts/use_cel.sh`) duplicated the same functions for stock Zig detection, platform checks, and logging.
- **Prevention rule**: When multiple shell scripts share utility functions, extract them to a shared `lib.sh` (e.g. `.cel/lib.sh`) and `source` it. This reduces drift between scripts and keeps behavior consistent.

## 2026-03-09 - Sourcing bug with set -euo pipefail
- **Root cause**: `set -euo pipefail` at the top of a script breaks when the script is `source`'d into an interactive shell, because the strict error/unset settings leak into the caller's environment.
- **Prevention rule**: Guard strict mode with `if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then set -euo pipefail; fi` so it only applies when the script is executed directly, not sourced. Applied to `tools/scripts/use_cel.sh`.

## 2026-03-09 - SIGINT trap handler for long-running build scripts
- **Root cause**: Long-running build scripts (e.g. `.cel/build.sh` building Zig from source) leave partial state (incomplete cmake build dirs, half-written binaries) when interrupted with Ctrl-C.
- **Prevention rule**: Trap SIGINT/SIGTERM in long-running build scripts to clean up partial state before exiting. Applied to `.cel/build.sh`.

## 2026-03-09 - Comptime string formatting to reduce build step boilerplate
- **Root cause**: Build steps in `build/cel.zig` that differed only by a flag string had duplicated logic for constructing step names and descriptions.
- **Prevention rule**: Use Zig's `std.fmt.comptimePrint` to parameterize build step creation when steps differ only by a flag or name string. Applied to `build/cel.zig` `addCelShellStep`.

## 2026-03-10 - Vendored bootstrap fixtures must stay out of repo-root fmt runs
- Root cause: `zig-bootstrap-emergency/zig/test/cases/compile_errors/` vendors upstream Zig compile-error fixtures, so `zig fmt .` at repo root walks intentionally invalid sources and reports false-positive formatter failures.
- Prevention rule: Use the repo-safe format surface (`zig fmt build.zig build src tools examples` or `zig build lint` / `zig build fix`) and never use `zig fmt .` from the ABI repo root.

## 2026-03-10 - Manifest-driven feature tests must share one module graph
- Root cause: Creating one synthetic Zig module per `feature_test_manifest` entry caused duplicate file ownership when feature files imported each other, and the per-entry path materialization could degrade into malformed cache paths like `sfeatures/...`.
- Prevention rule: Generate one ignored feature-test root under `src/` and import manifest entries through that shared module graph instead of creating a separate module for each manifest entry.
