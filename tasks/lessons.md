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
