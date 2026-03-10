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

## 2026-03-06 - Blocked Darwin validation needs a non-link fallback
- **Root cause**: Pre-built Zig toolchains can fail to link any binary (even the build runner) on futuristic Darwin environments (macOS 26+).
- **Prevention rule**: When the build runner cannot link, fall back to `run_build.sh`, `zig fmt --check`, compile-only `zig test -fno-emit-bin`, and Linux CI instead of inventing repo-local toolchain bridges.

## 2026-03-06 - Build runner links first; build.zig workarounds cannot fix it
- **Root cause**: When `zig build` fails with undefined symbols (e.g. `__availability_version_check`, `_arc4random_buf`) the first binary being linked is the **build runner** (the program that runs `build.zig`). That link happens before `build.zig` runs, so `use_llvm` / `use_lld` and other options set in `build.zig` do not apply to it.
- **Prevention rule**: If the failure is in the build runner, document it as an environment blocker and switch to wrapper-based, compile-only, or Linux CI validation. `build.zig` knobs alone cannot repair the build runner link.

## 2026-03-06 - Async Event Loop Polling in Zig 0.16
- **Root cause**: Busy-wait loops using `std.time.sleep` in TUI event handlers consume unnecessary CPU and degrade responsiveness compared to blocking I/O waits.
- **Prevention rule**: Use `std.posix.poll` (or equivalent platform-native non-blocking I/O multiplexing) on `std.posix.STDIN_FILENO` instead of `std.time.sleep` when waiting for input in asynchronous event loops.

## 2026-03-09 - Test manifest files must compile standalone
- **Root cause**: `src/services/lsp/client.zig` used cross-directory `@import("../../core/config/mod.zig")` and `@import("../shared/utils/zig_toolchain.zig")` which fail in standalone `zig test -fno-emit-bin` because the import paths go above the module root.
- **Prevention rule**: Files listed in `build/test_discovery.zig` must be self-contained. For small dependencies, inline the needed types/functions with a comment pointing to the canonical source. For larger dependencies, use build-system-provided modules via `addImport`. The Zig 0.16 `std.Io.Dir` API has no `makeDirAbsolute*` — use `createDirPath(.cwd(), io, path)` for recursive directory creation, `deleteTree(.cwd(), io, path)` for cleanup, and `file.writeStreamingAll(io, data)` instead of the removed `File.writeAll`.

## 2026-03-08 - Repo-local toolchain bridges create long-term coupling
- **Root cause**: Temporary repo-local toolchain workarounds sprawled across build wiring, shell helpers, docs, and diagnostics until they became part of the public workflow by accident.
- **Prevention rule**: Keep supported Zig resolution limited to the pinned Zig on PATH or ZVM plus documented wrapper/compile-only fallbacks. Do not add repo-local toolchain surfaces unless they are intended to remain public and permanent.

## 2026-03-09 - Zig nightly pins must come from artifact metadata, not GitHub master
- **Root cause**: Treating the current `ziglang/zig` `master` commit as interchangeable with the pinned nightly version drifted local tooling to a source snapshot that did not match ABI's expected Zig 0.16-dev API level.
- **Prevention rule**: When repinning Zig nightlies, validate the version/commit pair against the actual `ziglang.org/builds` artifact metadata first. Only then update `.zigversion`, `build.zig.zon`, `tools/scripts/baseline.zig`, and docs together.

## 2026-03-09 - Zig 0.16 removed usingnamespace; file splits need parameter-passing
- **Root cause**: Splitting large Zig files (e.g. GPU unified.zig, metal.zig) into submodules fails if submodules try to `@import` parent types circularly. Zig 0.16 removed `usingnamespace`.
- **Prevention rule**: When splitting large files into submodules, pass parent context as parameters to submodule init functions rather than circular imports. Keep the parent file as the orchestrator.

## 2026-03-09 - macOS linker prevents full verification on Darwin 25+
- **Root cause**: `zig build lint` and other build steps fail with undefined symbol errors (`_malloc_size`, `_nanosleep`, etc.) on macOS 25+ due to upstream Zig linker incompatibility.
- **Prevention rule**: On affected macOS versions, use `zig fmt --check`, `run_build.sh`, and compile-only validation locally. Don't block commits on `zig build lint` if the failure is the known linker issue.

## 2026-03-09 - lib.sh DRY pattern for shell scripts
- **Root cause**: Multiple shell scripts duplicated the same functions for Zig detection, platform checks, and logging.
- **Prevention rule**: When multiple shell scripts share utility functions, extract them to a shared helper and `source` it. This reduces drift between scripts and keeps behavior consistent.

## 2026-03-09 - Sourcing bug with set -euo pipefail
- **Root cause**: `set -euo pipefail` at the top of a script breaks when the script is `source`'d into an interactive shell, because the strict error/unset settings leak into the caller's environment.
- **Prevention rule**: Guard strict mode with `if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then set -euo pipefail; fi` so it only applies when the script is executed directly, not sourced.

## 2026-03-09 - SIGINT trap handler for long-running build scripts
- **Root cause**: Long-running build scripts can leave partial state (incomplete build dirs, half-written binaries) when interrupted with Ctrl-C.
- **Prevention rule**: Trap SIGINT/SIGTERM in long-running build scripts to clean up partial state before exiting.

## 2026-03-09 - Comptime string formatting to reduce build step boilerplate
- **Root cause**: Build steps that differed only by a flag string had duplicated logic for constructing step names and descriptions.
- **Prevention rule**: Use Zig's `std.fmt.comptimePrint` to parameterize build step creation when steps differ only by a flag or name string.

## 2026-03-10 - Vendored bootstrap fixtures must stay out of repo-root fmt runs
- Root cause: `zig-bootstrap-emergency/zig/test/cases/compile_errors/` vendors upstream Zig compile-error fixtures, so `zig fmt .` at repo root walks intentionally invalid sources and reports false-positive formatter failures.
- Prevention rule: Use the repo-safe format surface (`zig fmt build.zig build src tools examples` or `zig build lint` / `zig build fix`) and never use `zig fmt .` from the ABI repo root.

## 2026-03-10 - Manifest-driven feature tests must share one module graph
- Root cause: Creating one synthetic Zig module per `feature_test_manifest` entry caused duplicate file ownership when feature files imported each other, and the per-entry path materialization could degrade into malformed cache paths like `sfeatures/...`.
- Prevention rule: Generate one ignored feature-test root under `src/` and import manifest entries through that shared module graph instead of creating a separate module for each manifest entry.

## 2026-03-10 - Bulk find-and-replace can corrupt string literals across files
- Root cause: A bulk operation that stripped the word "zig" from file content also removed it from inside string literals (`@import("...zig")`, `"zig"` comparisons, `"which -a zig"` commands). The displaced `")` characters appeared as stray suffixes on nearby expression lines.
- Prevention rule: Never run bulk find-and-replace on source code without excluding string literal interiors. After any bulk text operation, run `zig fmt --check` immediately to catch truncated string literals (they show as "invalid byte: '\n'" errors). Always verify with format check before committing.

## 2026-03-10 - Corruption patterns cascade: one bulk operation creates multiple fix waves
- Root cause: The initial "zig" stripping corruption was fixed in 33 files, but 33 more files had the same pattern in different directories (services/, tools/, gpu/). A third wave found 22 more in doc comments and file path strings.
- Prevention rule: After fixing bulk corruption, run `zig fmt --check build.zig build/ src/ tools/` immediately and count remaining parse errors — they indicate more files with the same pattern. Don't commit until parse errors reach 0. Search for all corruption patterns systematically (not just the first directory).

## 2026-03-10 - Mod/stub parity must be checked after migration
- Root cause: After migrating database from features/ to core/, the features/database/mod.zig facade only re-exported 7 sub-modules while stub.zig provided the full 58-item API. Code using `database.open()` would compile with feat_database=true but fail with feat_database=false.
- Prevention rule: After any feature module migration, run a mod↔stub parity check to ensure both export identical public API surfaces. The stub-sync-validator agent in zig-abi-plugin can automate this.

## 2026-03-10 - Validation matrix no-X entries must enable ALL other features
- Root cause: 19 of 20 `no-X` entries in `build/flags.zig` validation_matrix were missing `.feat_mobile = true`, meaning they silently tested with mobile disabled — hiding potential mobile interaction bugs.
- Prevention rule: When adding a new feature flag, add it to ALL existing no-X entries (except no-<self>), not just the solo and no-self entries. Verify total count matches formula: 2 baseline + N solo + N no-X.
