# Lessons Learned

## Zig 0.16 API Changes
- `GeneralPurposeAllocator` → `DebugAllocator`. No `std.time.timestamp()` → use `unixSeconds()`. No `File.writeAll` → use `writeStreamingAll(io, data)`. No `makeDirAbsolute*` → use `createDirPath(.cwd(), io, path)`. No `usingnamespace` → pass parent context as parameters to submodule init functions.
- `LazyPath`: use `.cwd_relative`/`.src_path`, not `.path`. `addTest`/`addExecutable`: use `root_module`, not `root_source_file`. ZON parsing: use arena-backed `fromSliceAlloc`, deinit arena at scope end.
- **dev.2905+**: All `@import("path")` must have explicit `.zig` extensions. Single-module file ownership enforced — every `.zig` file belongs to exactly one named module. Cross-module relative-path imports are illegal. Solution: consolidate all `src/` into single `abi` module. The old `core` named module was removed entirely (files live in `src/core/` as part of `abi`). The old `shared_services` module was replaced by `foundation` (see below).
- `valueIterator()`/`keyIterator()` not `.values()`. `@enumFromInt(x)` not `intToEnum`. Use `std.posix.poll` on STDIN instead of `std.time.sleep` in event loops.

## Darwin 25+ Linker Workaround
- `zig build` fails with undefined symbols (`_malloc_size`, `_nanosleep`, etc.) because the **build runner** links first, before `build.zig` runs. No `build.zig` knob can fix this.
- Use `./tools/scripts/run_build.sh`, `zig fmt --check`, or `zig test -fno-emit-bin` locally. CI (Linux) is authoritative.
- LLD has zero Mach-O support — never `use_lld = true` on macOS targets.

## Version Pin Discipline
- When repinning Zig: update `.zigversion`, `build.zig.zon`, `baseline.zig`, `README.md`, CI config atomically. Validate version/commit pairs against `ziglang.org/builds` artifact metadata, not GitHub master HEAD.
- Update `roadmap_catalog.zig` and regenerate artifacts before updating `tasks/` files to prevent plan state drift.

## mod/stub Sync
- `stub.zig` must match `mod.zig` public signatures. After any feature migration, verify parity — code compiles with `feat_X=true` but fails with `feat_X=false` if stubs diverge.
- Validation matrix no-X entries must enable ALL other features. When adding a flag, add it to all existing no-X entries. Verify: 2 baseline + N solo + N no-X.
- Shared types go in `types.zig` — both `mod.zig` and `stub.zig` import from it. Use `StubFeature`/`StubFeatureNoConfig` from `core/stub_context.zig` for common stub boilerplate (-118 lines across 7 stubs).
- `ArrayListUnmanaged` and `AutoHashMapUnmanaged` must use `.empty` not `.{}` for initialization in Zig 0.16. The `.{}` literal triggers "missing struct field: items" errors.
- CLI tools accessing `abi.features.ai.<submodule>` will fail at compile time if the sub-module isn't re-exported from the AI stub. When adding new AI sub-modules accessed by CLI, add to both `mod.zig` AND `stub.zig`. Inline stubs need all methods the caller invokes — each returning `error.AiDisabled` or a safe default.
- Feature-gated sub-modules must not directly import other feature modules via relative paths (bypasses the gate). Use `build_options` conditional imports to match the caller's type path.

## Build System Patterns
- Files in `build/test_discovery.zig` must compile standalone with `zig test <file> -fno-emit-bin`. Cross-directory `@import("../../")` breaks this — inline small deps or use build-system modules.
- Use `std.fmt.comptimePrint` to parameterize build steps that differ only by a flag string. One shared module graph for manifest-driven tests, not per-entry modules.
- Tool-side Zig modules under `tools/` cannot reach into `../../build/*.zig` with relative imports. Pass shared build metadata as a named module import from `build.zig` instead.
- In `src/root.zig`, keep private module/type aliases distinct from public compatibility re-exports. Reusing the same identifier inside nested namespace structs creates ambiguous references under Zig master.
- Feature-test per-entry modules violate Zig 0.16 single-file ownership when entries share files through import graphs. Fix: use the `abi` module directly as the test root (`addTest(.{ .root_module = abi_module })`). The `feature_test_manifest` in `module_catalog.zig` is preserved as documentation.
- `@import("abi")` cannot be used within files that ARE part of the `abi` module — this creates a circular "no module named 'abi' available within module 'abi'" error. It only works from external modules (CLI, tests with separate roots) or lazy-evaluated code paths.

## `foundation` Namespace (Not a Separate Module)
- **What**: `src/services/shared/mod.zig` provides shared service types (allocators, logging, config). Exposed as `abi.foundation` via `pub const foundation = @import("services/shared/mod.zig")` in `src/root.zig`.
- **Architecture**: All files under `src/services/shared/` belong to the single `abi` module. There is no separate `foundation` named module — files are accessed via relative imports within the `abi` module graph.
- **Wiring**: `wireAbiImports(module, build_opts)` adds only `build_options` as a named import. The `foundation` namespace is wired through the normal `abi` module import graph, not as a named import.
- **Rule**: Files within the `abi` module should use relative paths to reach `src/services/shared/` (e.g., `@import("../../services/shared/mod.zig")`). External modules (CLI, tests with separate roots) access it through `@import("abi").foundation`.

## Bulk Operations Safety
- Never bulk find-replace without excluding string literal interiors. After any bulk text operation, run `zig fmt --check` immediately. Corruption cascades across multiple waves — don't commit until parse errors reach 0.

## Shell Script Patterns
- Guard `set -euo pipefail` with `if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then ... fi` so strict mode only applies when executed directly.
- Extract shared utility functions to a `lib.sh` to prevent drift. Trap SIGINT/SIGTERM in long-running scripts to clean up partial state.

## Tool & Workflow Discipline
- Use dedicated edit tools for file mutations, reserve shell for inspection. Review `tasks/lessons.md` and refresh `tasks/todo.md` before making repo-tracked edits.
- Keep supported Zig resolution to the pinned Zig on PATH, `ABI_HOST_ZIG`, or ABI's canonical host-built cache under `$HOME/.cache/abi-host-zig/<.zigversion>/bin/zig`. Don't add extra ad hoc toolchain surfaces beyond the permanent bootstrap flow.
- Bootstrapping a pinned host-built Zig is not enough by itself for direct `zig build` gates on Darwin 25+ / macOS 26+. `zig build` uses the compiler you invoked, so prepend the canonical cache bin dir to `PATH` (or invoke that binary explicitly) before expecting `zig build full-check` / `zig build check-docs` to leave degraded mode.
- On blocked Darwin hosts, direct `zig run tools/scripts/toolchain_doctor.zig` and `zig run tools/scripts/check_zig_version_consistency.zig` hit the same pre-`build.zig` linker wall as `zig build`. Use `./tools/scripts/inspect_toolchain.sh` plus `./tools/scripts/run_build.sh typecheck --summary all` for fallback evidence until a known-good host-built Zig exists.
- Resolve generated registry artifacts explicitly; keep deterministic parser paths for generated ZON.
- External hooks/linters may rewrite source files destructively (reordering imports before doc comments, changing `@import("abi")` to relative internal paths). Use `git checkout HEAD -- <file>` to restore, or atomic `sed -i '' + git add` for edits that must survive hooks.
- `src/services/tests/mod.zig` is a separate test root with named imports injected by `build.zig`. Child files under `src/services/tests/` and `src/services/tests/property/` should keep `@import("abi")`; swapping them to `src/root.zig` creates duplicate module ownership (`abi` and `root`) during `zig build test` / `typecheck`.
- Before appending `.zig` to a local import, resolve the target path. A suffix-only rewrite against a nonexistent target leaves the code just as broken and can hide that the real fix is a gated import or a different module path.

## Cross-Feature Import Safety
- Feature modules must not directly import other feature modules' `mod.zig` — this bypasses the compile-time feature gate. Use `build_options` conditional imports: `const obs = if (build_options.feat_profiling) @import("../../observability/mod.zig") else @import("../../observability/stub.zig");`
- `@import("abi")` cannot be used within files that are part of the `abi` module. Use relative imports instead: `@import("../types.zig")`, `@import("../../database/mod.zig")`.
- After adding new build flags, update `tools/cli/tests/build_options_stub.zig` to include them. The stub must match all `feat_*` fields in `build/options.zig`.
- The format-check surface must cover all source directories: `build.zig build/ src/ tools/ tests/ bindings/ lang/`. Keep `AGENTS.md`, `CLAUDE.md`, and `tools/scripts/fmt_repo.sh` in sync.

## Parallel Agent & PR Workflow
- Parallel agent dispatch (worktree agents) for multi-stream doc/code fixes works well but creates stale PRs when a large restructuring commit lands afterward. Triage PRs immediately after pushing restructuring changes.
- Code review by subagents catches import violations in new files that format checks miss. Always run both zig fmt and typecheck as complementary gates.
