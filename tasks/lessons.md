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

## `foundation` Named Module Pattern
- **What**: `src/services/shared/mod.zig` is the root of the `foundation` named module, created by `build/modules.zig:createFoundationModule`. It provides shared service types (allocators, logging, config) to all compilation targets.
- **Why**: Zig 0.16 dev.2905+ enforces single-module file ownership — a `.zig` file can only belong to one named module. Files in `src/services/shared/` cannot be imported via relative paths from the `abi` module because they belong to `foundation`. Instead, the `abi` module imports them via `@import("foundation")`.
- **Wiring**: `wireAbiImports(b, module, build_opts, target, optimize)` adds both `build_options` and `foundation` as named imports to any module that needs them. Every compilation target (main `abi`, CLI, tests, database, WASM, mobile) calls this helper to get both imports wired consistently.
- **Rule**: Never add files under `src/services/shared/` to another module's file list. They belong exclusively to `foundation`. Other modules access them through the named import.

## Bulk Operations Safety
- Never bulk find-replace without excluding string literal interiors. After any bulk text operation, run `zig fmt --check` immediately. Corruption cascades across multiple waves — don't commit until parse errors reach 0.

## Shell Script Patterns
- Guard `set -euo pipefail` with `if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then ... fi` so strict mode only applies when executed directly.
- Extract shared utility functions to a `lib.sh` to prevent drift. Trap SIGINT/SIGTERM in long-running scripts to clean up partial state.

## Tool & Workflow Discipline
- Use dedicated edit tools for file mutations, reserve shell for inspection. Review `tasks/lessons.md` and refresh `tasks/todo.md` before making repo-tracked edits.
- Keep supported Zig resolution to pinned Zig on PATH or ZVM. Don't add repo-local toolchain surfaces unless intended to be permanent.
- Resolve generated registry artifacts explicitly; keep deterministic parser paths for generated ZON.
