# Lessons Learned

## Zig 0.16 API Changes
- `GeneralPurposeAllocator` → `DebugAllocator`. No `std.time.timestamp()` → use `unixSeconds()`. No `File.writeAll` → use `writeStreamingAll(io, data)`. No `makeDirAbsolute*` → use `createDirPath(.cwd(), io, path)`. No `usingnamespace` → pass parent context as parameters to submodule init functions.
- `LazyPath`: use `.cwd_relative`/`.src_path`, not `.path`. `addTest`/`addExecutable`: use `root_module`, not `root_source_file`. ZON parsing: use arena-backed `fromSliceAlloc`, deinit arena at scope end.
- **dev.2905+**: Slash-path `@import("dir/subdir")` and `@import("../sibling")` banned inside build-system modules. Must wire sub-modules via `addImport()` in build.zig and use named imports.
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

## Build System Patterns
- Files in `build/test_discovery.zig` must compile standalone with `zig test <file> -fno-emit-bin`. Cross-directory `@import("../../")` breaks this — inline small deps or use build-system modules.
- Use `std.fmt.comptimePrint` to parameterize build steps that differ only by a flag string. One shared module graph for manifest-driven tests, not per-entry modules.

## Bulk Operations Safety
- Never bulk find-replace without excluding string literal interiors. After any bulk text operation, run `zig fmt --check` immediately. Corruption cascades across multiple waves — don't commit until parse errors reach 0.

## Shell Script Patterns
- Guard `set -euo pipefail` with `if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then ... fi` so strict mode only applies when executed directly.
- Extract shared utility functions to a `lib.sh` to prevent drift. Trap SIGINT/SIGTERM in long-running scripts to clean up partial state.

## Tool & Workflow Discipline
- Use dedicated edit tools for file mutations, reserve shell for inspection. Review `tasks/lessons.md` and refresh `tasks/todo.md` before making repo-tracked edits.
- Keep supported Zig resolution to pinned Zig on PATH or ZVM. Don't add repo-local toolchain surfaces unless intended to be permanent.
- Resolve generated registry artifacts explicitly; keep deterministic parser paths for generated ZON.
