# Lessons Learned

## Zig 0.16 API & Language

Common pitfalls from Zig 0.16 API renames, removed functions, and new language constraints.

- `GeneralPurposeAllocator` -> `DebugAllocator`. `std.time.timestamp()` -> `unixSeconds()`. `File.writeAll` -> `writeStreamingAll(io, data)`. `makeDirAbsolute*` -> `createDirPath(.cwd(), io, path)`. No `usingnamespace` -> pass parent context as parameters to submodule init functions.
- `LazyPath`: use `.cwd_relative`/`.src_path`, not `.path`. `addTest`/`addExecutable`: use `root_module`, not `root_source_file`. ZON parsing: use arena-backed `fromSliceAlloc`, deinit arena at scope end.
- **dev.2905+**: All `@import("path")` must have explicit `.zig` extensions. Single-module file ownership enforced -- every `.zig` file belongs to exactly one named module. Cross-module relative-path imports are illegal. Solution: consolidate all `src/` into single `abi` module.
- `valueIterator()`/`keyIterator()` not `.values()`. `@enumFromInt(x)` not `intToEnum`. Use `std.posix.poll` on STDIN instead of `std.time.sleep` in event loops.
- `ArrayListUnmanaged` and `AutoHashMapUnmanaged` must use `.empty` not `.{}` for initialization. The `.{}` literal triggers "missing struct field: items" errors.
- Feature-gated sub-modules must not directly import other feature modules via relative paths (bypasses the gate). Use `build_options` conditional imports: `const obs = if (build_options.feat_profiling) @import("../../observability/mod.zig") else @import("../../observability/stub.zig");`
- After adding new build flags, update `tools/cli/tests/build_options_stub.zig` to include them. The stub must match all `feat_*` fields in `build/options.zig`.

Root cause: Zig 0.16 introduced sweeping API renames, removed stdlib functions, and enforced single-module file ownership -- code written for 0.13/0.14 broke silently or with cryptic errors. Direct cross-feature imports bypassed comptime gates, and missing build option stubs caused test compilation failures.
Prevention rule: Before using any stdlib API, check the Zig 0.16 API Changes section in CLAUDE.md. Always use build_options conditional imports for cross-feature references. After adding any feat_* flag, update build_options_stub.zig immediately.

## Darwin 25+ / macOS Toolchain

Workarounds for the stock Zig LLD linker failure on Darwin 25+ and toolchain resolution.

- `zig build` fails with undefined symbols (`_malloc_size`, `_nanosleep`, etc.) because the build runner links first, before `build.zig` runs. No `build.zig` knob can fix this.
- Use `./tools/scripts/run_build.sh`, `zig fmt --check`, or `zig test -fno-emit-bin` locally. CI (Linux) is authoritative.
- LLD has zero Mach-O support -- never `use_lld = true` on macOS targets.
- Keep supported Zig resolution to the pinned Zig on PATH, `ABI_HOST_ZIG`, or the canonical host-built cache under `$HOME/.cache/abi-host-zig/<.zigversion>/bin/zig`. Don't add extra ad hoc toolchain surfaces.
- Bootstrapping a pinned host-built Zig is not enough by itself for direct `zig build` gates on Darwin 25+. Prepend the canonical cache bin dir to `PATH` (or invoke that binary explicitly) before expecting `zig build full-check` to leave degraded mode.
- On blocked Darwin hosts, `zig run tools/scripts/toolchain_doctor.zig` and `zig run tools/scripts/check_zig_version_consistency.zig` hit the same pre-`build.zig` linker wall. Use `./tools/scripts/inspect_toolchain.sh` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback.

- Building Zig from source on Darwin 25+ does not fix the linker issue. The C++ bootstrap (`zig2`) links fine with system clang/ld, but the stage 3 self-hosted build uses `zig2`'s embedded LLD which has the same Darwin 25 tbd stub bug. This is a chicken-and-egg problem — the fix must come from upstream Zig, not from local host builds.
- Zig 0.16.0-dev.2962 requires LLVM 21.x. Homebrew default is LLVM 22. Use `brew install llvm@21` and `-DCMAKE_PREFIX_PATH="/opt/homebrew/opt/llvm@21;/opt/homebrew/opt/zstd"`.
- zstd must be explicitly linked when building Zig from source on macOS with Homebrew — add it to `CMAKE_PREFIX_PATH`.

Root cause: The Zig build runner itself must link against Darwin system libraries before build.zig executes, so prebuilt Zig binaries fail on newer macOS versions. Ad hoc toolchain paths introduced subtle breakage. Host-building Zig doesn't escape the LLD bug because the self-hosted stage uses the bootstrap's embedded LLD.
Prevention rule: On Darwin 25+, bootstrap a host-built Zig via bootstrap_host_zig.sh and prepend its bin dir to PATH. Use only the canonical Zig resolution chain (PATH, ABI_HOST_ZIG, host cache). Never set use_lld = true on macOS. Wait for upstream Zig LLD fixes rather than attempting local source builds.

## Module System & Imports

Rules for `@import` usage, module boundaries, and the foundation namespace.

- `@import("abi")` cannot be used within files that are part of the `abi` module -- this creates a circular "no module named 'abi' available within module 'abi'" error. Use relative imports instead: `@import("../types.zig")`, `@import("../../database/mod.zig")`. Only external modules (CLI, tests with separate roots) use `@import("abi")`.
- `src/services/shared/mod.zig` provides shared service types (allocators, logging, config). Exposed as `abi.foundation` via `pub const foundation = @import("services/shared/mod.zig")` in `src/root.zig`. There is no separate `foundation` named module -- it is a re-export within the `abi` module. Internal files use relative paths to reach `src/services/shared/`; external modules use `@import("abi").foundation`.
- `wireAbiImports(module, build_opts)` adds only `build_options` as a named import. The foundation namespace is wired through the normal `abi` module import graph, not as a named import.
- `src/services/tests/mod.zig` is a separate test root with named imports injected by `build.zig`. Child files under `src/services/tests/` should keep `@import("abi")`; swapping them to `src/root.zig` creates duplicate module ownership (`abi` and `root`) during `zig build test`.
- In `src/root.zig`, keep private module/type aliases distinct from public compatibility re-exports. Reusing the same identifier inside nested namespace structs creates ambiguous references under Zig master.
- Before appending `.zig` to a local import, resolve the target path. A suffix-only rewrite against a nonexistent target hides that the real fix may be a gated import or a different module path.

Root cause: The foundation namespace was initially treated as a separate named module, and @import("abi") was used inside the abi module itself, causing duplicate module ownership and circular import errors.
Prevention rule: Never create a separate named module for foundation. Never use @import("abi") inside src/. Always resolve target paths before rewriting imports.

## Build System

Build infrastructure patterns, test discovery, and bulk operation safety.

- Files in `build/test_discovery.zig` must compile standalone with `zig test <file> -fno-emit-bin`. Cross-directory `@import("../../")` breaks this -- inline small deps or use build-system modules.
- Use `std.fmt.comptimePrint` to parameterize build steps that differ only by a flag string. One shared module graph for manifest-driven tests, not per-entry modules.
- Tool-side Zig modules under `tools/` cannot reach into `../../build/*.zig` with relative imports. Pass shared build metadata as a named module import from `build.zig` instead.
- Feature-test per-entry modules violate Zig 0.16 single-file ownership when entries share files through import graphs. Fix: use the `abi` module directly as the test root (`addTest(.{ .root_module = abi_module })`).
- Never bulk find-replace without excluding string literal interiors. After any bulk text operation, run `zig fmt --check` immediately. Corruption cascades across multiple waves -- don't commit until parse errors reach 0.
- The format-check surface must cover all source directories: `build.zig build/ src/ tools/ tests/ bindings/ lang/`. Keep `AGENTS.md`, `CLAUDE.md`, and `tools/scripts/fmt_repo.sh` in sync.

- `@hasField(std.Build, "graph")` is true on Zig dev.104 because `Build.Graph` exists, but it lacks `environ_map`, `io`, and `zig_exe`. Use `@hasField(std.Build, "graph") and @hasField(std.Build.Graph, "environ_map")` for the comptime gate. On dev.104, `comptime` keyword is redundant on module-level `const` (triggers error), and explicit `_ = param` discards are "pointless" when the parameter appears in a dead comptime branch.

Root cause: Build system files used cross-directory relative imports violating single-module ownership, and bulk text operations corrupted source files in ways that cascaded across compilation units. Comptime gates that only checked for `graph` field existence missed that older toolchains have a different Graph struct shape.
Prevention rule: Never use cross-directory relative imports in build/ or tools/. Pass shared metadata via named module imports from build.zig. Run zig fmt --check immediately after any bulk text operation. When gating on `b.graph.*` APIs, check for specific sub-fields, not just the graph field.

## Feature Module Contract (mod/stub)

Keeping mod.zig and stub.zig in sync for comptime-gated features.

- `stub.zig` must match `mod.zig` public signatures exactly. After any feature migration, verify parity -- code compiles with `feat_X=true` but fails with `feat_X=false` if stubs diverge.
- Shared types go in `types.zig` -- both `mod.zig` and `stub.zig` import from it. Use `StubFeature`/`StubFeatureNoConfig` from `core/stub_context.zig` for common stub boilerplate.
- Validation matrix no-X entries must enable ALL other features. When adding a flag, add it to all existing no-X entries. Verify: 2 baseline + N solo + N no-X.
- CLI tools accessing `abi.features.ai.<submodule>` will fail at compile time if the sub-module isn't re-exported from the stub. When adding new sub-modules accessed by CLI, add to both `mod.zig` AND `stub.zig`. Inline stubs need all methods the caller invokes -- each returning `error.AiDisabled` or a safe default.

- Stub-local type definitions (e.g. defining a new `ConfidenceLevel` enum in a stub instead of importing the canonical one from `types.zig`) create silent type divergence -- the stub compiles but has different enum variants, causing switch exhaustiveness errors or wrong behavior when the feature is disabled.
- Stubs that expose internal helper types as `pub` (e.g. `pub const StubChatHandler`) break `assertParity` because these names don't exist in `mod.zig`. Make intermediate types private (`const`) and only expose the public aliases that match mod.zig.
- When `mod.zig` delegates to sub-module types (e.g. `pub const MemoryInfo = unified.MemoryInfo`), the stub should define its own types inline. When `mod.zig` defines types inline, the stub must match. Never have both `mod.zig` inline AND sub-module types visible -- it creates nominally different types.
- Error sets in stubs must match: if `init()` returns `Error!void`, the error returned must be a member of `Error`. Use the feature-specific disabled error (e.g. `error.ObservabilityDisabled`) not a generic `error.FeatureDisabled`.
- Core-level stubs (`core/database/stub.zig`) must export all sub-module namespaces that the feature-level facade (`features/database/stub.zig`) references. Missing exports cause compilation errors only when the parity check is enabled.

Root cause: Feature stubs diverged from their mod.zig counterparts after migrations, causing compilation failures only when specific features were disabled -- a path not tested by default builds. Stub-local type definitions and extra pub declarations created subtle parity drift that the `assertParity` comptime check catches.
Prevention rule: After any mod.zig signature change, immediately update stub.zig to match. Import canonical types from `types.zig` instead of redefining locally. Keep helper types private. Run `zig build check-stub-parity` after every stub edit. Run the full flag validation matrix (baseline + solo + no-X) before committing feature module changes.

## Shell Scripts & Tooling

Patterns for shell scripts, hooks, and developer tooling.

- Guard `set -euo pipefail` with `if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then ... fi` so strict mode only applies when executed directly (not sourced).
- Extract shared utility functions to a `lib.sh` to prevent drift. Trap SIGINT/SIGTERM in long-running scripts to clean up partial state.
- Use dedicated edit tools for file mutations, reserve shell for inspection. Review `tasks/lessons.md` and refresh `tasks/todo.md` before making repo-tracked edits.
- Resolve generated registry artifacts explicitly; keep deterministic parser paths for generated ZON.
- External hooks/linters may rewrite source files destructively (reordering imports, changing `@import("abi")` to relative internal paths). Use `git checkout HEAD -- <file>` to restore, or atomic `sed -i '' + git add` for edits that must survive hooks.

Root cause: Shell scripts sourced by other scripts inherited strict mode, and external hooks rewrote source files destructively in ways that were hard to trace.
Prevention rule: Guard strict mode behind a direct-execution check. Extract shared functions to lib.sh. Restore hook-damaged files with git checkout HEAD.

## Workflow & CI

Version pinning, parallel agents, and CI gate discipline.

- When repinning Zig: update `.zigversion`, `build.zig.zon`, `baseline.zig`, `README.md`, CI config atomically. Validate version/commit pairs against `ziglang.org/builds` artifact metadata, not GitHub master HEAD.
- Update `roadmap_catalog.zig` and regenerate artifacts before updating `tasks/` files to prevent plan state drift.
- Parallel agent dispatch (worktree agents) for multi-stream doc/code fixes works well but creates stale PRs when a large restructuring commit lands afterward. Triage PRs immediately after pushing restructuring changes.
- Code review by subagents catches import violations in new files that format checks miss. Always run both `zig fmt` and typecheck as complementary gates.

- When cherry-picking from worktree branches, always close the superseded PRs immediately to prevent stale PR accumulation.
- `std.time.Instant` does not exist in Zig 0.16. Use `std.c.clock_gettime(.MONOTONIC, &ts)` for wall-clock timing in build/ and tools/ code.
- Files under `docs/` are managed by gendocs — placing non-generated `.md` files there triggers `check-docs` drift. Use `docs/plans/` for implementation plans instead.
- C bindings (`bindings/c/`) use `@import("abi")` (correct — outside src/). New feature exports follow the opaque handle pattern: `FooHandle = opaque {}`, `FooWrapper` struct, `export fn` with integer return codes.
- RLE compression for block storage: use 0xFF marker byte with escape sequence `[0xFF, 0x01, 0xFF]` for literal 0xFF bytes. Simple, no external deps, good for zero-padded vector data.
- POSIX file I/O (`std.posix.open/write/read/close/lseek`) works for block storage in Zig 0.16. The `std.Io.Threaded` API is more complex and better suited for full-featured applications, not low-level storage.

- `std.io.fixedBufferStream` does not exist in Zig 0.16. Agents generating code with this API will cause `check-zig-016-patterns` to fail. Use manual buffer slicing instead.
- `_ = param` after already referencing `param` triggers "pointless discard of function parameter" in Zig 0.16. If a function parameter is only sometimes used, use `_:` prefix in the signature instead.
- `defer` vs `errdefer` for lists returned via `toOwnedSlice()`: `defer list.deinit()` causes use-after-free because it frees the list even on the success path where ownership transferred to the caller. Always use `errdefer` when the function returns owned memory.
- `git add -A` can accidentally include build artifacts (e.g. `lang/swift/.build/`). Always prefer `git add <specific files>` and maintain `.gitignore` entries for build output directories.
- Feature modules that are pure re-export facades (like `features/database/mod.zig`) only need `refAllDecls` tests — adding duplicate tests would mirror the core module's test suite.

- Version pin bumps have abbreviated refs. When updating version strings like `0.16.0-dev.2934+47d2e5de9`, also grep for abbreviated forms like `dev.2934+` and `dev.2934` that won't match the full-string sed pattern. Use multiple grep passes to catch all occurrences.

Root cause: Partial version pin updates left inconsistent metadata, and large restructuring commits invalidated in-flight PRs from parallel agents. Zig 0.16 time APIs differ from older versions. Gendocs manages docs/ exclusively. Agent-generated code sometimes uses removed Zig APIs.
Prevention rule: Treat all version pin files as an atomic set. Triage stale PRs after restructuring commits. Run both zig fmt and typecheck as complementary verification gates. Use POSIX clock_gettime for timing, not std.time.Instant. Keep non-generated docs outside docs/. Always run check-zig-016-patterns after agent-generated code.
