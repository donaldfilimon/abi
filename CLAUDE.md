# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zig 0.16 framework for AI services, vector search, and GPU compute. Pinned to `0.16.0-dev.2905+5d71e3051` (`.zigversion`). Package entrypoint: `src/root.zig`, exposed as `@import("abi")`. Note: `src/abi.zig` is a legacy internal file — not the package root.

## Commands

```bash
zig build test --summary all          # primary tests
zig build feature-tests --summary all # feature coverage
zig build full-check                  # pre-commit gate
zig build validate-flags              # flag combo check
zig build toolchain-doctor            # inspect active Zig resolution
zig build check-zig-version           # verify pin + docs consistency
zig build preflight                   # integration environment diagnostics
zig build refresh-cli-registry        # after CLI changes
zig build gendocs                     # regenerate docs
zig build check-docs                  # verify docs consistency
zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/  # format check (always works)
./tools/scripts/bootstrap_host_zig.sh # build pinned host Zig into the canonical cache
```

**Running a single test**: `zig test src/path/to/file.zig -fno-emit-bin` for standalone files. For module-integrated tests, use `zig build test --summary all` with feature flags to narrow scope.

**Darwin 25+ / macOS 26+**: stock prebuilt Zig can fail at the linker before `build.zig` runs. Bootstrap the pinned host-built Zig with `./tools/scripts/bootstrap_host_zig.sh`, prepend `$HOME/.cache/abi-host-zig/$(cat .zigversion)/bin` to `PATH`, then run `zig build toolchain-doctor`, `zig build full-check`, and `zig build check-docs`. Fallback evidence remains `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all`. Never `use_lld = true` on macOS (zero Mach-O support). Format checks always work.

## Architecture

- `src/root.zig` — public package root; all `abi.<domain>` namespaces defined here
- `src/features/<name>/` — 19 comptime-gated modules, each with `mod.zig` + `stub.zig` + `types.zig`
- `src/services/` — Non-gated services: connectors, LSP, MCP, runtime, security, platform
- `src/services/shared/` — Shared foundations exposed as `abi.foundation` (logging, security, time/SIMD)
- `src/core/` — Config, feature catalog, registry (incl. plugin system), `stub_context.zig`
- `src/inference/` — ML inference: engine, scheduler, sampler, paged KV cache
- `build/` — Modular build system (options, flags, modules, test discovery, `module_catalog.zig`)
- `tools/cli/` — CLI commands and registry (`tools/cli/registry/`)
- `tools/gendocs/` — Documentation generator (edits go here, not in generated `docs/api/`)
- `bindings/` — C and WASM language bindings (C bindings include plugin registry API)
- `tests/integration/` — Integration test matrix manifest and preflight diagnostics

### Feature gating in root.zig

Features use comptime conditional imports:
```zig
pub const ai = if (build_options.feat_ai) @import("features/ai/mod.zig") else @import("features/ai/stub.zig");
```

### mod/stub contract

`stub.zig` must match `mod.zig` public signatures exactly. Shared types go in `types.zig` — both mod and stub import from it. Use `StubFeature`/`StubFeatureNoConfig` from `core/stub_context.zig` for common stub boilerplate. Sub-module stubs not needed, but CLI-accessed sub-modules must be re-exported from both mod and stub.

### foundation namespace

`abi.foundation` is **not** a separate named module. It's `src/services/shared/mod.zig` re-exported via `src/root.zig`. Files within the `abi` module reach it via relative paths (e.g., `@import("../../services/shared/mod.zig")`). External modules access it through `@import("abi").foundation`.

## Import Rules (Critical)

- **External code** (CLI, tests with separate roots): use `@import("abi")`
- **Within `src/`** (part of the `abi` module): use **relative imports only**. `@import("abi")` from within the module causes a circular "no module named 'abi' available within module 'abi'" error.
- **Cross-feature imports**: never import another feature's `mod.zig` directly (bypasses the gate). Use `build_options` conditional: `const obs = if (build_options.feat_profiling) @import("../../observability/mod.zig") else @import("../../observability/stub.zig");`
- **Explicit `.zig` extensions** required on all path imports (Zig 0.16)
- **Single-module file ownership**: every `.zig` file belongs to exactly one named module
- **Test roots**: `src/services/tests/mod.zig` is a separate test root with named imports from `build.zig`. Its child files should keep `@import("abi")` — switching them to relative `src/root.zig` imports creates duplicate module ownership during `zig build test`
- **Build options stub**: when adding new `feat_*` flags to `build/options.zig`, also update `tools/cli/tests/build_options_stub.zig` to match

## Conventions

- `zig fmt` only — never `zig fmt .` from root (walks vendored fixtures that intentionally contain invalid code)
- `lower_snake_case` functions/files, `PascalCase` types/error sets
- Conventional commits (`fix:`, `feat:`, `docs:`, `chore:`), atomic scope
- Explicit error sets, propagate with `try`
- Bulk find-replace must exclude string literal interiors; run `zig fmt --check` immediately after any bulk text operation

See [AGENTS.md](AGENTS.md) for the full contributor workflow contract.

## Zig 0.16 API Changes

- `std.heap.DebugAllocator(.{}){}` not `GeneralPurposeAllocator`
- `std.time.unixSeconds()` not `timestamp()`
- `file.writeStreamingAll(io, data)` not `writeAll`
- `std.Io.Dir.createDirPath(.cwd(), io, path)` not `makeDirAbsolute`
- `.cwd_relative` / `.src_path` not `.path` on `LazyPath`
- `root_module` field not `root_source_file`
- `valueIterator()` not `.values()` on hash maps
- `@enumFromInt(x)` not `intToEnum`
- `ArrayListUnmanaged` / `AutoHashMapUnmanaged` init: `.empty` not `.{}`
- `pub fn main(init: std.process.Init) !void` not `pub fn main() !void`

## Feature Flags

All enabled by default. Disable: `-Dfeat-<name>=false`. GPU backend: `-Dgpu-backend=metal`.
27 flags in `build/options.zig`, 56 combos validated in `build/flags.zig`.
Catalog source of truth: `src/core/feature_catalog.zig`.

## Env Vars

`ABI_OPENAI_API_KEY`, `ABI_ANTHROPIC_API_KEY`, `ABI_OLLAMA_HOST`, `ABI_OLLAMA_MODEL`, `ABI_HF_API_TOKEN`, `DISCORD_BOT_TOKEN`

## Markdown / .gitignore

`*.md` is globally ignored. Tracked markdown files require explicit `!/path.md` entries in `.gitignore`. When adding new `.md` files: create the file, add allowlist entry, verify with `git status`.

## Workflow

1. Review `tasks/lessons.md` at session start
2. Plan multi-file changes in `tasks/todo.md`
3. Run strongest available verification gate before completing (see AGENTS.md for gate table)
4. Update `stub.zig` when changing `mod.zig` signatures
5. Update `tasks/lessons.md` after corrections (entries are grouped by topic heading; new entries should include root cause and prevention rule)
6. Version pin changes: update `.zigversion`, `build.zig.zon`, `baseline.zig`, `README.md`, CI config atomically

## Plugin System

External modules register at runtime via `abi.registry.plugin.PluginRegistry`. Plugins declare capabilities (`ai_provider`, `connector`, `storage_backend`, `gpu_backend`, etc.) and follow a `registered → loading → active → unloading` lifecycle. C bindings available in `bindings/c/include/abi.h` (`abi_plugin_register`, etc.).

## Raft Consensus

`src/features/network/raft.zig` implements Raft with pre-vote protocol (prevents disruptive elections from partitioned nodes) and partition tolerance (leader steps down on quorum loss). Fault injection via `FaultInjector` for testing. Gated by `feat-network`.

## Benchmarks

```bash
zig build benchmarks                       # Run all suites
zig build benchmarks -- --suite=simd       # Run specific suite
zig build benchmarks -- --quick            # Fast CI-friendly run
zig build bench-competitive                # Industry comparisons
```

Suites cover SIMD, memory, concurrency, database, network, crypto, AI, and GPU workloads. See `benchmarks/README.md` for details.

## References

- [AGENTS.md](AGENTS.md) — Workflow contract and verification gates
- [tasks/lessons.md](tasks/lessons.md) — Correction log (prevents repeat mistakes)
- [docs/PATTERNS.md](docs/PATTERNS.md) — Zig 0.16 codebase patterns
- [docs/STRUCTURE.md](docs/STRUCTURE.md) — Full directory tree reference
- [docs/guides/integration-environment.md](docs/guides/integration-environment.md) — CI/local/degraded mode contract
- [docs/plans/index.md](docs/plans/index.md) — Active execution plans and status
