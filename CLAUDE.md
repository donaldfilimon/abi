# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABI is a Zig 0.16 framework for AI services, vector search, and high-performance compute. It has 19 comptime-gated feature modules, a CLI with 40 commands, and a TUI dashboard. The public API is exposed through `src/abi.zig` via `@import("abi")`.

## Build Commands

```bash
# Essential
zig build                                    # Build everything
zig build test --summary all                 # Main tests (~1290 pass, 6 skip)
zig build feature-tests --summary all        # Feature tests (~2836 pass, 9 skip)
zig build full-check                         # Format + tests + feature tests + flag validation + CLI smoke
zig build verify-all                         # Release gate (full-check + examples + wasm + cross + docs)

# Targeted
zig build cli-tests                          # CLI unit tests
zig build tui-tests                          # TUI unit tests
zig build launcher-tests                     # Shell/launcher tests
zig build wdbx-fast-tests                    # WDBX vector database tests
zig build validate-flags                     # Check 34 feature flag combos
zig build lint                               # Check formatting
zig build fix                                # Auto-format

# Registry and docs
zig build refresh-cli-registry               # Regenerate CLI registry snapshot
zig build check-cli-registry                 # Verify registry is current
zig build check-docs                         # Docs consistency check
zig build toolchain-doctor                   # Diagnose toolchain issues
zig build benchmarks                         # Run benchmarks

# Single test file (bypass build system)
zig test src/services/tests/mod.zig --test-filter "pattern"
```

## Architecture

### Comptime Feature Gating (Critical Pattern)

Every feature lives in `src/features/<name>/` with two files:
- `mod.zig` — real implementation
- `stub.zig` — disabled fallback (returns zero values, empty slices, or `error.FeatureDisabled`)

The build system selects mod vs stub based on `build_options.feat_<name>`. **When changing a public function in `mod.zig`, you must update `stub.zig` with the same signature** or disabled-flag builds break (`-Dfeat-<name>=false`).

Feature flags use the prefix `feat_<name>` (NOT `enable_<name>`). All default to `true`. Canonical sources:
- Flag definitions: `build/options.zig` (`BuildOptions` struct)
- Feature enum/metadata: `src/core/feature_catalog.zig`
- Flag validation matrix: `build/flags.zig`

### Import Rules

- External consumers: `@import("abi")`
- Feature modules in `src/features/`: **relative imports only** — never `@import("abi")`
- Enforced by `zig build check-imports`

### Source Layout

```
src/abi.zig              # Public API entry point (comptime feature selection)
src/core/                # Config, feature catalog, framework lifecycle
src/features/            # 19 feature modules (ai, gpu, database, network, web, ...)
  ai/                    # LLM inference, agents, training, streaming, abbey, personas
  database/              # Semantic store (WDBX alias), HNSW indexing, distributed
  gpu/                   # CUDA, Vulkan, Metal, WebGPU backends
  network/               # Distributed compute, Raft consensus
src/services/            # Shared runtime services and test roots
src/wdbx/               # WDBX vector database engine
build/                   # Modular build system (options, flags, modules, test discovery)
tools/cli/               # CLI executable and commands
  commands/              # Command implementations with `pub const meta: command.Meta`
  ui/dsl/                # TUI DSL for dashboard boilerplate
  generated/             # Auto-generated CLI registry snapshot
  terminal/launcher/     # TUI launcher catalog
tools/gendocs/           # Documentation generator (consumes CLI registry)
tools/scripts/           # Toolchain doctor, baseline, utilities
```

### Build System

`build.zig` imports modular components from `build/`:
- `options.zig` — `BuildOptions` struct and flag reading
- `flags.zig` — `FlagCombo` validation matrix (34 combos)
- `modules.zig` — module creation helpers
- `test_discovery.zig` — **single source of truth** for feature test manifest
- `cli_smoke_runner.zig` — descriptor-driven CLI smoke tests
- `gpu.zig` — GPU backend selection (`-Dgpu-backend=auto|cuda|vulkan|metal`)
- `link.zig` — Metal framework linking, Darwin SDK detection

### Test Structure

Two test roots:
1. **Main tests**: `src/services/tests/mod.zig` → `zig build test` (1289-1290 pass is normal variance due to one hardware-gated test)
2. **Feature tests**: `build/test_discovery.zig` manifest → `zig build feature-tests` (generated root, no tracked mirror file)

After adding tests: run `zig build update-baseline`.

### CLI Commands

Commands live in `tools/cli/commands/` organized by category (`ai/`, `core/`, `dev/`, `db/`, `infra/`). Each command exports `pub const meta: command.Meta` for options, help, and risk metadata. After adding or modifying commands, run `zig build refresh-cli-registry`.

### Key Modules

| Module | Entry | Purpose |
|--------|-------|---------|
| `abi.App` / `abi.AppBuilder` | `src/abi.zig` | Primary runtime types |
| `abi.wdbx` | `src/wdbx/wdbx.zig` | Vector database engine |
| `abi.wdbx.dist` | `src/wdbx/dist/mod.zig` | Coordinator (heartbeat state machine), node health; pass `now: i64` from your time source to `registerNode`, `updateHeartbeat`, `tick`/`tickNow`. |
| `abi.wdbx.dist.rpc` | `src/wdbx/dist/rpc.zig` | Binary codec for node-to-node messages: `Header`, `HeartbeatPayload`, `BlockSyncRequest`, `encodeMessage`/`decodeHeader`. |
| `abi.wdbx.dist.replication` | `src/wdbx/dist/replication.zig` | In-process block sync: `runRequesterPath` parses response stream (BlockSyncResponse + BlockChunk); optional trace callback. |
| `abi.features.ai` | `src/features/ai/mod.zig` | LLM, agents, training, streaming |
| `abi.features.database` | `src/features/database/mod.zig` | Semantic store, HNSW indexing |

### v3 API Surface

ABI exposes canonical v3 entrypoints only:
- `abi.App` / `abi.AppBuilder` — primary runtime types
- `abi.wdbx` — vector database, with `abi.hnsw` and `abi.simd` sub-modules
- `abi.personas` — multi-persona system; `abi.routing` — moderator
- `abi.inference_engine` — high-performance token generation
- `abi.server` — REST API server with OpenAI-compatible endpoints

Legacy v2 aliases are fully consolidated into v3.

### All Feature Flags

All default to `true`. Use `-Dfeat-<name>=false` to disable.

| Flag | Description |
|:-----|:------------|
| `-Dfeat-ai` | AI features, agents, connectors |
| `-Dfeat-llm` | Local LLM inference |
| `-Dfeat-gpu` | GPU acceleration |
| `-Dfeat-database` | Semantic store / vector database (`wdbx` alias) |
| `-Dfeat-network` | Distributed compute |
| `-Dfeat-web` | HTTP client utilities |
| `-Dfeat-profiling` | Performance profiling |

GPU backend: `-Dgpu-backend=auto|cuda|vulkan|metal` (comma-separated for multiple).

## Workflow

- Canonical workflow contract: `AGENTS.md`
- Active task tracker: `tasks/todo.md`
- Correction log: `tasks/lessons.md` — review at session start, update after corrections
- Always run `zig build full-check` before marking work complete
- Release gate: `zig build verify-all`

## Environment Variables

| Variable | Description |
|:---------|:------------|
| `ABI_OPENAI_API_KEY` | OpenAI API key |
| `ABI_ANTHROPIC_API_KEY` | Anthropic/Claude API key |
| `ABI_OLLAMA_HOST` | Ollama host (default: `http://127.0.0.1:11434`) |
| `ABI_OLLAMA_MODEL` | Default Ollama model |
| `ABI_HF_API_TOKEN` | HuggingFace API token |
| `DISCORD_BOT_TOKEN` | Discord bot token |

## Common Pitfalls

These patterns are distilled from `tasks/lessons.md`:

1. **mod.zig ↔ stub.zig sync**: When changing a public function signature in any `src/features/*/mod.zig`, always update the matching `stub.zig`. Validate with `zig build validate-flags`.
2. **Version pin wave**: When repinning Zig, update `.zigversion`, `build.zig.zon`, `tools/scripts/baseline.zig`, `README.md`, and generated artifacts in one atomic wave.
3. **ZON parsing ownership**: Use arena-backed parsing for complex ZON inputs — `std.zon.parse.fromSliceAlloc` returns struct-owned slices, not wrapper-owned.
4. **Build runner links first**: If `zig build` fails with undefined symbols (`__availability_version_check`, `_arc4random_buf`), the build runner itself can't link. No `build.zig` workaround helps — you need a Zig built from source on the same host (see CEL toolchain below).
5. **Registry refresh after CLI changes**: Always run `zig build refresh-cli-registry` after adding/modifying commands in `tools/cli/commands/`.
6. **Async I/O in TUI**: Use `std.posix.poll` on STDIN instead of `std.time.sleep` in event loops.

## Known Environment Issue

The local Darwin/macOS 26+ environment has an upstream Zig linker incompatibility (`__availability_version_check`, `_arc4random_buf` undefined symbols). Binary-emitting build steps fail. See `docs/ZIG_MACOS_LINKER_RESEARCH.md` for root-cause research and upstream issue refs.

**For this arch (Darwin):** Prefer the repo-local CEL toolchain so the repo pin, Zig binary, and ZLS stay aligned:
- `./.cel/build.sh` then `eval "$(./tools/scripts/use_cel.sh)"`.
- Ensures Zig matches `.zigversion` (`0.16.0-dev.2650+738d2be9d`) and keeps `.cel/bin/zls` beside the compiler.

**Recommended fix (primary path):** Use the `.cel` toolchain — a patched Zig built from source with macOS 26 fixes:
```bash
# Full guided migration:
./tools/scripts/cel_migrate.sh           # Build + activate + validate

# Or step-by-step:
./.cel/build.sh                          # Build patched Zig + ZLS (reuses bootstrap LLVM artifacts)
eval "$(./tools/scripts/use_cel.sh)"     # Set PATH to .cel/bin/{zig,zls}

# Diagnostics:
zig build cel-check                      # Quick status check
zig build cel-doctor                     # Full diagnostics
zig build cel-status                     # Detailed build status
zig build cel-verify                     # Verify binary exists
```
See `.cel/README.md` for details. The `.cel` fork pins the same commit as `.zigversion` and applies patches from `.cel/patches/`.

**CEL build steps in `build.zig`:** `cel-check`, `cel-build`, `cel-status`, `cel-verify`, `cel-doctor`.

**Fallbacks:** `zig-bootstrap-emergency/` (see `zig-bootstrap-emergency/ABI-USAGE.md`), or `zig test -fno-emit-bin` for compile-only validation. The repo sets `use_llvm`/`use_lld` on macOS 26+ for all host executables once the build runner links.

## Zig 0.16 std API notes

- **Time:** `std.time` has no `timestamp()`; use platform time (e.g. `abi.services.shared.time` for `timestampSec()`/`timestampNs()`) or pass `now: i64` from the caller. See `src/wdbx/dist/mod.zig` (Coordinator) and `src/features/ai/training/self_learning.zig` (getCurrentTimestamp).
- **Enums:** Prefer `@enumFromInt(x)` for int→enum; for validated parsing use a `switch (x) { 0 => .a, 1 => .b, else => return error.Invalid; }`. Do not rely on `std.meta.intToEnum` (removed in 0.16).
- **HashMap iteration:** Use `valueIterator()` / `keyIterator()` (e.g. `while (map.valueIterator().next()) |v|`) rather than `.values()` on `AutoHashMapUnmanaged` (`.values()` is not part of the public API).
- **Allocator vtable (0.16):** `alloc`/`resize`/`free` use `alignment: std.mem.Alignment`, not `u8`. Use `std.mem.Alignment.fromByteUnits(n)` when converting from byte alignment.
- **Build:** Use `b.createModule(.{ .root_source_file = b.path(...), ... })` and `addTest`/`addExecutable` with `.root_module = mod`. No `root_source_file` on the compile step; LazyPath is `b.path(...)` (no `.path` field).
- **mem.readInt/writeInt:** Signatures take `*const [N]u8` / `*[N]u8`; slices of the right length are accepted. Use `std.builtin.Endian.little`/`.big`.
- **std.Io:** For concurrent/async I/O and time, see `std.Io` (Threaded, async/concurrent, Clock). TUI/data fetches can be refactored to use `std.Io` patterns where applicable.
