# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABI is a Zig 0.16 framework for AI services, vector search, and high-performance compute. It has 27 comptime-gated feature modules (across 19 feature directories), a CLI with 40 registered commands (plus subcommands), and a TUI dashboard. The public API is exposed through `src/abi.zig` via `@import("abi")`.

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
zig build validate-flags                     # Check 40 feature flag combos
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
src/features/            # 19 feature directories (27 catalog entries)
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
- `flags.zig` — `FlagCombo` validation matrix (40 combos)
- `modules.zig` — module creation helpers
- `test_discovery.zig` — **single source of truth** for feature test manifest
- `cli_smoke_runner.zig` — descriptor-driven CLI smoke tests
- `gpu.zig` / `gpu_policy.zig` — GPU backend selection and policy
- `link.zig` — Metal framework linking, Darwin SDK detection
- `cel.zig` — CEL toolchain integration (build steps: `cel-check`, `cel-build`, etc.)
- `targets.zig` — Cross-compilation target definitions
- `wasm.zig` — WebAssembly build support
- `mobile.zig` — Mobile platform build support

### Test Structure

Two test roots:
1. **Main tests**: `src/services/tests/mod.zig` → `zig build test` (1289-1290 pass is normal variance due to one hardware-gated test)
2. **Feature tests**: `build/test_discovery.zig` manifest → `zig build feature-tests` (generated root, no tracked mirror file)

After adding tests: add entries to `build/test_discovery.zig` manifest (the single source of truth).

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

All default to `true`. Use `-Dfeat-<name>=false` to disable. Full list in `build/options.zig` (`BuildOptions` struct).

| Flag | Description |
|:-----|:------------|
| `-Dfeat-ai` | AI features, agents, connectors |
| `-Dfeat-llm` | Local LLM inference |
| `-Dfeat-gpu` | GPU acceleration |
| `-Dfeat-database` | Semantic store / vector database (`wdbx` alias) |
| `-Dfeat-network` | Distributed compute |
| `-Dfeat-web` | HTTP client utilities |
| `-Dfeat-profiling` | Performance profiling |
| `-Dfeat-analytics` | Analytics and metrics collection |
| `-Dfeat-auth` | Authentication and authorization |
| `-Dfeat-cache` | Caching layer |
| `-Dfeat-cloud` | Cloud provider integrations |
| `-Dfeat-compute` | Compute engine (work-stealing scheduler) |
| `-Dfeat-desktop` | Desktop platform support |
| `-Dfeat-documents` | Document processing |
| `-Dfeat-gateway` | API gateway |
| `-Dfeat-messaging` | Message queues and pub/sub |
| `-Dfeat-mobile` | Mobile platform support |
| `-Dfeat-search` | Search engine |
| `-Dfeat-storage` | Storage backends |
| `-Dfeat-training` | Training pipelines |
| `-Dfeat-reasoning` | Reasoning / chain-of-thought |
| `-Dfeat-benchmarks` | Benchmark suite |
| `-Dfeat-pages` | Static page serving |

Internal (derived, not user-facing): `feat_explore`, `feat_vision` — derived from `feat_ai`.

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

## Workflow Rules

- **Read before edit**: For batch operations across multiple files, read ALL target files first to get current state before making any edits. Never rely on cached/stale file contents.
- **Plan before execute**: For multi-file batch operations, present the full plan with specific file paths and changes BEFORE executing. Wait for user confirmation on ambiguous changes.
- **Validate after edit**: After editing YAML, ZON, or configuration files, validate syntax by running appropriate check commands.
- **Version pin atomicity**: When changing version strings, grep for all occurrences first, then update all files in one pass. Never leave partial updates.

## Common Pitfalls

These patterns are distilled from `tasks/lessons.md`:

1. **mod.zig ↔ stub.zig sync**: When changing a public function signature in any `src/features/*/mod.zig`, always update the matching `stub.zig`. Validate with `zig build validate-flags`.
2. **Version pin wave**: When repinning Zig, update `.zigversion`, `build.zig.zon`, `tools/scripts/baseline.zig`, `README.md`, and generated artifacts in one atomic wave.
3. **ZON parsing ownership**: Use arena-backed parsing for complex ZON inputs — `std.zon.parse.fromSliceAlloc` returns struct-owned slices, not wrapper-owned.
4. **Build runner links first**: If `zig build` fails with undefined symbols (`__availability_version_check`, `_arc4random_buf`), the build runner itself can't link. No `build.zig` workaround helps — you need a Zig built from source on the same host (see CEL toolchain below).
5. **Registry refresh after CLI changes**: Always run `zig build refresh-cli-registry` after adding/modifying commands in `tools/cli/commands/`.
6. **Async I/O in TUI**: Use `std.posix.poll` on STDIN instead of `std.time.sleep` in event loops.
7. **Never `use_lld` on macOS**: LLD has zero Mach-O support. Use Apple's `/usr/bin/ld` via `run_build.sh` or the CEL toolchain.
8. **Standalone test files**: Files in `build/test_discovery.zig` must compile with `zig test <file> -fno-emit-bin`. Inline small cross-directory deps rather than using relative `@import("../../")`.

## Adding or Modifying Feature Modules

1. Create/edit `src/features/<name>/mod.zig` (real implementation)
2. Mirror **every public function signature** in `src/features/<name>/stub.zig`
3. Add the `feat_<name>` flag to `build/options.zig` (`BuildOptions` + `CanonicalFlags`)
4. Register in `src/core/feature_catalog.zig`
5. Add test entries to `build/test_discovery.zig` manifest
6. Add flag combo rows to `build/flags.zig` if needed
7. Wire up in `src/abi.zig` (comptime feature selection)
8. Validate: `zig build validate-flags` then `zig build full-check`

## Known Environment Issue (Darwin/macOS 26+)

Upstream Zig linker incompatibility (`__availability_version_check`, `_arc4random_buf` undefined). The build runner itself fails to link — no `build.zig` workaround helps. See `docs/ZIG_MACOS_LINKER_RESEARCH.md`.

**Primary fix — CEL toolchain** (patched Zig built from source):
```bash
./tools/scripts/cel_migrate.sh           # Full guided build + activate + validate
# Or step-by-step:
./.zig-bootstrap/build.sh && eval "$(./tools/scripts/use_zig_bootstrap.sh)"
# Diagnostics: zig build cel-check | cel-doctor | cel-status | cel-verify
```

**Current workaround — `run_build.sh`** (relinks build runner with Apple ld):
```bash
./tools/scripts/run_build.sh lint         # Format check (works on Darwin 25+)
./tools/scripts/run_build.sh test         # Tests (may hit wdbx module conflict)
zig fmt --check build.zig build/ src/ tools/  # Direct format check (no build runner)
```

**Fallbacks:** `zig-bootstrap-emergency/` (see `zig-bootstrap-emergency/ABI-USAGE.md`), or `zig test -fno-emit-bin` for compile-only validation.

## Zig 0.16 std API notes

- **Time:** `std.time` has no `timestamp()`; use platform time (e.g. `abi.services.shared.time` for `timestampSec()`/`timestampNs()`) or pass `now: i64` from the caller. See `src/wdbx/dist/mod.zig` (Coordinator) and `src/features/ai/training/self_learning.zig` (getCurrentTimestamp).
- **Enums:** Prefer `@enumFromInt(x)` for int→enum; for validated parsing use a `switch (x) { 0 => .a, 1 => .b, else => return error.Invalid; }`. Do not rely on `std.meta.intToEnum` (removed in 0.16).
- **HashMap iteration:** Use `valueIterator()` / `keyIterator()` (e.g. `while (map.valueIterator().next()) |v|`) rather than `.values()` on `AutoHashMapUnmanaged` (`.values()` is not part of the public API).
- **Allocator vtable (0.16):** `alloc`/`resize`/`free` use `alignment: std.mem.Alignment`, not `u8`. Use `std.mem.Alignment.fromByteUnits(n)` when converting from byte alignment.
- **Build:** Use `b.createModule(.{ .root_source_file = b.path(...), ... })` and `addTest`/`addExecutable` with `.root_module = mod`. No `root_source_file` on the compile step; LazyPath is `b.path(...)` (no `.path` field).
- **mem.readInt/writeInt:** Signatures take `*const [N]u8` / `*[N]u8`; slices of the right length are accepted. Use `std.builtin.Endian.little`/`.big`.
- **std.Io:** For concurrent/async I/O and time, see `std.Io` (Threaded, async/concurrent, Clock). TUI/data fetches can be refactored to use `std.Io` patterns where applicable.
- **std.Io.Dir (filesystem):** No `makeDirAbsolute*` — use `std.Io.Dir.createDirPath(.cwd(), io, path)` for recursive creation. No `deleteTreeAbsolute` — use `std.Io.Dir.deleteTree(.cwd(), io, path)`. File writes use `file.writeStreamingAll(io, data)` (no `File.writeAll`). File existence check: `Io.Dir.openFileAbsolute(io, path, .{}) catch return false` then close.
- **Standalone compilation:** Files in `build/test_discovery.zig` manifest must compile with `zig test <file> -fno-emit-bin`. Cross-directory `@import("../../")` breaks this — inline small dependencies or use build-system-provided modules.
