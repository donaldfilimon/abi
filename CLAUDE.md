# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

| Key | Value |
|-----|-------|
| **Zig** | `0.16.0-dev.2535+b5bd49460` or newer (pinned in `.zigversion`), use `./zigw` to run it |
| **Entry Point** | `src/abi.zig` |
| **Version** | 0.4.0 |
| **Test baseline** | 983 pass, 5 skip (988 total) — must be maintained |

## Build & Test Commands

Use `./zigw` or ensure your system `zig` matches `.zigversion` (e.g. via `zvm use master`).
If your system `zig` is already `0.16.0-dev.2535+b5bd49460` or newer, plain `zig` works.

```bash
zig build                                    # Build with default flags
zig build test --summary all                 # Run full test suite
zig test src/path/to/file.zig                # Test a single file
zig test src/services/tests/mod.zig --test-filter "pattern"  # Filter tests by name
zig fmt .                                    # Format all source
zig build full-check                         # Format + tests + flag validation + CLI smoke tests
zig build validate-flags                     # Compile-check 16 feature flag combos
zig build cli-tests                          # CLI smoke tests
zig build lint                               # CI formatting check
zig build benchmarks                         # Performance benchmarks
zig build examples                           # Build all examples
zig build check-wasm                         # Check WASM compilation
```

### Running the CLI

```bash
zig build run -- --help                      # CLI help
zig build run -- system-info                 # System and feature status
zig build run -- plugins list                # List plugins
```

### Feature Flags

`zig build -Denable-ai=true -Denable-gpu=false -Dgpu-backend=vulkan,cuda`

All features default to `true` except `-Denable-mobile`. GPU backends: `auto`, `none`,
`cuda`, `vulkan`, `metal`, `stdgpu`, `webgpu`, `webgl2`, `opengl`, `opengles`, `fpga`, `simulated`.
The `simulated` backend is always enabled as a software fallback for testing without GPU hardware.

| Feature Module | Build Flag | Notes |
|----------------|------------|-------|
| `ai` | `-Denable-ai` | Also: `-Denable-llm`, `-Denable-vision`, `-Denable-explore` |
| `analytics` | `-Denable-analytics` | |
| `cloud` | `-Denable-web` | Shares flag with web (no separate flag) |
| `database` | `-Denable-database` | |
| `gpu` | `-Denable-gpu` | Backend selection via `-Dgpu-backend=` |
| `network` | `-Denable-network` | |
| `observability` | `-Denable-profiling` | Not `-Denable-observability` |
| `web` | `-Denable-web` | Also gates cloud module |

## Critical Gotchas

These are the mistakes most likely to cause compilation failures:

| Mistake | Fix |
|---------|-----|
| `std.fs.cwd()` | `std.Io.Dir.cwd()` — Zig 0.16.0-dev.2535+b5bd49460 moved filesystem to I/O backend |
| `std.time.Instant.now()` for elapsed time | `std.time.Timer.start()` — use Timer for benchmarks/elapsed |
| `list.init()` | `std.ArrayListUnmanaged(T).empty` |
| `@tagName(x)` / `@errorName(e)` in format | `{t}` format specifier for errors and enums |
| Editing `mod.zig` only | Update `stub.zig` to match exported signatures |
| `std.fs.cwd().openFile(...)` | Must init `std.Io.Threaded` first and pass `io` handle |
| `file.read()` / `file.write()` | `file.reader(io).read()` / `file.writer(io).write()` — I/O ops need `io` handle |
| `std.time.sleep()` | `abi.shared.time.sleepMs()` / `sleepNs()` for cross-platform |
| `std.time.nanoTimestamp()` | Doesn't exist in `0.16.0-dev.2535+b5bd49460` — use `Instant.now()` + `.since(anchor)` for absolute time |
| `std.process.getEnvVar()` | Doesn't exist in `0.16.0-dev.2535+b5bd49460` — use `std.c.getenv()` for POSIX |
| `@typeInfo` tags `.Type`, `.Fn` | Lowercase in `0.16.0-dev.2535+b5bd49460`: `.type`, `.@"fn"`, `.@"struct"`, `.@"enum"`, `.@"union"` |
| `b.createModule()` for named modules | `b.addModule("name", ...)` — `createModule` is anonymous |
| `defer allocator.free(x)` then return `x` | Use `errdefer` — `defer` frees on success too (use-after-free) |
| `@panic` in library code | Return an error instead — library code should never panic |
| `std.time.Timer.read()` → `u64` | Returns `usize` in `0.16.0-dev.2535+b5bd49460`, not `u64` — cast or use `@as(u64, timer.read())` |
| `std.log.err` in tests | Test runner treats error-level log output as a test failure, even if caught. Skip the test before entering error paths |
| `defer allocator.free(x)` in `loadFromEnv()` | When returning owned data from a function, use `errdefer` not `defer` — defer frees on success path (use-after-free) |

### I/O Backend (Required for any file/network ops)

```zig
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty, // .empty for library, init.environ for CLI
});
defer io_backend.deinit();
const io = io_backend.io();

const content = try std.Io.Dir.cwd().readFileAlloc(
    io, path, allocator, .limited(10 * 1024 * 1024),
);
```

## Architecture: Comptime Feature Gating

The central architectural pattern is **comptime feature gating** in `src/abi.zig`. Each
feature module has two implementations selected at compile time via `build_options`:

```zig
// src/abi.zig — this pattern repeats for every feature
pub const gpu = if (build_options.enable_gpu)
    @import("features/gpu/mod.zig")    // Real implementation
else
    @import("features/gpu/stub.zig");  // Returns error.FeatureDisabled
```

This means:
- Every feature directory has `mod.zig` (real) and `stub.zig` (stub)
- `mod.zig` and `stub.zig` must keep matching public signatures
- Test both paths: `zig build -Denable-<feature>=true` and `=false`
- Disabled features have zero binary overhead

### Module Hierarchy

```
src/abi.zig              → Public API, comptime feature selection, type aliases
src/core/                → Framework lifecycle, config builder, registry
src/features/<name>/     → mod.zig + stub.zig per feature (8 modules: ai, analytics, cloud, database, gpu, network, observability, web)
src/services/            → Always-available infrastructure (runtime, platform, shared, ha, tasks)
tools/cli/               → Primary CLI entry point and command registration
src/api/                 → Additional executable entry points (e.g., `main.zig`)
```

Import convention: public API uses `@import("abi")`, internal modules import
via their parent `mod.zig`. Never use direct file paths for cross-module imports.

### v2 Integration Map

The v2 adoption is wired through `shared` and `runtime` surfaces, not feature-local deep
imports.

| Area | Source Location | Public Access Path |
|------|------------------|--------------------|
| Primitive helpers | `src/services/shared/utils/v2_primitives.zig` | `abi.shared.utils.v2_primitives` |
| Structured errors | `src/services/shared/utils/structured_error.zig` | `abi.shared.utils.structured_error` |
| SwissMap | `src/services/shared/utils/swiss_map.zig` | `abi.shared.utils.swiss_map` |
| ABIX serialization | `src/services/shared/utils/abix_serialize.zig` | `abi.shared.utils.abix_serialize` |
| Profiler / benchmark | `src/services/shared/utils/{profiler,benchmark}.zig` | `abi.shared.utils.{profiler,benchmark}` |
| Arena/combinator allocators | `src/services/shared/utils/memory/{arena_pool,combinators}.zig` | `abi.shared.memory.{ArenaPool,FallbackAllocator}` |
| Vyukov channel | `src/services/runtime/concurrency/channel.zig` | `abi.runtime.Channel` |
| Work-stealing thread pool | `src/services/runtime/scheduling/thread_pool.zig` | `abi.runtime.ThreadPool` |
| DAG pipeline scheduler | `src/services/runtime/scheduling/dag_pipeline.zig` | `abi.runtime.DagPipeline` |

When updating any entry above, verify import-chain stability:
`src/abi.zig` -> `src/services/{shared,runtime}/mod.zig` -> sub-module.

### Framework Lifecycle

The `Framework` struct (`src/core/framework.zig`) manages feature initialization through
a state machine: `uninitialized → initializing → running → stopping → stopped` (or `failed`).

Three initialization patterns:
```zig
// 1. Default (all compile-time features enabled)
var fw = try abi.initDefault(allocator);

// 2. Custom config
var fw = try abi.initWithConfig(allocator, .{ .gpu = .{ .backend = .vulkan } });

// 3. Builder pattern
var fw = try abi.Framework.builder(allocator)
    .withGpuDefaults()
    .withAiDefaults()
    .build();
```

### Legacy Compatibility Layer

`src/abi.zig` re-exports many types at the top level for backward compatibility:
`Gpu`, `GpuConfig`, `TransformerConfig`, `DiscordClient`, etc. For new code, prefer
the namespaced versions (`abi.gpu.Gpu`, `abi.ai.TransformerConfig`). Deprecated
functions `createDefaultFramework` and `createFramework` wrap the new `initDefault`
and `initWithConfig`.

## AI Module (Largest Module — 255 files)

Located at `src/features/ai/`. Has 17 submodules that follow the stub pattern
(agents, database, documents, embeddings, eval, explore, llm, memory, models,
multi_agent, orchestration, personas, rag, streaming, templates, training, vision)
and 6 that don't (abbey, core, prompts, tools, transformer, federated).

The `abbey/` submodule is the advanced reasoning system with meta-learning,
self-reflection, theory of mind, and neural attention mechanisms.

## GPU Module

`src/features/gpu/` supports 11 backends through a vtable-based abstraction in
`backends/`. The `dsl/` directory provides a kernel DSL with codegen targeting
SPIR-V and other backends. `mega/` handles multi-GPU orchestration.

Prefer one primary backend to avoid conflicts. On macOS, `metal` is the natural
choice. WASM targets auto-disable `database`, `network`, and `gpu`.

## Key File Locations

| Need to... | Look at |
|------------|---------|
| Add/modify public API | `src/abi.zig` |
| Change build flags | `build.zig` |
| Add a new feature module | 8 files: `mod.zig` + `stub.zig`, `build.zig` (5 places), `src/abi.zig`, `src/core/config/mod.zig`, `src/core/registry/types.zig`, `src/core/framework.zig`, `src/services/tests/parity/mod.zig`. **Verify:** `zig build validate-flags` |
| Add a CLI command | `tools/cli/commands/`, register in `tools/cli/main.zig` |
| Add config for a feature | `src/core/config/` |
| Write integration tests | `src/services/tests/` |
| Add a GPU backend | `src/features/gpu/backends/` |
| Security infrastructure | `src/services/shared/security/` (16 modules) |
| Generate API docs | `zig build gendocs` → `docs/api/` |
| Examples | `examples/` (19 examples) |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ABI_OPENAI_API_KEY` | OpenAI connector |
| `ABI_ANTHROPIC_API_KEY` | Claude connector |
| `ABI_OLLAMA_HOST` | Ollama host (default: `http://127.0.0.1:11434`) |
| `ABI_OLLAMA_MODEL` | Default Ollama model |
| `ABI_HF_API_TOKEN` | HuggingFace token |
| `ABI_MASTER_KEY` | Secrets encryption (production) |
| `DISCORD_BOT_TOKEN` | Discord bot token |

## Coding Style

- 4 spaces, no tabs; lines under 100 chars
- `PascalCase` types, `camelCase` functions/variables, `*Config` for config structs
- Explicit imports only (no `usingnamespace`); prefer `std.ArrayListUnmanaged`
- Always `zig fmt .` before committing
- Import public API via `@import("abi")`, not deep file paths
- Feature modules cannot `@import("abi")` (circular) — use relative imports to `services/shared/`
- `std.log.*` in library code; `std.debug.print` only in CLI tools and display functions
- For null-terminated C strings: `std.fmt.allocPrintSentinel(alloc, fmt, args, sentinel)` or use string literal `.ptr` (which is `[*:0]const u8`)

## Commit Convention

`<type>: <short summary>` — types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
Keep commits focused; don't mix refactors with behavior changes.

## Testing Patterns

**Current baseline**: 983 pass, 5 skip (988 total). **This baseline must be maintained** — any
PR that reduces passing tests or increases skipped tests requires justification.

**Test root**: `src/services/tests/mod.zig` (NOT `src/abi.zig`). Feature tests are
discovered through the `abi` import chain. Cannot `@import()` outside the test module
path — use `abi.<feature>` instead.

- Unit tests: `*_test.zig` files alongside code
- Integration/stress/chaos/parity/property tests: `src/services/tests/`
- Skip hardware-gated tests with `error.SkipZigTest`
- Parity tests verify `mod.zig` and `stub.zig` export the same interface

## References

- `AGENTS.md` — Project structure overview and v2 module notes
- `CONTRIBUTING.md` — Development workflow and PR checklist
- `docs/api/` — Auto-generated API docs (`zig build gendocs`)
