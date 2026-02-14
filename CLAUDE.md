# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

| Key | Value |
|-----|-------|
| **Zig** | `0.16.0-dev.2535+b5bd49460` or newer (pinned in `.zigversion`) |
| **Entry Point** | `src/abi.zig` |
| **Version** | 0.4.0 |
| **Test baseline** | 1220 pass, 5 skip (1225 total) — must be maintained |
| **Feature tests** | 684 pass (684 total) — `zig build feature-tests` |
| **CLI commands** | 28 commands + 7 aliases |

## Build & Test Commands

Ensure your system `zig` matches `.zigversion` (e.g. via `zvm use master`).

```bash
zig build                                    # Build with default flags
zig build test --summary all                 # Run full test suite
zig build feature-tests --summary all        # Run feature module inline tests
zig test src/path/to/file.zig                # Test a single file
zig test src/services/tests/mod.zig --test-filter "pattern"  # Filter tests by name
zig fmt .                                    # Format all source
zig build full-check                         # Format + tests + feature tests + flag validation + CLI smoke tests
zig build validate-flags                     # Compile-check 30 feature flag combos
zig build cli-tests                          # CLI smoke tests (top-level + nested, e.g. help llm, bench micro hash)
zig build lint                               # CI formatting check
zig build benchmarks                         # Performance benchmarks
zig build bench-all                          # Run all benchmark suites
zig build examples                           # Build all examples
zig build check-wasm                         # Check WASM compilation
zig build verify-all                         # full-check + version script + examples + bench-all + check-wasm
scripts/check_zig_version_consistency.sh     # Verify .zigversion matches build.zig/docs
zig std                                     # View/get current std library (stdlib source path / docs)
```

### Running the CLI

```bash
zig build run -- --help                      # CLI help (28 commands + 7 aliases)
zig build run -- system-info                 # System and feature status
zig build run -- plugins list                # List plugins
zig build run -- mcp serve                   # Start MCP server (stdio JSON-RPC)
zig build run -- mcp tools                   # List MCP tools
zig build run -- acp card                    # Print agent card JSON
zig build run -- serve -m model.gguf         # Alias for `llm serve`
```

### Feature Flags

`zig build -Denable-ai=true -Denable-gpu=false -Dgpu-backend=vulkan,cuda`

All features default to `true` except `-Denable-mobile`. GPU backends: `auto`, `none`,
`cuda`, `vulkan`, `metal`, `stdgpu`, `webgpu`, `tpu`, `webgl2`, `opengl`, `opengles`, `fpga`, `simulated`.
Neural networks can use GPU (CUDA/Metal/Vulkan), WebGPU, TPU (when runtime linked), or multi-threaded CPU via `abi.runtime.ThreadPool` / `InferenceConfig.num_threads`.
The `simulated` backend is always enabled as a software fallback for testing without GPU hardware.

| Feature Module | Build Flag | Notes |
|----------------|------------|-------|
| `ai` | `-Denable-ai` | Also: `-Denable-llm`, `-Denable-vision`, `-Denable-explore` |
| `ai_core` | `-Denable-ai` | Agents, tools, prompts, personas, memory |
| `inference` | `-Denable-llm` | LLM, embeddings, vision, streaming, transformer |
| `training` | `-Denable-training` | Training pipelines, federated learning |
| `reasoning` | `-Denable-reasoning` | Abbey, RAG, eval, templates, orchestration |
| `analytics` | `-Denable-analytics` | |
| `auth` | `-Denable-auth` | Re-exports `services/shared/security/` (16 modules) |
| `cache` | `-Denable-cache` | In-memory LRU/LFU, TTL, eviction |
| `cloud` | `-Denable-cloud` | Own flag (decoupled from web) |
| `database` | `-Denable-database` | |
| `gpu` | `-Denable-gpu` | Backend selection via `-Dgpu-backend=` |
| `messaging` | `-Denable-messaging` | Event bus, pub/sub, message queues |
| `mobile` | `-Denable-mobile` | Defaults to `false` |
| `network` | `-Denable-network` | |
| `observability` | `-Denable-profiling` | Not `-Denable-observability` |
| `search` | `-Denable-search` | Full-text search |
| `storage` | `-Denable-storage` | Unified file/object storage |
| `gateway` | `-Denable-gateway` | API gateway: routing, rate limiting, circuit breaker |
| `web` | `-Denable-web` | |

## Critical Gotchas

**Top 5 (cause 80% of failures):**

1. `std.fs.cwd()` → `std.Io.Dir.cwd()` (requires I/O backend init)
2. Editing `mod.zig` without updating `stub.zig` → always keep signatures in sync
3. `defer allocator.free(x)` then `return x` → use `errdefer` (use-after-free)
4. `@tagName(x)` / `@errorName(e)` in format → use `{t}` specifier
5. `std.io.fixedBufferStream()` → removed; use `std.Io.Writer.fixed(&buf)`

**Full reference:**

| Mistake | Fix |
|---------|-----|
| `std.fs.cwd()` | `std.Io.Dir.cwd()` — Zig 0.16.0-dev.2535+b5bd49460 moved filesystem to I/O backend |
| `std.time.Instant.now()` for elapsed time | `std.time.Timer.start()` — use Timer for benchmarks/elapsed |
| `list.init()` | `std.ArrayListUnmanaged(T).empty` |
| `@tagName(x)` / `@errorName(e)` in format | `{t}` format specifier for errors and enums |
| Editing `mod.zig` only | Update `stub.zig` to match exported signatures |
| `std.fs.cwd().openFile(...)` | Must init `std.Io.Threaded` first and pass `io` handle |
| `file.read()` / `file.write()` | `file.reader(io, &buf).read()` / `file.writer(io, &buf).write()` — I/O ops need `io` handle + read buffer |
| `std.time.sleep()` | `abi.shared.time.sleepMs()` / `sleepNs()` for cross-platform |
| `std.time.nanoTimestamp()` | Doesn't exist in `0.16.0-dev.2535+b5bd49460` — use `Instant.now()` + `.since(anchor)` for absolute time |
| `std.process.getEnvVar()` | Doesn't exist in `0.16.0-dev.2535+b5bd49460` — use `std.c.getenv()` for POSIX |
| `@typeInfo` tags `.Type`, `.Fn` | Lowercase in `0.16.0-dev.2535+b5bd49460`: `.type`, `.@"fn"`, `.@"struct"`, `.@"enum"`, `.@"union"` |
| `b.createModule()` for named modules | `b.addModule("name", ...)` — `createModule` is anonymous |
| `defer allocator.free(x)` then return `x` | Use `errdefer` — `defer` frees on success too (use-after-free). Applies anywhere caller takes ownership: `loadFromEnv()`, builder patterns, etc. |
| `@panic` in library code | Return an error instead — library code should never panic |
| `std.time.Timer.read()` → `u64` | Returns `usize` in `0.16.0-dev.2535+b5bd49460`, not `u64` — cast or use `@as(u64, timer.read())` |
| `std.log.err` in tests | Test runner treats error-level log output as a test failure, even if caught. Skip the test before entering error paths |
| `opaque` as identifier | `opaque` is a keyword in 0.16 — use `is_opaque` or `@"opaque"` |
| `FallbackAllocator` double-free | Can't call `rawFree` on both backing allocators — use `rawResize(..0..)` to probe ownership |
| `Timer.start() catch` with bogus fallback | `catch std.time.Timer{ .started = .{...} }` produces wrong `.read()` values — only acceptable because `Timer.start()` virtually never fails |
| SwissMap `@typeInfo(.int)` branch | Matches all integer types — explicit checks after it are unreachable |
| `std.io.fixedBufferStream()` | Removed — use `std.Io.Writer.fixed(&buf)`, read via `buf[0..writer.end]` |
| `ArrayListUnmanaged.writer()` | Doesn't exist in 0.16 — build output manually with `appendSlice` |
| `comptime { _ = @import(...); }` for tests | Doesn't discover tests — use `test { _ = @import(...); }` blocks |

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

### Stdio I/O (for CLI tools reading stdin/stdout)

```zig
var stdin_file = std.Io.File.stdin();
var read_buf: [65536]u8 = undefined;
var reader = stdin_file.reader(io, &read_buf);

var stdout_file = std.Io.File.stdout();
var write_buf: [65536]u8 = undefined;
var writer = stdout_file.writer(io, &write_buf);
// Always flush after writing: writer.flush()
```

### Fixed-Buffer Writer (for tests — replaces removed `std.io.fixedBufferStream`)

```zig
var buf: [256]u8 = undefined;
var writer = std.Io.Writer.fixed(&buf);
try writer.writeAll("hello");
const written = buf[0..writer.end]; // "hello"
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
build.zig                → Top-level build script (delegates to build/)
build/                   → Split build system (options, modules, flags, targets, gpu, mobile, wasm)
src/abi.zig              → Public API, comptime feature selection, type aliases
src/core/                → Framework lifecycle, config builder, registry
src/features/<name>/     → mod.zig + stub.zig per feature (15 core + 4 AI split = 19 modules)
src/services/            → Always-available infrastructure (runtime, platform, shared, ha, tasks)
src/services/mcp/        → MCP server (JSON-RPC 2.0 over stdio, WDBX tools)
src/services/acp/        → ACP server (agent communication protocol)
src/services/connectors/ → LLM provider connectors (8 providers + discord + scheduler)
tools/cli/               → Primary CLI entry point and command registration (28 commands)
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

Two initialization patterns:
```zig
// 1. Default (all compile-time features enabled)
var fw = try abi.initDefault(allocator);

// 2. Custom config
var fw = try abi.init(allocator, .{ .gpu = .{ .backend = .vulkan } });
```

### Convenience Aliases

`src/abi.zig` re-exports `Gpu` and `GpuBackend` at the top level. All other
functions use namespaced paths: `abi.simd.vectorAdd`, `abi.simd.hasSimdSupport`,
`abi.connectors.discord`.

## AI Modules

The AI feature is split into 5 independent modules:
- `ai` (`src/features/ai/`) — Full monolith (17 submodules with stubs + 6 without)
- `ai_core` (`src/features/ai_core/`) — Agents, tools, prompts, personas, memory
- `inference` (`src/features/ai_inference/`) — LLM, embeddings, vision, streaming
- `training` (`src/features/ai_training/`) — Training pipelines, federated learning
- `reasoning` (`src/features/ai_reasoning/`) — Abbey, RAG, eval, templates, orchestration

The `abbey/` submodule is the advanced reasoning system with meta-learning,
self-reflection, theory of mind, and neural attention mechanisms.

**System-level data types (Zig 0.16):** Models can be trained to process and generate
all types (text, images, video, audio, documents, any). Core config exposes
`abi.config.ContentKind` and `TrainingConfig.enable_vision` / `enable_video` /
`enable_audio` / `enable_all_modalities`. Use `abi.ai.training.selfLearningConfigFromCore(core_training_config)` to flow system config into self-learning.

## GPU Module

`src/features/gpu/` supports 10 backends through a vtable-based abstraction in
`backends/`. The `dsl/` directory provides a kernel DSL with codegen targeting
SPIR-V and other backends. `mega/` handles multi-GPU orchestration.

Prefer one primary backend to avoid conflicts. On macOS, `metal` is the natural
choice. WASM targets auto-disable `database`, `network`, and `gpu`.

## Connectors

`src/services/connectors/` provides LLM provider integrations accessed via `abi.connectors`:
- 8 LLM providers: `openai`, `anthropic`, `ollama`, `huggingface`, `mistral`, `cohere`, `lm_studio`, `vllm`
- Discord REST client: `abi.connectors.discord`
- Job scheduler: `abi.connectors.local_scheduler`
- Local servers (LM Studio, vLLM) use OpenAI-compatible `/v1/chat/completions` endpoint
- All expose `isAvailable()` for zero-allocation env var checks
- Shared types in `shared.zig`: `ChatMessage`, `Role`, `ConnectorError`, `secureFree`
- All use `model_owned: bool` for ownership tracking (prevents use-after-free in `loadFromEnv`)

## Key File Locations

| Need to... | Look at |
|------------|---------|
| Add/modify public API | `src/abi.zig` |
| Change build flags | `build.zig` + `build/options.zig`, `build/flags.zig` |
| Add a new feature module | See checklist below (8 integration points) |
| Add a CLI command | `tools/cli/commands/`, register in `tools/cli/main.zig` |
| Add config for a feature | `src/core/config/` |
| Write integration tests | `src/services/tests/` |
| Add a GPU backend | `src/features/gpu/backends/` |
| Security infrastructure | `src/services/shared/security/` (16 modules) |
| C API bindings | `bindings/c/src/abi_c.zig` (36 exports) |
| Generate API docs | `zig build gendocs` → `docs/api/` |
| Examples | `examples/` (22 examples) |
| MCP service | `src/services/mcp/` (JSON-RPC 2.0 server for WDBX) |
| ACP service | `src/services/acp/` (agent communication protocol) |

### Adding a New Feature Module (8 integration points)

1. `build/options.zig` — add `enable_<name>` field + CLI option
2. `build/flags.zig` — add to `FlagCombo`, `validation_matrix`, `comboToBuildOptions()`
3. `src/features/<name>/mod.zig` + `stub.zig` — implementation + disabled stub
4. `src/abi.zig` — comptime conditional import
5. `src/core/config/mod.zig` — Feature enum, description, Config field, Builder methods, validation
6. `src/core/registry/types.zig` — `isFeatureCompiledIn` switch case
7. `src/core/framework.zig` — import, context field, init/deinit, getter, builder
8. `src/services/tests/stub_parity.zig` — basic parity test

**Verify:** `zig build validate-flags`

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
| `ABI_LM_STUDIO_HOST` | LM Studio host (default: `http://localhost:1234`) |
| `ABI_LM_STUDIO_MODEL` | Default LM Studio model |
| `ABI_LM_STUDIO_API_KEY` | LM Studio API key (optional) |
| `ABI_VLLM_HOST` | vLLM host (default: `http://localhost:8000`) |
| `ABI_VLLM_MODEL` | Default vLLM model |
| `ABI_VLLM_API_KEY` | vLLM API key (optional) |

## Coding Style

- 4 spaces, no tabs; lines under 100 chars
- `PascalCase` types, `camelCase` functions/variables, `*Config` for config structs
- Explicit imports only (no `usingnamespace`); prefer `std.ArrayListUnmanaged`
- Always `zig fmt .` before committing
- Import public API via `@import("abi")`, not deep file paths
- Feature modules cannot `@import("abi")` (circular) — use relative imports to `services/shared/`
- `std.log.*` in library code; `std.debug.print` only in CLI tools and display functions
- Modern format specifiers: `{t}` for enums/errors, `{B}`/`{Bi}` for byte sizes, `{D}` for durations, `{b64}` for base64
- For null-terminated C strings: `std.fmt.allocPrintSentinel(alloc, fmt, args, sentinel)` or use string literal `.ptr` (which is `[*:0]const u8`)

## Commit Convention

`<type>: <short summary>` — types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
Keep commits focused; don't mix refactors with behavior changes.

## Testing Patterns

**Main tests**: 1220 pass, 5 skip (1225 total) — `zig build test --summary all`
**Feature tests**: 684 pass (684 total) — `zig build feature-tests --summary all`
Both baselines must be maintained.

**Two test roots** (each is a separate binary with its own module path):
- `src/services/tests/mod.zig` — main tests; discovers tests via `abi.<feature>` import chain
- `src/feature_test_root.zig` — feature inline tests; can `@import("features/...")` and `@import("services/...")` directly

**Why two roots?** Module path restrictions prevent `src/services/tests/mod.zig` from importing
`src/services/mcp/server.zig` (outside its module path). The feature test root at `src/` level
can reach both `features/` and `services/` subdirectories.

- Unit tests: `*_test.zig` files alongside code
- Integration/stress/chaos/parity/property tests: `src/services/tests/`
- Skip hardware-gated tests with `error.SkipZigTest`
- Parity tests verify `mod.zig` and `stub.zig` export the same interface
- **Test discovery**: Use `test { _ = @import(...); }` to include submodule tests — `comptime {}` does NOT discover tests

## After Making Changes

| Changed... | Run |
|------------|-----|
| Any `.zig` file | `zig fmt .` |
| Feature `mod.zig` | Also update `stub.zig`, then `zig build -Denable-<feature>=false` |
| Feature inline tests | `zig build feature-tests --summary all` (must stay at 684+) |
| Build flags / options | `zig build validate-flags` |
| Public API | `zig build test --summary all` + update examples |
| Anything (full gate) | `zig build full-check` |
| Build artifacts in `exe/` | Add `exe/` to `.gitignore` (see .gitignore) — standard output is `zig-out/` |

## Claude Code Configuration

- **MCP servers** (`.mcp.json`): `zig-docs` — Zig stdlib documentation lookup
- **Hooks** (`.claude/settings.json`): Auto-format `.zig` on save, sensitive file guard, force-push blocker, test baseline guard
- **Project agents** (`.claude/agents/`): 12 specialized agents for this repo
- **Project skills** (`.claude/skills/`): 6 skills including `/validate`, `/stub-sync`, `/feature-module`

## References

- `AGENTS.md` — Project structure overview and v2 module notes
- `CONTRIBUTING.md` — Development workflow and PR checklist
- `.claude/rules/zig.md` — Zig 0.16 rules enforced by Claude Code (overlaps with gotchas above)
- `docs/api/` — Auto-generated API docs (`zig build gendocs`)
