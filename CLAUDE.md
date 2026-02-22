# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Status: ready for use.** Single entry for AI assistants (Claude, Codex, Cursor). For rules, skills, plans, and agents see [Skills, Plans, and Agents](#skills-plans-and-agents-full-index).

## Quick Reference

| Key | Value |
|-----|-------|
| **Zig** | `0.16.0-dev.2623+27eec9bd6` or newer (pinned in `.zigversion`) |
| **Entry Point** | `src/abi.zig` |
| **Version** | 0.4.0 |
| **Test baseline** | 1261 pass, 5 skip (1266 total) — must be maintained |
| **Feature tests** | 2119 pass (2123 total), 4 skip — `zig build feature-tests` |
| **CLI commands** | 30 commands + 8 aliases |
| **Feature modules** | 21 (comptime-gated; see Feature Flags) |

## Build & Test Commands

Ensure your system `zig` matches `.zigversion`.

```bash
# One-time/periodic toolchain sync
zvm upgrade

# Developer convenience (latest dev build)
zvm use master

# Reproducible/local CI parity (pinned in repo)
PINNED_ZIG="$(cat .zigversion)"
zvm install "$PINNED_ZIG"
zvm use "$PINNED_ZIG"

# Validate active binary and pinned version
which zig
zig version
cat .zigversion
zig run tools/scripts/toolchain_doctor.zig

# If needed, fix shell precedence:
export PATH="$HOME/.zvm/bin:$PATH"
```

```bash
zig build                                    # Build with default flags
zig build test --summary all                 # Run full test suite
zig build feature-tests --summary all        # Run feature module inline tests
zig test src/path/to/file.zig                # Test a single file
zig test src/services/tests/mod.zig --test-filter "pattern"  # Filter tests by name
zig fmt .                                    # Format all source
zig build full-check                         # Format + tests + feature tests + flag validation + CLI smoke tests
zig build toolchain-doctor                   # Diagnose local Zig PATH/version drift vs .zigversion
zig build validate-flags                     # Compile-check 34 feature flag combos
zig build cli-tests                          # CLI smoke tests (top-level + nested, e.g. help llm, bench micro hash)
zig build lint                               # CI formatting check
zig build fix                                # Format source files in place
zig build examples                           # Build all examples
zig build check-wasm                         # Check WASM compilation
zig build verify-all                         # full-check + consistency + examples + check-wasm
zig build ralph-gate                         # Require live Ralph report and threshold pass
tools/scripts/check_zig_version_consistency.zig     # Verify .zigversion matches build.zig/docs
zig run tools/scripts/toolchain_doctor.zig          # Diagnose PATH precedence and active zig mismatch
zig std                                      # Print stdlib source path (useful for reading std lib internals)
```

### Running the CLI

```bash
zig build run -- --help                      # CLI help (30 commands + 8 aliases)
zig build run -- system-info                 # System and feature status (incl. Feature Matrix)
zig build run -- --list-features             # List features (COMPILED/DISABLED) and exit
zig build run -- status                      # Framework health and feature count
zig build run -- plugins list                # List plugins
zig build run -- mcp serve                   # Start MCP server (stdio JSON-RPC)
zig build run -- mcp tools                   # List MCP tools
zig build run -- acp card                    # Print agent card JSON
zig build run -- serve -m model.gguf         # Alias for `llm serve`
```

Feature-related commands and flags: [Features and the CLI](#features-and-the-cli).

## After Making Changes

| Changed... | Run |
|------------|-----|
| Any `.zig` file | `zig fmt .` |
| Feature `mod.zig` | Also update `stub.zig`, then `zig build -Denable-<feature>=false` |
| Feature inline tests | `zig build feature-tests --summary all` (must stay at 2082+) |
| Build flags / options | `zig build validate-flags` |
| Public API | `zig build test --summary all` + update examples |
| Anything (full gate) | `zig build full-check` |
| Everything (release gate) | `zig build verify-all` (full-check + consistency + examples + check-wasm) |
| Build artifacts in `exe/` | Add `exe/` to `.gitignore` (see .gitignore) — standard output is `zig-out/` |

## Critical Gotchas

**Top 7 (cause 80% of failures):**

1. `std.fs.cwd()` → `std.Io.Dir.cwd()` (requires I/O backend init)
2. Editing `mod.zig` without updating `stub.zig` → always keep signatures in sync
3. `defer allocator.free(x)` then `return x` → use `errdefer` (use-after-free)
4. `@tagName(x)` / `@errorName(e)` in format → use `{t}` specifier
5. `std.io.fixedBufferStream()` → removed; use `std.Io.Writer.fixed(&buf)`
6. `@field(build_options, field_name)` requires comptime context — use `inline for` not runtime `for`
7. **API break (v0.4.0):** Facade aliases and flat exports removed — see [AI Modules](#ai-modules) and [GPU Module](#gpu-module) for the new namespaced access patterns

**Full reference (19 entries):** `.claude/rules/zig.md` is auto-loaded for all `.zig` files. It covers I/O, `@typeInfo`, allocators, test discovery, format specifiers, and more.

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
src/abi.zig              → Public API, comptime feature selection (no flat type aliases)
src/core/                → Framework lifecycle, config builder, registry
src/features/<name>/     → mod.zig + stub.zig per feature (17 core + 4 AI split = 21 modules)
src/services/            → Always-available infrastructure (runtime, platform, shared, ha, tasks)
src/services/mcp/        → MCP server (JSON-RPC 2.0 over stdio, WDBX tools)
src/services/acp/        → ACP server (agent communication protocol)
src/services/connectors/ → LLM provider connectors (9 providers + discord + scheduler)
tools/cli/               → Primary CLI entry point and command registration (30 commands)
src/api/                 → Additional executable entry points (e.g., `main.zig`)
```

Import convention: public API uses `@import("abi")`, internal modules import
via their parent `mod.zig`. Never use direct file paths for cross-module imports.

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

### Access Patterns

All access uses namespaced paths through `abi.<module>`. There are no top-level
convenience aliases or flat re-exports. Examples: `abi.gpu.unified.MatrixDims`,
`abi.ai.agent.Agent`, `abi.simd.vectorAdd`, `abi.connectors.discord`.

**Removed (v0.4.0):** `abi.ai_core`, `abi.inference`, `abi.training`, `abi.reasoning`
(facade aliases); ~156 flat AI type aliases (e.g., `abi.ai.Agent`); ~173 flat GPU type
aliases (e.g., `abi.gpu.MatrixDims`). Use submodule paths instead.

### vNext Migration (Staged)

`src/vnext/` provides a forward API surface wrapping `Framework`:
- `abi.vnext.App` — thin wrapper around Framework (init, deinit, getFramework)
- `abi.vnext.AppConfig` — config with `strict_capability_check` + `required_capabilities`
- `abi.vnext.Capability` — mirrors `Feature` enum with conversion functions

**Current state (v0.4.0):** Capability checking and basic `App` methods (`.feature()`,
`.has()`, `.state()`) are implemented in `src/vnext/app.zig`. Use `app.getFramework()`
to access the full Framework during transition. Compatibility tests: `zig build vnext-compat`.

## AI Modules

All AI code lives under `src/features/ai/`. AI sub-features are accessed as
submodules of `abi.ai`, each independently gated by its build flag:
- `abi.ai.core` — Agents, tools, prompts, personas, memory (`-Denable-ai`)
- `abi.ai.llm` — LLM, embeddings, vision, streaming (`-Denable-llm`)
- `abi.ai.training` — Training pipelines, federated learning (`-Denable-training`)
- `abi.ai.reasoning` — Abbey, RAG, eval, templates, orchestration (`-Denable-reasoning`)

Types are accessed via their submodule, e.g., `abi.ai.agent.Agent`,
`abi.ai.training.TrainingConfig`, `abi.ai.tools.Tool`.

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

Types are accessed via their submodule, e.g., `abi.gpu.unified.MatrixDims`,
`abi.gpu.profiling.Profiler`. There are no flat re-exports at the `abi.gpu` level.

Prefer one primary backend to avoid conflicts. On macOS, `metal` is the natural
choice. WASM targets auto-disable `database`, `network`, and `gpu`.

## Connectors

`src/services/connectors/` provides LLM provider integrations accessed via `abi.connectors`:
- 9 LLM providers: `openai`, `anthropic`, `ollama`, `huggingface`, `mistral`, `cohere`, `lm_studio`, `vllm`, `mlx`
- Discord REST client: `abi.connectors.discord`
- Job scheduler: `abi.connectors.local_scheduler`
- Local servers (LM Studio, vLLM) use OpenAI-compatible `/v1/chat/completions` endpoint
- All expose `isAvailable()` for zero-allocation env var checks
- Shared types in `shared.zig`: `ChatMessage`, `Role`, `ConnectorError`, `secureFree`
- All use `model_owned: bool` for ownership tracking (prevents use-after-free in `loadFromEnv`)

## Coding Style

- 4 spaces, no tabs; lines under 100 chars
- `PascalCase` types, `camelCase` functions/variables, `*Config` for config structs
- Explicit imports only (no `usingnamespace`)
- Prefer `std.ArrayListUnmanaged` over `std.ArrayList` — unmanaged passes allocator per-call, giving better control over memory ownership and reducing hidden allocator dependencies:
  ```zig
  // Preferred
  var list: std.ArrayListUnmanaged(u8) = .empty;
  defer list.deinit(allocator);
  try list.append(allocator, item);

  // Avoid (hides allocator dependency)
  var list: std.ArrayList(u8) = .empty;
  ```
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

**Main tests**: 1261 pass, 5 skip (1266 total) — `zig build test --summary all`
**Feature tests**: 2119 pass (2123 total), 4 skip — `zig build feature-tests --summary all`
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
- **Standalone test files**: For modules whose `mod.zig` re-exports sub-modules with compile issues (gpu, auth, database), create `*_test.zig` alongside mod.zig that imports only the parent — avoids triggering lazy compilation of broken sub-modules
- **GPU/database test gap**: These modules cannot be registered in `feature_test_root.zig` — backend source files have Zig 0.16 compatibility issues (`*const DynLib`, stale struct fields, extern enum tag width). They compile fine through `zig build test` via the named `abi` module. Needs dedicated migration pass.

## Key File Locations

| Need to... | Look at |
|------------|---------|
| Add/modify public API | `src/abi.zig` |
| Change build flags | `build.zig` + `build/options.zig`, `build/flags.zig` |
| Feature catalog (canonical list) | `src/core/feature_catalog.zig` |
| Add a new feature module | See checklist below (9 integration points) |
| Add a CLI command | `tools/cli/commands/`, register in `tools/cli/main.zig` |
| Feature listing / system-info | `tools/cli/commands/system_info.zig` (Feature Matrix), `tools/cli/utils/global_flags.zig` (`--list-features`, `--enable-*`, `--disable-*`) |
| Add config for a feature | `src/core/config/` |
| Write integration tests | `src/services/tests/` |
| Add a GPU backend | `src/features/gpu/backends/` |
| Security infrastructure | `src/services/shared/security/` (17 modules) |
| C API bindings | `zig build c-header` → `zig-out/include/abi.h`; `zig build lib` → static library |
| Generate API docs | `abi gendocs` (CLI command) |
| Examples | `examples/` (37 examples, including `training/train_demo.zig`) |
| MCP service | `src/services/mcp/` (JSON-RPC 2.0 server for WDBX) |
| ACP service | `src/services/acp/` (agent communication protocol) |

### Adding a New Feature Module (9 integration points)

1. `build/options.zig` — add `enable_<name>` field + CLI option
2. `build/flags.zig` — add to `FlagCombo`, `validation_matrix`, `comboToBuildOptions()`
3. `src/core/feature_catalog.zig` — add `Feature` enum variant, `ParitySpec` if needed, and `Metadata` entry (description, compile flag, real/stub paths)
4. `src/features/<name>/mod.zig` + `stub.zig` — implementation + disabled stub
5. `src/abi.zig` — comptime conditional import
6. `src/core/config/mod.zig` — Feature enum, description, Config field, Builder methods, validation
7. `src/core/registry/types.zig` — `isFeatureCompiledIn` switch case
8. `src/core/framework.zig` — import, context field, init/deinit, getter, builder
9. `src/services/tests/stub_parity.zig` — basic parity test

**When editing an existing feature:** (1) Update `stub.zig` to match any `mod.zig` API change. (2) Build and test with the feature off and on: `zig build -Denable-<name>=false` and `zig build -Denable-<name>=true`. (3) Run `zig build validate-flags` and `zig build full-check`.

**Stub conventions**: Use anonymous parameter discard (`_: Type`) instead of multi-line
`_ = param;` blocks. Keep function bodies on one line where possible: `pub fn foo(_: *@This(), _: []const u8) !void { return error.FeatureDisabled; }`.
Use `StubContext(ConfigT)` from `src/core/stub_context.zig` when the stub defines a Context struct.

**Shared infrastructure**: Use `services/shared/resilience/circuit_breaker.zig` for circuit breakers
(parameterized by `.atomic`, `.mutex`, or `.none` sync strategy). Use `services/shared/security/rate_limit.zig`
for HTTP/API-level rate limiting with per-key tracking, bans, and whitelist.

**Verify:** `zig build validate-flags`

## Feature Flags

The codebase has **21 comptime-gated feature modules**. Each is switched by a build flag; when disabled, the stub is linked and the feature returns `error.FeatureDisabled`. The canonical list lives in `src/core/feature_catalog.zig`; build options in `build/options.zig`.

**Usage:** `zig build -Denable-ai=true -Denable-gpu=false -Dgpu-backend=vulkan,cuda`

All features default to `true` except `-Denable-mobile`. GPU backends: `auto`, `none`,
`cuda`, `vulkan`, `metal`, `stdgpu`, `webgpu`, `tpu`, `webgl2`, `opengl`, `opengles`, `fpga`, `simulated`.
The `simulated` backend is always enabled as a software fallback for testing without GPU hardware.

**By area:**

| Feature | Build Flag | Notes |
|---------|------------|-------|
| *AI* | | |
| `ai` | `-Denable-ai` | Also: `-Denable-llm`, `-Denable-vision`, `-Denable-explore` |
| `ai.core` | `-Denable-ai` | Agents, tools, prompts, personas, memory (access: `abi.ai.core`) |
| `ai.llm` | `-Denable-llm` | LLM, embeddings, vision, streaming, transformer (access: `abi.ai.llm`) |
| `ai.training` | `-Denable-training` | Training pipelines, federated learning (access: `abi.ai.training`) |
| `ai.reasoning` | `-Denable-reasoning` | Abbey, RAG, eval, templates, orchestration (access: `abi.ai.reasoning`) |
| *Infrastructure* | | |
| `analytics` | `-Denable-analytics` | |
| `auth` | `-Denable-auth` | Re-exports `services/shared/security/` (17 modules) |
| `cache` | `-Denable-cache` | In-memory LRU/LFU, TTL, eviction |
| `cloud` | `-Denable-cloud` | Own flag (decoupled from web) |
| `database` | `-Denable-database` | |
| `gpu` | `-Denable-gpu` | Backend selection via `-Dgpu-backend=` |
| `messaging` | `-Denable-messaging` | Event bus, pub/sub, message queues |
| `network` | `-Denable-network` | |
| `observability` | `-Denable-profiling` | Not `-Denable-observability` |
| `search` | `-Denable-search` | Full-text search |
| `storage` | `-Denable-storage` | Unified file/object storage |
| `gateway` | `-Denable-gateway` | API gateway: routing, rate limiting, circuit breaker |
| `web` | `-Denable-web` | |
| *Platform / tooling* | | |
| `mobile` | `-Denable-mobile` | Defaults to `false` |
| `pages` | `-Denable-pages` | Dashboard/UI pages with URL path routing |
| `benchmarks` | `-Denable-benchmarks` | Built-in benchmark suite |

**Checking feature status:** Run `zig build run -- system-info` for a summary of which features are compiled in and runtime status. In code, use `build_options.enable_<name>` (comptime) or the Framework getters after init.

**When editing a feature:** Keep `mod.zig` and `stub.zig` in sync (same public API); run `zig build -Denable-<feature>=false` and `zig build -Denable-<feature>=true`; finish with `zig build validate-flags`.

### Features and the CLI

The CLI exposes feature status and lets you inspect or override (runtime) which features are enabled.

| CLI | Purpose |
|-----|---------|
| `abi system-info` (aliases: `info`, `sysinfo`) | Full report: platform, SIMD, GPU, network, AI connectors, and **Feature Matrix** (each `abi.Feature` with enabled yes/no). |
| `abi --list-features` | List all features with COMPILED / DISABLED (comptime). Exits after printing. |
| `abi status` | Framework health and feature count (e.g. "N/M features active"). |
| `abi config show` | Shows config including selected `enable_*` flags. |

**Runtime overrides (optional):** You can pass `--enable-<feature>` or `--disable-<feature>` before a command (e.g. `abi --disable-gpu bench all`). Only features that are **compiled in** can be enabled; disabled-at-build-time features cannot be turned on at runtime. Rebuild with `-Denable-<name>=true` to compile in a feature.

**Commands that require a feature:** These exit with a clear error if the feature is disabled (compile or runtime): `db` → database; `embed`, `convert`, `explore`, `multi-agent` (info/run) → ai; `discord` → web; `train` (llm/vision) → llm/vision; `bench` (database/training suites) → database/training. Use `abi system-info` or `abi --list-features` to confirm before running them.

## v2 Integration Map

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
| Shared radix tree | `src/services/shared/utils/radix_tree.zig` | Used by gateway + pages for URL routing |
| Circuit breaker | `src/services/shared/resilience/circuit_breaker.zig` | `abi.shared.resilience.{Simple,Atomic,Mutex}CircuitBreaker` |

When updating any entry above, verify import-chain stability:
`src/abi.zig` -> `src/services/{shared,runtime}/mod.zig` -> sub-module.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ABI_GPU_BACKEND` | Runtime GPU backend override (`auto`, `cuda`, `vulkan`, `metal`, `none`, etc.) |
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
| `ABI_MLX_HOST` | MLX host (default: `http://localhost:8080`) |
| `ABI_MLX_MODEL` | Default MLX model |
| `ABI_MLX_API_KEY` | MLX API key (optional) |

## Working with Ralph (Agent Loop)

You are **outside the Ralph loop** unless the user explicitly runs `abi ralph run`. Do not drive Ralph-style iterative loops or read/write `.ralph/` state in normal sessions.

**Your workflow:** Edit code → run `zig build full-check` → done. Suggest Ralph only if the user wants autonomous multi-step execution.

**Quality gates:** `zig build full-check` (standard) or `zig build verify-all` (release-grade, adds consistency + examples + wasm).

**Ralph reference:** Config in `ralph.yml`, state in `.ralph/`, CLI via `abi ralph init | run | super | multi | status | gate | improve | skills`. Power command: `abi ralph super --task "goal"`. Multi-agent: `abi ralph multi -t "g1" -t "g2"`. Skills: `.claude/skills/super-ralph/SKILL.md`.

**Learning channels:** Outside the loop, update CLAUDE.md / `.claude/rules/zig.md` / `.claude/skills/`. Inside the loop, Ralph stores lessons in Abbey semantic memory (`.skill` category) via `--auto-skill` or `abi ralph skills add "lesson"`.

## Skills, Plans, and Agents (full index)

### When to use what

| Context | Use |
|--------|-----|
| **This session (Cursor/Claude)** | CLAUDE.md, `.claude/rules/zig.md`. Edit, build, test; suggest `/baseline-sync` or `/zig-migrate` when relevant. |
| **Ralph iterative loop** | User runs `abi ralph run` or `abi ralph improve`; skills live in Abbey memory (`.ralph/`). |
| **Codex / external runner** | Same baselines; `tools/scripts/baseline.zig` is source of truth. |

### Rules (auto-loaded)

| Path | Purpose |
|------|---------|
| `.claude/rules/zig.md` | Zig 0.16 gotchas (19 entries), test baselines, I/O backend, stub conventions, import rules. Auto-loaded for all `.zig` files. |

### Custom skills (invoke by name)

| Skill | Invocation | Purpose |
|-------|------------|---------|
| **baseline-sync** | `/baseline-sync` | Sync test baseline numbers from `tools/scripts/baseline.zig` to doc files. Run after test count changes. |
| **zig-migrate** | `/zig-migrate [file-or-dir]` | Apply Zig 0.16 migration patterns (DynLib, I/O backend, format specifiers, etc.). |
| **super-ralph** | `/super-ralph` or suggest | Run Ralph: `abi ralph super --task "..."` with optional `--gate`/`--auto-skill`. |

Skill index: `.claude/skills/README.md` (if present) or list `ls .claude/skills/*/SKILL.md`.

### Ready for use — checklist

| Check | Action |
|-------|--------|
| **Entry** | Use this file (CLAUDE.md) as single entry. |
| **Baseline** | Source of truth: `tools/scripts/baseline.zig`. After test count changes run `/baseline-sync`. |
| **Gate** | Before claiming done: `zig build full-check`. |
| **Ralph** | You are outside the loop unless the user runs `abi ralph run`. |

---

## Claude Code Configuration

- **MCP servers** (`.mcp.json`): `zig-docs` — Zig stdlib documentation lookup
- **Local settings** (`.claude/settings.local.json`): MCP server enablement
- **Rules** (`.claude/rules/zig.md`): Zig 0.16 gotchas and baseline markers (auto-loaded for `.zig` files)

### Hooks (`.claude/settings.json`)

Two PostToolUse hooks fire automatically after `Edit` or `Write`:
1. **Auto-format**: Runs `zig fmt` on any `.zig` file you create or edit — no manual step needed
2. **Stub parity reminder**: Warns when you edit a `mod.zig` that has a sibling `stub.zig`

Custom skills: see [Custom skills (invoke by name)](#custom-skills-invoke-by-name) above.

## CI Quality Gate Scripts

Beyond `zig build full-check`, these scripts enforce additional invariants:

| Script | Purpose |
|--------|---------|
| `tools/scripts/check_test_baseline_consistency.zig` | Verify baseline numbers match across all 10+ doc files |
| `tools/scripts/check_zig_version_consistency.zig` | Verify `.zigversion` matches `build.zig` and docs |
| `tools/scripts/check_zig_016_patterns.zig` | Scan for deprecated Zig pre-0.16 patterns |
| `tools/scripts/check_feature_catalog.zig` | Verify feature catalog matches build options |
| `tools/scripts/check_import_rules.zig` | Enforce import conventions (no circular `@import("abi")` in features) |
| `tools/scripts/toolchain_doctor.zig` | Diagnose PATH precedence and active zig mismatch |

### Updating Test Baselines

When test counts change, update `tools/scripts/baseline.zig` (source of truth), then run
`/baseline-sync` or manually update files listed in `.claude/skills/baseline-sync/SKILL.md`.
Verify with `zig run tools/scripts/check_test_baseline_consistency.zig`.

## References

- `CONTRIBUTING.md` — Development workflow and PR checklist
- `.claude/rules/zig.md` — Zig 0.16 complete gotchas table (auto-loaded for `.zig` files)
- `.claude/skills/` — Custom skills (baseline-sync, zig-migrate); see [Skills, Plans, and Agents](#skills-plans-and-agents-full-index)
- `tools/scripts/baseline.zig` — Canonical test baseline (source of truth for CI checks)

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
