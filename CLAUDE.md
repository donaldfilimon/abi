# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Status: ready for use.** Single entry for AI assistants (Claude, Codex, Cursor). Use the Quick Reference and Build & Test Commands below; for rules, skills, plans, and agents see [Skills, Plans, and Agents](#skills-plans-and-agents-full-index).

## Working outside the Ralph loop (Cursor / Claude)

When you assist in **Cursor** (or any direct editor/chat session), you are **outside the Ralph loop**. Use this section to stay in the right mode and avoid confusing the two workflows.

### Two modes at a glance

| Aspect | Outside the loop (you, in Cursor) | Inside the loop (Ralph) |
|--------|-----------------------------------|-------------------------|
| **Trigger** | User asks you in chat; you edit and suggest commands | User runs `abi ralph run` (or `abi agent ralph --task "..."`); Abbey engine drives an iterative loop |
| **Input** | Chat messages, open files, repo context | `PROMPT.md` or `--task "..."`, plus optional `.ralph/` skills and state |
| **Flow** | Single conversational turn or short back-and-forth; you run `zig build` / tests as needed | Multi-step loop until `LOOP_COMPLETE` or max iterations; engine calls LLM, runs tools, updates state |
| **State** | No reliance on `.ralph/` or `ralph.yml` for *this* session | Uses `.ralph/` (state, lock, logs), `ralph.yml` (backend, prompt_file, max_iterations) |
| **Quality** | Validate with `zig build full-check` (or `zig build test`, `zig fmt`, etc.) | Optional: `abi ralph gate` / `zig build ralph-gate` on Ralph run outputs; `zig build verify-all` includes ralph-gate |
| **Self-improvement / learning** | Update CLAUDE.md, `.claude/rules/zig.md`, or baselines; no `.ralph/` skill store this session | Skills in Abbey memory (`.skill`); `ralph improve`, `ralph run --auto-skill`, `ralph skills add/list/clear`; lessons injected into next run’s system prompt |

### Do (outside the loop)

- **Edit and run locally:** Make changes, then run `zig build`, `zig build test`, `zig build full-check`, `zig fmt`, etc. directly. Do not invoke or depend on `abi ralph run` or `PROMPT.md` for this session.
- **Single conversational flow:** Treat this as one chat/agent session. There is no iterative Ralph loop here—no Abbey engine steps, no `LOOP_COMPLETE` promise, no `.ralph/` state machine.
- **Follow CLAUDE.md:** Use the build/test/format commands, gotchas, and conventions in this file. Stub parity, feature flags, and test baselines apply to your edits.
- **Suggest Ralph when it fits:** If the user describes a long, multi-step task that would benefit from an iterative agent run, you can suggest they try `abi ralph run --task "..."` or `abi ralph improve` as a *separate* workflow—without you driving that loop yourself.

### Don’t (outside the loop)

- **Don’t** assume you are inside a Ralph run: no reading/writing `PROMPT.md` or `.ralph/` state as part of your normal response flow.
- **Don’t** require the user to run `abi ralph run` to validate the changes you make in this session; use `zig build full-check` (and optionally `zig build verify-all`).
- **Don’t** drive multi-turn “Ralph-style” loops (e.g. “I will now run N iterations until LOOP_COMPLETE”) unless the user explicitly asks you to use the Ralph CLI.

### Quality gates for your changes

- **Standard:** `zig build full-check` (format + tests + feature tests + flag validation + CLI smoke).
- **Release-grade:** `zig build verify-all` (adds consistency checks, examples, and check-wasm).

### Where Ralph lives (for reference)

- **Config:** `ralph.yml` (backend, prompt_file, max_iterations, etc.).
- **Runtime state:** `.ralph/` (e.g. state, lock, logs, agent data).
- **CLI:** `abi ralph init | run | super | multi | status | gate | improve | skills`; see `abi ralph help` and `tools/cli/commands/ralph/`.

### Self-improvement and learning

Learning and improvement happen in different ways depending on mode.

**Outside the loop (you, in Cursor):**

- **Improve by following and updating project knowledge:** Use CLAUDE.md and `.claude/rules/zig.md` as the source of truth. When you discover a new gotcha, a better pattern, or a doc fix, propose edits to those files so future sessions (and humans) benefit.
- **Validate and correct:** Run `zig build full-check`, fix failing tests, run `/baseline-sync` when test counts change. Learning here is “get the repo to a good state and capture it in docs and baselines.”
- **No persistent skill store for this session:** You do not write to `.ralph/` or Abbey memory. Session context is your only memory unless the user explicitly asks you to add a skill via Ralph (e.g. `abi ralph skills add "..."`).
- **When you learn something worth reusing:** Prefer updating CLAUDE.md or `.claude/rules/zig.md` (or a skill file under `.claude/skills/`) so it applies to all future Cursor sessions. Suggest “add this to CLAUDE.md” or “run baseline-sync” rather than only mentioning it in chat.

**Inside the loop (Ralph):**

- **Skills:** Ralph stores lessons in Abbey semantic memory (category `.skill`). They are injected into the system prompt for future `abi ralph run` / `abi ralph improve` so the agent can reuse them across runs.
- **How skills are added:**
  - **Manual:** `abi ralph skills add "lesson text"` or `abi ralph run --store-skill "lesson"`.
  - **Automatic:** `abi ralph run --auto-skill` or `abi ralph improve` (default) call the engine’s `extractAndStoreSkill(goal, result)` so the LLM summarizes a lesson from the run and it is stored.
- **Self-improvement pass:** `abi ralph improve` runs a loop with a task like “analyze source, identify issues, extract a lesson”; result can be auto-stored as a skill. Use when the user wants Ralph to reflect on the codebase and persist a lesson.
- **Inspect or clear:** `abi ralph skills` lists count and stats; `abi ralph skills clear` wipes Abbey memory (all skills). Implementation: `src/features/ai/abbey/engine.zig` (`storeSkill`, `extractAndStoreSkill`), `memory/mod.zig` (`getSkillsContext`).

**Summary:** Outside the loop, you improve by editing code and docs and running checks; inside the loop, Ralph improves by storing and reusing skills in Abbey memory. Use the right channel for the current mode.

### Super Ralph (power use)

When the user wants **autonomous multi-step execution** with skill memory and optional quality gates, use or suggest the Ralph CLI.

- **One-shot power command:** `abi ralph super --task "goal"` — inits workspace if missing, runs the loop, optional `--auto-skill` and `--gate`.
- **When to suggest:** User says "Ralph", "super ralph", "run the loop", "do everything", "full migration", or describes a long multi-step task that should run iteratively with verification.
- **When not to:** Single-file edits, bounded refactors in one session — do those yourself; suggest Ralph only if they explicitly want the loop.
- **Quality gate:** After Ralph produces report JSON, `abi ralph gate` (or `zig build ralph-gate`) enforces the score threshold; `zig build verify-all` includes ralph-gate.
- **Multi-Ralph (Zig, fast):** Lock-free RalphBus (`ralph_multi`) + parallel swarm (`ralph_swarm`) — `abi ralph multi -t "g1" -t "g2"` runs N agents on the runtime ThreadPool; or from Zig: `ThreadPool.schedule(abi.ai.abbey.ralph_swarm.parallelRalphWorker, .{ &ctx, index })`.
- **Skill:** `.claude/skills/super-ralph/SKILL.md`.

## Quick Reference

| Key | Value |
|-----|-------|
| **Zig** | `0.16.0-dev.2623+27eec9bd6` or newer (pinned in `.zigversion`) |
| **Entry Point** | `src/abi.zig` |
| **Version** | 0.4.0 |
| **Test baseline** | 1270 pass, 5 skip (1275 total) — must be maintained |
| **Feature tests** | 1535 pass (1535 total) — `zig build feature-tests` |
| **CLI commands** | 30 commands + 8 aliases |
| **Feature modules** | 21 (comptime-gated; see Feature Flags) |

## Build & Test Commands

Ensure your system `zig` matches `.zigversion`.

```bash
# One-time/periodic toolchain sync
zvm upgrade
PINNED_ZIG="$(cat .zigversion)"
zvm install "$PINNED_ZIG"
zvm use "$PINNED_ZIG"

# Validate active binary and pinned version
which zig
zig version
cat .zigversion
bash scripts/toolchain_doctor.sh

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
zig build toolchain-doctor                  # Diagnose local Zig PATH/version drift vs .zigversion
zig build validate-flags                     # Compile-check 34 feature flag combos
zig build cli-tests                          # CLI smoke tests (top-level + nested, e.g. help llm, bench micro hash)
zig build lint                               # CI formatting check
zig build fix                                # Format source files in place
zig build examples                           # Build all examples
zig build check-wasm                         # Check WASM compilation
zig build verify-all                         # full-check + consistency + examples + check-wasm
zig build ralph-gate                         # Require live Ralph report and threshold pass
scripts/check_zig_version_consistency.sh     # Verify .zigversion matches build.zig/docs
bash scripts/toolchain_doctor.sh             # Diagnose PATH precedence and active zig mismatch
zig std                                     # Print stdlib source path (useful for reading std lib internals)
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

### Feature Flags

The codebase has **21 comptime-gated feature modules**. Each is switched by a build flag; when disabled, the stub is linked and the feature returns `error.FeatureDisabled`. The canonical list lives in `src/core/feature_catalog.zig`; build options in `build/options.zig`.

**Usage:** `zig build -Denable-ai=true -Denable-gpu=false -Dgpu-backend=vulkan,cuda`

All features default to `true` except `-Denable-mobile`. GPU backends: `auto`, `none`,
`cuda`, `vulkan`, `metal`, `stdgpu`, `webgpu`, `tpu`, `webgl2`, `opengl`, `opengles`, `fpga`, `simulated`.
Neural networks can use GPU (CUDA/Metal/Vulkan), WebGPU, TPU (when runtime linked), or multi-threaded CPU via `abi.runtime.ThreadPool` / `InferenceConfig.num_threads`.
The `simulated` backend is always enabled as a software fallback for testing without GPU hardware.

**By area:**

| Feature | Build Flag | Notes |
|---------|------------|-------|
| *AI* | | |
| `ai` | `-Denable-ai` | Also: `-Denable-llm`, `-Denable-vision`, `-Denable-explore` |
| `ai_core` | `-Denable-ai` | Agents, tools, prompts, personas, memory |
| `inference` | `-Denable-llm` | LLM, embeddings, vision, streaming, transformer |
| `training` | `-Denable-training` | Training pipelines, federated learning |
| `reasoning` | `-Denable-reasoning` | Abbey, RAG, eval, templates, orchestration |
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

#### Features and the CLI

The CLI exposes feature status and lets you inspect or override (runtime) which features are enabled.

| CLI | Purpose |
|-----|---------|
| `abi system-info` (aliases: `info`, `sysinfo`) | Full report: platform, SIMD, GPU, network, AI connectors, and **Feature Matrix** (each `abi.Feature` with enabled yes/no). Use this to see what is compiled in and currently enabled. |
| `abi --list-features` | List all features with COMPILED / DISABLED (comptime). Exits after printing; no other command runs. |
| `abi status` | Framework health and feature count (e.g. “N/M features active”). |
| `abi config show` | Shows config including selected `enable_*` flags. |

**Runtime overrides (optional):** You can pass `--enable-<feature>` or `--disable-<feature>` before a command (e.g. `abi --disable-gpu bench all`). Only features that are **compiled in** can be enabled; disabled-at-build-time features cannot be turned on at runtime. Rebuild with `-Denable-<name>=true` to compile in a feature.

**Commands that require a feature:** These exit with a clear error if the feature is disabled (compile or runtime): `db` → database; `embed`, `convert`, `explore`, `multi-agent` (info/run) → ai; `discord` → web; `train` (llm/vision) → llm/vision; `bench` (database/training suites) → database/training. Use `abi system-info` or `abi --list-features` to confirm before running them.

## Critical Gotchas

**Top 6 (cause 80% of failures):**

1. `std.fs.cwd()` → `std.Io.Dir.cwd()` (requires I/O backend init)
2. Editing `mod.zig` without updating `stub.zig` → always keep signatures in sync
3. `defer allocator.free(x)` then `return x` → use `errdefer` (use-after-free)
4. `@tagName(x)` / `@errorName(e)` in format → use `{t}` specifier
5. `std.io.fixedBufferStream()` → removed; use `std.Io.Writer.fixed(&buf)`
6. `@field(build_options, field_name)` requires comptime context — use `inline for` not runtime `for`

**Full reference:** See [`.claude/rules/zig.md`](.claude/rules/zig.md) (auto-loaded for all `.zig` files) for the complete gotchas table with 19 entries covering I/O, `@typeInfo`, allocators, test discovery, and more. These patterns are also duplicated in CLAUDE.md above — when updating, keep both in sync.

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
| Shared radix tree | `src/services/shared/utils/radix_tree.zig` | Used by gateway + pages for URL routing |
| Circuit breaker | `src/services/shared/resilience/circuit_breaker.zig` | `abi.shared.resilience.{Simple,Atomic,Mutex}CircuitBreaker` |

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

### vNext Migration (Staged)

`src/vnext/` provides a forward API surface wrapping `Framework`:
- `abi.vnext.App` — thin wrapper around Framework (init, deinit, getFramework)
- `abi.vnext.AppConfig` — config with `strict_capability_check` + `required_capabilities`
- `abi.vnext.Capability` — mirrors `Feature` enum with conversion functions

**Current state (v0.4.0):** Capability checking and basic `App` methods (`.feature()`,
`.has()`, `.state()`) are implemented in `src/vnext/app.zig`. Use `app.getFramework()`
to access the full Framework during transition. Compatibility tests: `zig build vnext-compat`.

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
- 9 LLM providers: `openai`, `anthropic`, `ollama`, `huggingface`, `mistral`, `cohere`, `lm_studio`, `vllm`, `mlx`
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

**Main tests**: 1270 pass, 5 skip (1275 total) — `zig build test --summary all`
**Feature tests**: 1535 pass (1535 total) — `zig build feature-tests --summary all`
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

## After Making Changes

| Changed... | Run |
|------------|-----|
| Any `.zig` file | `zig fmt .` |
| Feature `mod.zig` | Also update `stub.zig`, then `zig build -Denable-<feature>=false` |
| Feature inline tests | `zig build feature-tests --summary all` (must stay at 1535+) |
| Build flags / options | `zig build validate-flags` |
| Public API | `zig build test --summary all` + update examples |
| Anything (full gate) | `zig build full-check` |
| Everything (release gate) | `zig build verify-all` (full-check + consistency + examples + check-wasm) |
| Build artifacts in `exe/` | Add `exe/` to `.gitignore` (see .gitignore) — standard output is `zig-out/` |

## Skills, Plans, and Agents (full index)

Use this section to find rules, skills, execution plans, and agent definitions. You are working **outside the Ralph loop** (Cursor/Claude direct-assist); Ralph is a separate iterative agent invoked via `abi ralph run`.

### When to use what

| Context | Use |
|--------|-----|
| **This session (Cursor/Claude)** | CLAUDE.md, `.claude/rules/zig.md`. Edit, build, test; suggest `/baseline-sync` or `/zig-migrate` when relevant. |
| **Ralph iterative loop** | User runs `abi ralph run` or `abi ralph improve`; skills live in Abbey memory (`.ralph/`). You do not drive that loop unless asked. |
| **Codex / external runner** | Same baselines; `scripts/project_baseline.env` is source of truth. |

### Rules (auto-loaded)

| Path | Purpose |
|------|---------|
| `.claude/rules/zig.md` | Zig 0.16 gotchas (19 entries), test baselines, I/O backend, stub conventions, import rules. Auto-loaded for all `.zig` files. Keep in sync with CLAUDE.md Critical Gotchas. |

### Custom skills (invoke by name)

| Skill | Invocation | Purpose |
|-------|------------|---------|
| **baseline-sync** | `/baseline-sync` | Sync test baseline numbers from `scripts/project_baseline.env` to doc files. Run after test count changes. See `.claude/skills/baseline-sync/SKILL.md`. |
| **zig-migrate** | `/zig-migrate [file-or-dir]` | Apply Zig 0.16 migration patterns (DynLib, I/O backend, format specifiers, etc.). See `.claude/skills/zig-migrate/SKILL.md`. |
| **super-ralph** | `/super-ralph` or suggest | Run or suggest Ralph: `abi ralph super --task "..."` (init-if-needed, run, optional `--gate`/`--auto-skill`); multi-Ralph via `abi.ai.abbey.ralph_multi`. See `.claude/skills/super-ralph/SKILL.md` and [Super Ralph (power use)](#super-ralph-power-use). |

Skill index: `.claude/skills/README.md` (if present) or list `ls .claude/skills/*/SKILL.md`.

### CI quality gate scripts (reference)

See table in [CI Quality Gate Scripts](#ci-quality-gate-scripts) below. Key: `scripts/check_test_baseline_consistency.sh`, `scripts/project_baseline.env`, `zig build full-check`, `zig build validate-flags`.

### Ready for use — checklist

| Check | Action |
|-------|--------|
| **Entry** | Use this file (CLAUDE.md) as single entry; open [Skills, Plans, and Agents](#skills-plans-and-agents-full-index) for rules, skills, plans, agents. |
| **Baseline** | Source of truth: `scripts/project_baseline.env`. After test count changes run `/baseline-sync`. |
| **Gate** | Before claiming done: `zig build full-check`. |
| **Ralph** | You are outside the loop unless the user runs `abi ralph run`; do not drive the loop from this session. |

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
| `scripts/check_test_baseline_consistency.sh` | Verify baseline numbers match across all 10+ doc files |
| `scripts/check_zig_version_consistency.sh` | Verify `.zigversion` matches `build.zig` and docs |
| `scripts/check_zig_016_patterns.sh` | Scan for deprecated Zig pre-0.16 patterns |
| `scripts/check_feature_catalog_consistency.sh` | Verify feature catalog matches build options |
| `scripts/check_import_rules.sh` | Enforce import conventions (no circular `@import("abi")` in features) |
| `scripts/toolchain_doctor.sh` | Diagnose PATH precedence and active zig mismatch |

### Updating Test Baselines

When test counts change, update `scripts/project_baseline.env` (source of truth), then run
`/baseline-sync` or manually update files listed in `.claude/skills/baseline-sync/SKILL.md`.
Verify with `bash scripts/check_test_baseline_consistency.sh`.

## References

- `CONTRIBUTING.md` — Development workflow and PR checklist
- `.claude/rules/zig.md` — Zig 0.16 complete gotchas table (auto-loaded for `.zig` files)
- `.claude/skills/` — Custom skills (baseline-sync, zig-migrate); see [Skills, Plans, and Agents](#skills-plans-and-agents-full-index)
- `scripts/project_baseline.env` — Canonical test baseline (source of truth for CI checks)
