# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read `AGENTS.md` first for baseline rules (code style, post-edit checklist, Zig 0.16 API
migration table). This file adds deeper architectural context that requires reading multiple
files to understand.

| Key | Value |
|-----|-------|
| **Zig Required** | 0.16.x (`0.16.0-dev.2471+e9eadee00`+) — pinned in `.zigversion` |
| **Entry Point** | `src/abi.zig` |
| **Version** | 0.4.0 |

## Build Commands

```bash
zig build                                    # Build
zig build test --summary all                 # Full test suite
zig build run -- --help                      # CLI help
zig test src/path/to/file.zig --test-filter "pattern"  # Single test
zig fmt .                                    # Format (required after edits)
zig build full-check                         # Format + tests + CLI smoke tests
zig build lint                               # CI formatting check
zig build cli-tests                          # CLI smoke tests
zig build benchmarks                         # Performance benchmarks
zig build examples                           # Build all examples
```

Feature flags: `zig build -Denable-ai=true -Denable-gpu=false -Dgpu-backend=vulkan,cuda`

All features default to `true` except `-Denable-mobile`. GPU backends accept
comma-separated values: `auto`, `none`, `cuda`, `vulkan`, `metal`, `stdgpu`,
`webgpu`, `webgl2`, `opengl`, `opengles`, `fpga`.

## Critical Gotchas

These are the mistakes most likely to cause compilation failures:

| Mistake | Fix |
|---------|-----|
| `std.fs.cwd()` | `std.Io.Dir.cwd()` — Zig 0.16 moved filesystem to I/O backend |
| `std.time.Instant.now()` | `std.time.Timer.start()` |
| `list.init()` | `std.ArrayListUnmanaged(T).empty` |
| `@tagName(x)` in format | `{t}` format specifier for errors and enums |
| Editing `mod.zig` only | **Always update `stub.zig` too** — signatures must match |
| `std.fs.cwd().openFile(...)` | Must init `std.Io.Threaded` first and pass `io` handle |
| `std.time.sleep()` | `abi.shared.time.sleepMs()` / `sleepNs()` for cross-platform |

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
- **Both files must export identical public signatures** — if you add/change a function
  in `mod.zig`, the same signature must exist in `stub.zig` returning an error
- Test both paths: `zig build -Denable-<feature>=true` and `=false`
- Disabled features have zero binary overhead

### Module Hierarchy

```
src/abi.zig              → Public API, comptime feature selection, type aliases
src/core/                → Framework lifecycle, config builder, feature flags, registry
src/features/<name>/     → mod.zig + stub.zig per feature
src/services/            → Always-available infrastructure (runtime, platform, shared, ha, tasks)
tools/cli/               → CLI entry point and 24 commands
```

Import convention: public API uses `@import("abi")`, internal modules import
via their parent `mod.zig`. Never use direct file paths for cross-module imports.

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

## AI Module (Largest Module — 280+ files)

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
| Change build flags | `build.zig`, `src/core/flags.zig` |
| Add a new feature module | `src/features/<name>/mod.zig` + `stub.zig`, wire in `src/abi.zig` and `build.zig` |
| Add a CLI command | `tools/cli/commands/`, register in `tools/cli/main.zig` |
| Add config for a feature | `src/core/config/` |
| Write integration tests | `src/services/tests/` |
| Add a GPU backend | `src/features/gpu/backends/` |
| Security infrastructure | `src/services/shared/security/` (15 modules) |
| Language bindings | `bindings/` (C, Python, Go, JS, Rust) — build C lib first |
| Examples | `examples/` (18 examples) |

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

## Testing Patterns

- Unit tests: `*_test.zig` files alongside code
- Integration/stress/chaos/parity/property tests: `src/services/tests/`
- Skip hardware-gated tests with `error.SkipZigTest`
- Parity tests verify `mod.zig` and `stub.zig` export the same interface

## References

| Document | Purpose |
|----------|---------|
| `AGENTS.md` | Baseline rules, code style, Zig 0.16 migration table |
| `CONTRIBUTING.md` | Development workflow |
| `PLAN.md` | Development roadmap |
| `DEPLOYMENT_GUIDE.md` | Production deployment |
| `SECURITY.md` | Security practices |
