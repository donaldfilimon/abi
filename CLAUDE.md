# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
# Build and test
zig build                              # Build the project
zig build test --summary all           # Run tests with detailed output
zig fmt .                              # Format code (run after edits)
zig build run -- --help                # CLI help

# Feature-gated builds
zig build -Denable-ai=true -Denable-gpu=false -Denable-database=true

# Single file testing (use zig test, NOT zig build test)
zig test src/compute/runtime/engine.zig
zig test src/tests/mod.zig --test-filter "pattern"

# GPU and system info
zig build run -- gpu backends
zig build run -- gpu devices
zig build run -- tui                   # Interactive launcher
```

## Critical Gotchas

| Issue | Solution |
|-------|----------|
| `--test-filter` syntax | Use `zig test file.zig --test-filter "pattern"`, NOT `zig build test --test-filter` |
| File system operations | Use `std.Io.Dir.cwd()` instead of deprecated `std.fs.cwd()` (Zig 0.16) |
| Reserved keywords | Escape with `@"error"` syntax, not bare `error` |
| Feature disabled errors | Rebuild with `-Denable-<feature>=true` |
| GPU backend conflicts | Enable only one backend at a time |
| WASM limitations | `database`, `network`, `gpu` features auto-disabled for WASM targets |

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | Full ABI agent system (LLM, vision, agent, training sub-features) |
| `-Denable-gpu` | true | GPU acceleration framework |
| `-Denable-database` | true | Vector database integration (WDBX) |
| `-Denable-network` | true | Distributed compute capabilities |
| `-Denable-web` | true | Web utilities and HTTP support |
| `-Denable-profiling` | true | Performance profiling and metrics |

**GPU Backends:** `-Dgpu-vulkan` (default), `-Dgpu-cuda`, `-Dgpu-metal`, `-Dgpu-webgpu`, `-Dgpu-opengl`, `-Dgpu-stdgpu` (CPU fallback)

## Architecture

```
src/
├── abi.zig              # Public API entry point: init(), shutdown(), version()
├── core/                # I/O, diagnostics, collections
├── compute/             # Runtime, concurrency, gpu, memory, profiling
├── features/            # ai/, vision/, agent/, training/, database/, network/, connectors/
├── framework/           # Lifecycle management and feature orchestration
└── shared/              # logging/, observability/, security/, utils/, platform/

tools/cli/               # CLI implementation (commands/, tui/)
```

**Module Convention:** Each feature uses `mod.zig` (entry point), `stub.zig` (feature-gated placeholder)

### Feature Gating Pattern

Compile-time selection via `build_options.enable_*`. Disabled features use stub modules:

```zig
const impl = if (build_options.enable_feature) @import("real.zig") else @import("stub.zig");
```

**Stub Requirements:** Must mirror complete API (structs, functions, constants); always return `error.<Feature>Disabled`.

### GPU Architecture

```
Unified API (Gpu.vectorAdd, etc.)
       ↓
KernelDispatcher (dispatcher.zig) - Includes CPU fallback
       ↓
Builtin Kernels (builtin_kernels.zig) - DSL-generated IR
       ↓
Backend Factory (backend_factory.zig) - Dynamic instantiation
       ↓
Backend VTables (backends/cuda/vtable.zig, etc.)
```

### Concurrency Primitives (`src/compute/`)

- `WorkStealingQueue` - Owner LIFO, thieves FIFO
- `LockFreeQueue/Stack` - Atomic CAS-based collections
- `PriorityQueue` - Lock-free task scheduling
- `ShardedMap` - Partitioned data structure reducing contention

### Runtime (`src/compute/runtime/`)

- `Future` - Async results with `.then()`, `.catch()`, `.finally()`
- `CancellationToken` - Cooperative cancellation
- `TaskGroup` - Hierarchical task organization

## Zig 0.16 Patterns

### File I/O

```zig
var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
defer io_backend.deinit();
const io = io_backend.io();

const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024));
defer allocator.free(content);
```

### Timing

```zig
var timer = std.time.Timer.start() catch return error.TimerFailed;
const elapsed_ns = timer.read();

// For sleeping, use the time utilities:
const time_utils = @import("src/shared/utils/time.zig");
time_utils.sleepMs(100);
```

### Memory Management

```zig
// Unmanaged containers for explicit control
var list = std.ArrayListUnmanaged(u8).empty;
try list.append(allocator, item);
list.deinit(allocator);
```

## Code Style

- **Indentation:** 4 spaces, no tabs
- **Line Length:** 100 characters max
- **Types:** PascalCase (`Engine`, `TaskConfig`)
- **Functions/Variables:** camelCase (`createEngine`, `taskId`)
- **Constants:** SCREAMING_SNAKE_CASE (`MAX_TASKS`)
- **Struct fields:** `allocator` first, then config/state, collections, resources, flags

## Environment Variables

Connector config prioritizes ABI-prefixed variables with fallback: `ABI_OPENAI_API_KEY` → `OPENAI_API_KEY`

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_OPENAI_API_KEY` / `OPENAI_API_KEY` | - | OpenAI API authentication |
| `ABI_OLLAMA_HOST` / `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server |
| `ABI_OLLAMA_MODEL` | `llama3.2` | Default Ollama model |
| `ABI_HF_API_TOKEN` / `HF_API_TOKEN` | - | HuggingFace API access |
| `DISCORD_BOT_TOKEN` | - | Discord integration token |

## Key Principles

- Preserve API stability: minimal changes consistent with existing patterns
- Maintain feature gating integrity: stub modules must mirror real APIs
- Resource hygiene: comprehensive `defer`/`errdefer` cleanup
- Use specific error sets - never `anyerror`
- Run `zig fmt .` after edits
- Run `zig build test --summary all` when behavior changes

## Reference

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview and quick start |
| [API_REFERENCE.md](API_REFERENCE.md) | Complete public API reference |
| [TODO.md](TODO.md) | Development status and remaining work |
| [docs/gpu.md](docs/gpu.md) | GPU programming guide |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Problem resolution |
