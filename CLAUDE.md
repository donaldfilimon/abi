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

# Additional build targets
zig build benchmarks                   # Run comprehensive benchmarks
zig build bench-competitive            # Run competitive benchmarks
zig build benchmark-legacy             # Run legacy performance benchmarks
zig build gendocs                      # Generate API documentation
zig build profile                      # Build with performance profiling
zig build wasm                         # Build WASM bindings
zig build check-wasm                   # Check WASM compilation
zig build examples                     # Build all examples

# Run example programs
zig build run-hello                    # Run hello example
zig build run-database                 # Run database example
zig build run-agent                    # Run agent example
zig build run-compute                  # Run compute example
zig build run-gpu                      # Run GPU example
zig build run-network                  # Run network example
zig build run-discord                  # Run discord example
```

## Critical Gotchas

| Issue | Solution |
|-------|----------|
| `--test-filter` syntax | Use `zig test file.zig --test-filter "pattern"`, NOT `zig build test --test-filter` |
| File system operations | Use `std.Io.Dir.cwd()` instead of deprecated `std.fs.cwd()` (Zig 0.16) |
| Reserved keywords | Escape with `@"error"` syntax, not bare `error` |
| Feature disabled errors | Rebuild with `-Denable-<feature>=true` |
| GPU backend conflicts | Enable only one GPU backend at a time |
| WASM limitations | `database`, `network`, `gpu` features auto-disabled for WASM targets |
| libc linking | CLI and examples require libc for environment variable access |

## Feature Flags

> **Complete Reference:** See [docs/feature-flags.md](docs/feature-flags.md) for comprehensive documentation including dependencies and WASM limitations.

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | Full ABI agent system (LLM, vision, agent, training sub-features) |
| `-Denable-gpu` | true | GPU acceleration framework |
| `-Denable-database` | true | Vector database integration (WDBX) |
| `-Denable-network` | true | Distributed compute capabilities |
| `-Denable-web` | true | Web utilities and HTTP support |
| `-Denable-profiling` | true | Performance profiling and metrics |
| `-Denable-explore` | true | Codebase exploration (requires `-Denable-ai`) |
| `-Denable-llm` | true | Local LLM inference (requires `-Denable-ai`) |

**GPU Backends:** `-Dgpu-vulkan` (default), `-Dgpu-cuda`, `-Dgpu-metal`, `-Dgpu-webgpu`, `-Dgpu-opengl`, `-Dgpu-opengles`, `-Dgpu-webgl2`, `-Dgpu-stdgpu` (CPU fallback)

**Flag Dependencies:**
```
enable-ai ──┬── enable-explore
            └── enable-llm
enable-web ──┬── gpu-webgpu (auto-enabled)
             └── gpu-webgl2 (auto-enabled)
```

**Cache Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-Dcache-dir` | `.zig-cache` | Directory for build cache |
| `-Dglobal-cache-dir` | (none) | Directory for global build cache |

## WASM Target Limitations

When building for WebAssembly (`zig build wasm`), these features are auto-disabled:

| Feature | Status | Reason |
|---------|--------|--------|
| `enable-database` | Disabled | No `std.Io.Threaded` support |
| `enable-network` | Disabled | No socket operations |
| `enable-gpu` | Disabled | Native GPU unavailable |
| `enable-web` | Disabled | Simplifies initial pass |
| `enable-profiling` | Disabled | Platform limitations |
| All GPU backends | Disabled | Including WebGPU (for now) |

Use `zig build check-wasm` to verify WASM compilation without full build.

## Architecture

```
src/
├── abi.zig              # Public API entry point: init(), shutdown(), version()
├── core/                # I/O, diagnostics, collections
├── compute/             # Runtime, concurrency, gpu, memory, profiling
│   └── gpu/             # Multi-backend GPU acceleration (73 files)
├── features/            # ai/, vision/, agent/, training/, database/, network/, connectors/
│   └── ai/              # LLM, agents, training (147 files)
├── framework/           # Lifecycle management and feature orchestration
└── shared/              # logging/, observability/, security/, utils/, platform/

tools/cli/               # CLI implementation (commands/, tui/)
bindings/wasm/           # WASM bindings entry point
benchmarks/              # Performance benchmarks
examples/                # Example programs
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
User Code (abi.Gpu.vectorAdd, etc.)
       ↓
Unified API (unified.zig) - Device/buffer management
       ↓
KernelDispatcher (dispatcher.zig) - Compilation, caching, CPU fallback
       ↓
Builtin Kernels (builtin_kernels.zig) - Pre-defined operations via DSL
       ↓
Kernel DSL (dsl/) - Portable IR with code generators
       ↓
Backend Factory (backend_factory.zig) - Runtime backend selection
       ↓
Backend VTables (interface.zig) - Polymorphic dispatch
       ↓
Native Backends (CUDA, Vulkan, Metal, WebGPU, OpenGL, OpenGL ES, WebGL2, stdgpu)
```

**GPU Backends Table:**

| Backend | Flag | Platform |
|---------|------|----------|
| Vulkan | `-Dgpu-vulkan` | Cross-platform (default) |
| CUDA | `-Dgpu-cuda` | NVIDIA |
| Metal | `-Dgpu-metal` | Apple |
| WebGPU | `-Dgpu-webgpu` | Web/Native |
| OpenGL | `-Dgpu-opengl` | Desktop (legacy) |
| OpenGL ES | `-Dgpu-opengles` | Mobile/Embedded |
| WebGL2 | `-Dgpu-webgl2` | Web browsers |
| stdgpu | `-Dgpu-stdgpu` | CPU fallback |

Key GPU files:
- `src/compute/gpu/unified.zig` - Unified GPU API
- `src/compute/gpu/dsl/` - Kernel DSL compiler (builder, codegen, optimizer)
- `src/compute/gpu/backends/` - Backend implementations with vtables
- `src/compute/gpu/tensor/` - Tensor operations

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

### Error Types

Always use specific error sets, never `anyerror`:

```zig
const MyError = error{
    InvalidInput,
    ResourceExhausted,
    FeatureDisabled,
};

fn myFunction() MyError!Result {
    // ...
}
```

### Format Specifiers

```zig
// Use {t} for errors and enums in Zig 0.16
std.debug.print("Error: {t}", .{err});
std.debug.print("State: {t}", .{state});
```

## Code Style

- **Indentation:** 4 spaces, no tabs
- **Line Length:** 100 characters max
- **Types:** PascalCase (`Engine`, `TaskConfig`)
- **Functions/Variables:** camelCase (`createEngine`, `taskId`)
- **Constants:** SCREAMING_SNAKE_CASE (`MAX_TASKS`)
- **Struct fields:** `allocator` first, then config/state, collections, resources, flags

## CLI Commands

| Command | Purpose |
|---------|---------|
| `db` | Database operations (stats, query) |
| `agent` | AI agent interaction |
| `llm` | LLM inference (info, generate, chat, bench, list, download) |
| `train` | Training pipeline (run, llm, resume, info) |
| `bench` | Performance benchmarks |
| `embed` | Embeddings generation |
| `gpu` | GPU management (backends, devices, status) |
| `network` | Network node operations |
| `config` | Configuration management |
| `explore` | Codebase search |
| `discord` | Discord bot integration |
| `simd` | SIMD operations demo |
| `system-info` | System information |
| `tui` | Interactive launcher |
| `version` | Version information |
| `help` | Help and usage |

## Example Programs

Available in `examples/` directory:

| Example | Command | Purpose |
|---------|---------|---------|
| hello | `zig build run-hello` | Basic framework usage |
| database | `zig build run-database` | Vector database operations |
| agent | `zig build run-agent` | AI agent demonstration |
| compute | `zig build run-compute` | Runtime and task execution |
| gpu | `zig build run-gpu` | GPU acceleration demo |
| network | `zig build run-network` | Distributed compute example |
| discord | `zig build run-discord` | Discord bot integration |

Build all examples: `zig build examples`

## Environment Variables

Connector config prioritizes ABI-prefixed variables with fallback: `ABI_OPENAI_API_KEY` -> `OPENAI_API_KEY`

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

## Local LLM Training

**Default Model:** GPT-2 Small (124M parameters) - open source, no authentication required.

```bash
zig build run -- train info                    # Show configuration and download instructions
zig build run -- train run --epochs 2          # Basic training test
zig build run -- llm list                      # List supported model formats
```

## Debugging

**Debug builds:** `zig build -Doptimize=Debug` (default) or `zig build -Doptimize=ReleaseSafe` for release with debug info.

### GDB/LLDB Quick Reference

| GDB | LLDB | Description |
|-----|------|-------------|
| `break file:line` | `b file:line` | Set breakpoint |
| `run` | `run` | Start program |
| `continue` | `continue` | Continue execution |
| `next` | `next` | Step over |
| `step` | `step` | Step into |
| `print var` | `print var` | Print variable |
| `backtrace` | `bt` | Show call stack |
| `info threads` | `thread list` | List threads |

### Memory Leak Detection

**GeneralPurposeAllocator:**
```zig
var gpa = std.heap.GeneralPurposeAllocator(.{ .stack_trace_frames = 10 }){};
defer {
    const check = gpa.deinit();
    if (check == .leak) @panic("Memory leak detected");
}
```

**TrackingAllocator for detailed analysis:**
```zig
const tracking = @import("src/shared/utils/memory/tracking.zig");
var tracker = tracking.TrackingAllocator.init(std.heap.page_allocator, .{});
defer {
    if (tracker.detectLeaks()) {
        tracker.dumpLeaks(std.io.getStdErr().writer()) catch {};
    }
    tracker.deinit();
}
```

### GPU Profiling

```zig
const gpu_profiling = @import("src/compute/gpu/profiling.zig");
var profiler = gpu_profiling.Profiler.init(allocator);
defer profiler.deinit(allocator);
profiler.enable();
```

### Performance Backoff Pattern

For performance-sensitive code, use exponential backoff:
```zig
const backoff = @import("src/shared/utils/backoff.zig");
var retry = backoff.ExponentialBackoff.init(.{
    .initial_delay_ms = 10,
    .max_delay_ms = 1000,
    .multiplier = 2,
});
while (retry.shouldRetry()) {
    // attempt operation
    retry.wait();
}
```

## Reference

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview and quick start |
| [API_REFERENCE.md](API_REFERENCE.md) | Public API reference |
| [docs/gpu.md](docs/gpu.md) | GPU programming guide |
| [docs/feature-flags.md](docs/feature-flags.md) | Complete feature flags reference |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Problem resolution |
| [AGENTS.md](AGENTS.md) | AI agent guidance (Claude, GPT, Gemini) |
