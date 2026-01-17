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
zig test src/runtime/engine/engine.zig
zig test src/tests/mod.zig --test-filter "pattern"

# Runtime feature flags (CLI)
zig build run -- --list-features          # List features and their status
zig build run -- --enable-gpu db stats    # Enable feature for this run
zig build run -- --disable-ai llm info    # Disable feature for this run

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
| Import paths | Always use `@import("abi")` for public API, not direct file paths |
| Stub API mismatch | When adding to real module, mirror the change in the corresponding `stub.zig` |

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

The codebase uses a flat domain structure with unified configuration and framework orchestration.

```
src/
├── abi.zig              # Public API entry point: init(), shutdown(), version()
├── config.zig           # Unified configuration system
├── framework.zig        # Framework orchestration with builder pattern
├── runtime/             # Always-on infrastructure (task execution, scheduling)
│   ├── engine/         # Task engine
│   ├── scheduling/     # Future, CancellationToken, TaskGroup
│   ├── concurrency/    # Lock-free primitives, priority queue
│   └── memory/         # Memory utilities
├── gpu/                 # GPU acceleration (primary location)
│   ├── mod.zig         # Module entry, exports unified API
│   ├── unified.zig     # Gpu struct, GpuConfig, high-level ops
│   ├── dsl/            # Kernel DSL compiler (builder, codegen, optimizer)
│   ├── backends/       # Backend implementations (cuda/, vulkan, metal, etc.)
│   ├── diagnostics.zig # GPU state debugging
│   ├── error_handling.zig # Structured error context
│   └── failover.zig    # Graceful degradation to CPU
├── ai/                  # AI module - Public API (llm, embeddings, agents, training)
├── database/           # Vector database (WDBX)
├── network/            # Distributed compute
├── observability/      # Metrics, tracing, profiling
├── web/                # Web/HTTP utilities
├── internal/           # Shared utilities (logging, plugins, platform, simd)
├── registry/           # Plugin registry system (comptime, runtime-toggle, dynamic)
├── features/           # Full implementations
│   └── ai/            # AI implementation (agent, training, embeddings, llm)
├── core/               # I/O, diagnostics, collections
└── shared/             # Legacy shared utilities

tools/cli/               # CLI implementation (commands/, tui/)
bindings/wasm/           # WASM bindings entry point
benchmarks/              # Performance benchmarks
examples/                # Example programs
```

**Import guidance:**
- **Public API**: Always use `@import("abi")` - this provides all public types and functions
- **Direct module access**: Use `abi.gpu`, `abi.ai`, `abi.database`, etc. through the framework
- **For GPU**: Use `src/gpu/` (fully migrated, primary location)
- **For Network**: Use `src/network/` (fully migrated, primary location)
- **For Database**: Use `src/database/` (fully migrated, primary location)
- **For Web**: Use `src/web/` (fully migrated, primary location)
- **For AI**: Use `src/ai/` for public API; implementation in `src/features/ai/`
- **Never import file paths directly** in application code - use the abi module

**Module Convention:** Each feature uses `mod.zig` (entry point), `stub.zig` (feature-gated placeholder)

### Framework Initialization Patterns

The framework supports multiple initialization styles:

```zig
const abi = @import("abi");

// 1. Default initialization (all compile-time enabled features)
var fw = try abi.init(allocator);
defer fw.deinit();

// 2. Struct literal configuration
var fw = try abi.init(allocator, .{
    .gpu = .{ .backend = .vulkan },
    .ai = .{ .llm = .{} },
});
defer fw.deinit();

// 3. Builder pattern (fluent API)
var fw = try abi.Framework.builder(allocator)
    .withGpu(.{ .backend = .cuda })
    .withAi(.{ .llm = .{ .model_path = "./models/llama.gguf" } })
    .withDatabase(.{ .path = "./data" })
    .build();
defer fw.deinit();

// 4. Access features via framework
if (fw.isEnabled(.gpu)) {
    const gpu_ctx = try fw.getGpu();
    // Use GPU context
}

const ai_ctx = try fw.getAi();
const runtime = fw.getRuntime(); // Always available
```

### Configuration System (`src/config.zig`)

The `Config` struct is the single source of truth for all framework settings:

```zig
pub const Config = struct {
    gpu: ?GpuConfig = null,           // GPU acceleration
    ai: ?AiConfig = null,             // AI with sub-features (llm, embeddings, agents, training)
    database: ?DatabaseConfig = null,  // Vector database
    network: ?NetworkConfig = null,    // Distributed compute
    observability: ?ObservabilityConfig = null,  // Metrics, tracing
    web: ?WebConfig = null,           // HTTP utilities
    plugins: PluginConfig = .{},
};

// Feature checking
if (config.isEnabled(.llm)) { ... }

// Get enabled features list
const features = try config.enabledFeatures(allocator);
```

### Feature Gating Pattern

Compile-time selection via `build_options.enable_*`. Disabled features use stub modules:

```zig
const impl = if (build_options.enable_feature) @import("real.zig") else @import("stub.zig");
```

**Stub Requirements:** Must mirror complete API (structs, functions, constants); always return `error.<Feature>Disabled`.

### GPU Architecture

All GPU code is in `src/gpu/` (fully migrated from `src/compute/gpu/`):

```
User Code (abi.Gpu.vectorAdd, etc.)
       ↓
Unified API (src/gpu/unified.zig) - Device/buffer management
       ↓
KernelDispatcher (src/gpu/dispatcher.zig) - Compilation, caching, CPU fallback
       ↓
Builtin Kernels (src/gpu/builtin_kernels.zig) - Pre-defined operations via DSL
       ↓
Kernel DSL (src/gpu/dsl/) - Portable IR with code generators
       ↓
Backend Factory (src/gpu/backend_factory.zig) - Runtime backend selection
       ↓
Backend VTables (src/gpu/interface.zig) - Polymorphic dispatch
       ↓
Native Backends (src/gpu/backends/) - CUDA, Vulkan, Metal, WebGPU, OpenGL, stdgpu
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
- `src/gpu/mod.zig` - Module entry point, exports all public types
- `src/gpu/unified.zig` - Unified GPU API (Gpu struct, GpuConfig)
- `src/gpu/dsl/` - Kernel DSL compiler (builder, codegen, optimizer)
- `src/gpu/backends/` - Backend implementations with vtables
- `src/gpu/diagnostics.zig` - GPU state debugging (DiagnosticsInfo)
- `src/gpu/error_handling.zig` - Structured error context (ErrorContext)
- `src/gpu/failover.zig` - Graceful degradation (FailoverManager)

### Runtime Infrastructure (`src/runtime/`)

The runtime module provides always-available infrastructure:

- **Context** - Runtime orchestration handle
- **Task execution** - Work scheduling and management
- Integrated with framework lifecycle

### Concurrency Primitives (`src/runtime/concurrency/`)

- `WorkStealingQueue` - Owner LIFO, thieves FIFO
- `LockFreeQueue/Stack` - Atomic CAS-based collections
- `PriorityQueue` - Lock-free task scheduling
- `ShardedMap` - Partitioned data structure reducing contention

### Runtime Primitives (`src/runtime/scheduling/`)

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

### Global Flags (Runtime Feature Control)

| Flag | Description |
|------|-------------|
| `--list-features` | List all features and their enabled/disabled status |
| `--enable-<feature>` | Enable a feature for this run (e.g., `--enable-gpu`) |
| `--disable-<feature>` | Disable a feature for this run (e.g., `--disable-ai`) |

**Examples:**
```bash
zig build run -- --list-features              # Show feature status
zig build run -- --enable-gpu db stats        # Run with GPU enabled
zig build run -- --disable-ai llm info        # Run with AI disabled
```

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
| `ABI_OLLAMA_MODEL` | `gpt-oss` | Default Ollama model |
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
const gpu_profiling = @import("src/gpu/profiling.zig");
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

## Diagnostics and Error Context (2026.01)

Structured diagnostics and error context APIs for debugging production issues.

### GPU Diagnostics

```zig
const gpu = @import("abi").gpu;  // Or: const gpu = @import("src/gpu/mod.zig");

// Collect comprehensive GPU state
const diag = gpu.DiagnosticsInfo.collect(allocator);

// Check health and format for logging
if (!diag.isHealthy()) {
    const msg = try diag.formatToString(allocator);
    defer allocator.free(msg);
    std.log.warn("{s}", .{msg});
}

// Structured error reporting
const ctx = gpu.ErrorContext.init(.backend_error, .cuda, "Kernel launch failed");
ctx.reportErrorFull(allocator);

// Graceful degradation
var manager = gpu.FailoverManager.init(allocator);
manager.setDegradationMode(.automatic);  // Auto-fallback to CPU
if (manager.isDegraded()) {
    std.log.info("Running in CPU fallback mode", .{});
}
```

### Database Diagnostics

```zig
const abi = @import("abi");
var fw = try abi.init(allocator);
if (fw.getDatabase()) |db| {
    const diag = db.diagnostics();
    std.log.info("Vectors: {d}, Memory: {d}KB, Healthy: {}", .{
        diag.vector_count,
        diag.memory.total_bytes / 1024,
        diag.isHealthy(),
    });
}
```

### AI Agent Error Context

```zig
const abi = @import("abi");
var fw = try abi.init(allocator);
if (try fw.getAi()) |ai| {
    // Use AI context
}
// Real implementation: src/features/ai/agent.zig
```

## Reference

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview and quick start |
| [API_REFERENCE.md](API_REFERENCE.md) | Public API reference |
| [docs/gpu.md](docs/gpu.md) | GPU programming guide |
| [docs/database.md](docs/database.md) | Vector database guide |
| [docs/ai.md](docs/ai.md) | AI and agents guide |
| [docs/feature-flags.md](docs/feature-flags.md) | Complete feature flags reference |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Problem resolution |
| [AGENTS.md](AGENTS.md) | AI agent guidance (Claude, GPT, Gemini) |
