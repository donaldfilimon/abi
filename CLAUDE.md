# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Table of Contents

- [Project Overview](#project-overview)
- [LLM Instructions](#llm-instructions-shared)
- [Build Commands](#build-commands)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Zig 0.16 Conventions](#zig-016-conventions)
- [Environment Variables](#environment-variables)
- [Key API Notes](#key-api-notes)
- [Network Features](#network-features)
- [CLI Commands](#cli-commands)
- [Running Examples](#running-examples)
- [Testing Utilities](#testing-utilities)

## Project Overview

ABI is a modern Zig 0.16.x framework for modular AI services, vector search, and high-performance systems tooling. It provides a layered architecture with feature-gated compilation.

## LLM Instructions (Shared)

- Keep changes minimal and consistent with existing patterns; avoid breaking public APIs unless requested.
- Preserve feature gating: stub modules must mirror the real API and return `error.*Disabled`.
- Use Zig 0.16 conventions (`std.Io`, `std.ArrayListUnmanaged`, `{t}` formatting, explicit allocators).
- Always clean up resources with `defer`/`errdefer`; use specific error sets (no `anyerror`).
- Run `zig fmt .` after code edits and `zig build test --summary all` when behavior changes.
- Update docs/examples when APIs or behavior change so references stay in sync.

## Build Commands

```bash
zig build                    # Build the project
zig build test               # Run all tests
zig build test --summary all # Run tests with detailed output
zig build run -- --help      # Run CLI with help
zig build run -- --version   # Show version info
zig fmt .                    # Format all code
zig fmt --check .            # Check formatting without changes
zig build -Doptimize=ReleaseFast  # Optimized release build
```

### Running Tests for a Specific Module

```bash
zig test src/compute/runtime/engine.zig     # Test specific file
zig build test --summary all                # Show all test results
zig build test -Denable-gpu=true -Denable-network=true  # Test with features
```

**Note**: `--test-filter` is passed to the test runner, not to `zig build test`. To filter tests, use:
```bash
zig test src/tests/mod.zig --test-filter "pattern"
```

### CLI Entrypoint

CLI resolution prefers `tools/cli/main.zig` and falls back to `src/main.zig` if not present.

### Feature Flags

Build with specific features enabled/disabled:

```bash
zig build -Denable-ai=true -Denable-gpu=false -Denable-web=true -Denable-database=true
```

Core flags (defaults in parentheses):
- `-Denable-ai` (true) - AI features and connectors
- `-Denable-explore` (true if AI enabled) - AI code exploration tools
- `-Denable-llm` (true if AI enabled) - Local LLM inference
- `-Denable-gpu` (true) - GPU acceleration
- `-Denable-web` (true) - Web utilities and HTTP
- `-Denable-database` (true) - Vector database and storage
- `-Denable-network` (true) - Distributed network compute
- `-Denable-profiling` (true) - Profiling and metrics

**Note**: See `build.zig` `Defaults` struct for current default values. Some features may default to `false` in production builds.

GPU backends (Vulkan enabled by default when GPU is enabled):
`-Dgpu-cuda`, `-Dgpu-vulkan`, `-Dgpu-metal`, `-Dgpu-webgpu`, `-Dgpu-opengl`, `-Dgpu-opengles`, `-Dgpu-webgl2`, `-Dgpu-stdgpu`

**Note**: `-Dgpu-stdgpu` enables a software CPU fallback backend (not related to Zig's `std.gpu`), useful for testing GPU code paths without hardware.

### Additional Build Targets

```bash
zig build examples           # Build all example programs
zig build wasm               # Build WASM bindings (outputs to zig-out/wasm/)
zig build check-wasm         # Check WASM compilation without installing
zig build profile            # Build with profiling enabled (ReleaseFast)
zig build benchmarks         # Run comprehensive benchmarks
zig build benchmark-legacy   # Run legacy compute benchmarks
```

**Benchmark Organization**:
- `benchmarks/framework.zig` - Advanced benchmark harness with statistical analysis (warm-up, outlier detection, percentiles)
- `benchmarks/main.zig` - Entry point for running all benchmarks
- `benchmarks/<domain>.zig` - Domain-specific benchmark suites (simd, memory, concurrency, database, network, crypto, ai)
- Each suite registers benchmarks with the framework and reports ops/sec, mean/median/p99, memory tracking
- Framework features: auto-calibration, coefficient of variation targeting, throughput calculations

## Project Structure

```
abi/
├── src/
│   ├── abi.zig              # Public API entry point
│   ├── root.zig             # Root module
│   ├── core/                # Core infrastructure (I/O, diagnostics, collections)
│   ├── compute/             # Compute engine: concurrency/, gpu/, memory/, network/, profiling/, runtime/
│   ├── features/            # Feature modules: ai/, database/, gpu/, monitoring/, network/, connectors/
│   ├── framework/           # Lifecycle and orchestration
│   ├── shared/              # Cross-cutting: logging/, observability/, security/, utils/
│   └── tests/               # Test utilities (proptest)
├── tools/cli/               # CLI implementation (preferred over src/main.zig)
├── benchmarks/              # Modular benchmark suites with statistical framework
├── examples/                # Example programs
├── bindings/wasm/           # WASM bindings
└── docs/                    # Documentation
```

**Key directories to understand**:
- `src/compute/gpu/backends/` - CUDA, Vulkan, Metal, WebGPU, OpenGL, stdgpu, simulated backends
- `src/compute/gpu/unified.zig` - Unified GPU API with high-level operations
- `src/compute/gpu/dsl/` - Portable kernel DSL and cross-backend compiler
- `src/features/ai/llm/` - Local LLM inference: GGUF loading, tokenization, transformers, KV cache
- `src/features/ai/explore/` - Code exploration: AST parsing, callgraph, dependency analysis
- `src/features/database/` - WDBX vector database: HNSW, hybrid search, batch operations
- `src/features/monitoring/alerting.zig` - Alerting rules system with configurable thresholds

**Module File Organization Convention**:
- `mod.zig` - Re-exports and facade (module entry point)
- `stub.zig` - Parallel stub when feature-gated (must mirror full API)
- Submodules use underscores: `error_handling.zig`, `kernel_cache.zig`
- Complex features use subdirectories: `llm/`, `embeddings/`, `backends/`
- Tests: inline via `test` blocks or separate `*_test.zig` files

## Architecture

### Layered Structure

1. **Public API** (`src/abi.zig`) - Main entry point with curated re-exports. Use `abi.init()`, `abi.shutdown()`, `abi.version()`.

2. **Framework Layer** (`src/framework/`) - Lifecycle management, feature orchestration, runtime configuration.

3. **Compute Engine** (`src/compute/`) - Work-stealing scheduler, lock-free data structures, memory arenas, GPU integration with CPU fallback.

4. **Feature Stacks** (`src/features/`) - Vertical feature modules (ai, gpu, database, web, monitoring, network, connectors).

5. **Shared Utilities** (`src/shared/`) - Platform abstractions, SIMD, crypto, JSON, filesystem helpers.

### Key Patterns

- **Feature Gating with Stub Modules**: Compile-time feature enabling via build options checked with `build_options.enable_*`. When a feature is disabled, stub modules provide compile-time compatible placeholders that return `error.*Disabled` (e.g., `error.AiDisabled`, `error.NetworkDisabled`).

  **Stub Module Requirements**:
  - Must mirror the full API surface of the real module (all structs, functions, constants, type definitions)
  - Functions return appropriate `error.*Disabled` errors
  - Struct types must be defined (can be opaque or minimal)
  - Constants should have appropriate placeholder values
  - Import pattern: `const impl = if (build_options.enable_feature) @import("real.zig") else @import("stub.zig");`

  **Examples**: `src/features/ai/stub.zig`, `src/compute/network/stub.zig`, `src/features/ai/llm/stub.zig`, `src/compute/profiling/stub.zig`, `src/features/ai/explore/stub.zig`

- **GPU Backend Selection**: Multiple backends with automatic fallback. Device scoring: discrete=1000, integrated=500, virtual=100, cpu=50, other=10. Backends: CUDA, Vulkan, Metal, WebGPU, OpenGL/ES, stdgpu (software fallback), simulated (testing).

  Key infrastructure files in `src/compute/gpu/`:
  - `memory_pool_advanced.zig` - Size-class allocation (64B-4MB), coalescing, pressure handling
  - `metrics.zig` - Performance tracking (kernels, transfers, utilization)
  - `recovery.zig` - Device failure recovery with fallback hierarchy
  - `error_handling.zig` - Structured error tracking with health monitoring

- **VTable Pattern**: Polymorphic workload execution for CPU and GPU variants (`WorkloadVTable`, `GPUWorkloadVTable`). Allows the same `WorkItem` to specify both CPU and GPU execution paths, selected at runtime based on availability and hints.

- **Allocator Ownership**: Explicit memory management throughout. Prefer `std.ArrayListUnmanaged` over `std.ArrayList` to make allocator passing explicit. All allocations must have corresponding cleanup in `deinit()` or via `defer`/`errdefer`.

- **Lifecycle Management**: Strict init/deinit patterns. Use `defer` for cleanup in success paths and `errdefer` for cleanup on error. Resources must be cleaned up in reverse order of initialization.

- **Module Lifecycle Pattern**: Feature modules use standardized lifecycle from `src/shared/utils/lifecycle.zig`:
  - `ModuleLifecycle` (thread-safe with mutex) for multi-threaded modules
  - `SimpleModuleLifecycle` (lock-free) for single-threaded modules
  - All feature modules implement: `init()`, `deinit()`, `isEnabled()`, `isInitialized()`
  - Feature lifecycle coordinated centrally via `src/features/mod.zig`

- **Memory Pooling**: GPU buffers use pooled allocation via `gpu.MemoryPool`. CPU-side memory uses arena allocation in compute contexts (`src/compute/memory/mod.zig`).

- **Error Context and Handling**: Use `ErrorContext` from `src/shared/utils/errors.zig` for structured error logging with operation context, category, and source location. Always use specific error sets instead of `anyerror`. Feature-disabled errors follow the pattern `error.<Feature>Disabled` (e.g., `error.AiDisabled`, `error.NetworkDisabled`, `error.ProfilingDisabled`).

  **Error utilities in `src/shared/utils/errors.zig`**:
  - `Result(T, E)` wrapper: `unwrap()`, `unwrapOr()`, `isSuccess()` methods
  - `ResourceManager`: Generic resource cleanup with `set()`, `take()`, `get()`
  - `ErrorPatterns`: `handleAllocError()`, `handleIoError()`, `validateInput()`
  - Error composition: Combine error sets with `||` operator

  **GPU Error Handling Pattern**:
  ```zig
  // Define specific error sets for each module
  pub const InitError = error{ InitializationFailed, DriverNotFound, DeviceNotFound };
  pub const AllocationError = error{ OutOfMemory, DeviceLost };

  // Use errdefer for cleanup on failure
  pub fn allocateDeviceMemory(allocator: std.mem.Allocator, size: usize) AllocationError!*anyopaque {
      const buffer = try allocator.create(Buffer);
      errdefer allocator.destroy(buffer);
      // ... rest of initialization
  }

  // Report errors to metrics/recovery systems
  if (error_occurred) {
      metrics_collector.recordError(.kernel_launch);
      _ = recovery_manager.reportError(.cuda, device_id, .device_lost);
  }
  ```

### Concurrency Primitives

The compute engine (`src/compute/`) provides several concurrency utilities:

- **WorkStealingQueue**: LIFO for owner (`pop`), FIFO for thieves (`steal`). Used by worker threads in the engine.
- **LockFreeQueue/LockFreeStack**: Atomic CAS-based collections for high-contention scenarios.
- **PriorityQueue**: Lock-free priority queue for task scheduling (`src/compute/concurrency/priority_queue.zig`)
- **ShardedMap**: Reduces lock contention by partitioning data across multiple shards (e.g., `RESULT_SHARD_COUNT=16` in engine).
- **Backoff**: Exponential backoff with spin-loop hints before thread yield. Used in busy-wait loops.

### Runtime Patterns

The compute runtime (`src/compute/runtime/`) provides advanced execution patterns:

- **Futures**: Asynchronous task results with `.then()`, `.catch()`, `.finally()` chaining (`future.zig`)
- **Cancellation**: Cooperative task cancellation via `CancellationToken` (`cancellation.zig`)
- **Task Groups**: Hierarchical task grouping with automatic cleanup on scope exit (`task_group.zig`)
- **Engine Types**: Separated type definitions for cleaner module boundaries (`engine_types.zig`)

### NUMA and CPU Affinity

The engine supports NUMA-aware scheduling (`src/compute/runtime/numa.zig`):
- `CpuTopology`: Discovers CPU topology and NUMA nodes
- `setThreadAffinity(cpu_id)`: Pins thread to a specific CPU
- `getCurrentCpuId()`: Gets current CPU for scheduling decisions
- Enable with `EngineConfig{ .numa_enabled = true, .cpu_affinity_enabled = true }`

### GPU Backend Development Patterns

All backends in `src/compute/gpu/backends/` must implement:

```zig
// Lifecycle: init() !void, deinit() void
// Kernel ops: compileKernel, launchKernel, destroyKernel (all take allocator)
// Memory ops: allocateDeviceMemory, freeDeviceMemory, memcpyHostToDevice, memcpyDeviceToHost
```

**Key patterns**:
- **Thread Safety**: Use mutex for initialization (see CUDA backend's `init_mutex`)
- **Symmetric Operations**: Every `allocate*` must have matching `free*` with same allocator
- **Recovery**: Register devices with `recovery.registerDevice()`, report failures with `recovery.reportError()`
- **Metrics**: Record kernels, transfers, allocations via `metrics.record*()` methods

### Unified GPU API

The `src/compute/gpu/unified.zig` provides a high-level API covering all 8 backends:

```zig
var gpu = try abi.Gpu.init(allocator, .{
    .enable_profiling = true,
    .memory_mode = .automatic,  // API handles transfers
});
defer gpu.deinit();

// High-level operations
const a = try gpu.createBufferFromSlice(f32, &data_a, .{});
const b = try gpu.createBufferFromSlice(f32, &data_b, .{});
const result = try gpu.createBuffer(size, .{});
defer { gpu.destroyBuffer(a); gpu.destroyBuffer(b); gpu.destroyBuffer(result); }

_ = try gpu.vectorAdd(a, b, result);
```

**Built-in operations**: `vectorAdd`, `matrixMultiply`, `reduceSum`, `dotProduct`, `softmax`

**Memory modes**:
- `automatic` - API handles all host/device transfers (recommended)
- `explicit` - User controls transfers via `buffer.toDevice()`/`toHost()`
- `unified` - Use unified memory where available

### Portable Kernel DSL

Write kernels once, compile to CUDA/GLSL/WGSL/MSL (`src/compute/gpu/dsl/`):

```zig
const dsl = abi.gpu.dsl;

var builder = dsl.KernelBuilder.init(allocator, "scale_vector");
defer builder.deinit();

_ = builder.setWorkgroupSize(256, 1, 1);
const input = try builder.addBuffer("input", 0, .{ .scalar = .f32 }, .read_only);
const output = try builder.addBuffer("output", 1, .{ .scalar = .f32 }, .write_only);
const scale = try builder.addUniform("scale", 2, .{ .scalar = .f32 });

// Build IR and compile to target backend
const ir = try builder.build();
const source = try dsl.compiler.compile(allocator, &ir, .cuda, .{});
```

**DSL structure**:
- `dsl/builder.zig` - Fluent API for kernel construction
- `dsl/kernel.zig` - Kernel IR representation
- `dsl/compiler.zig` - Compiles IR to backend-specific code
- `dsl/codegen/*.zig` - Backend code generators (cuda, glsl, wgsl, msl)

### Multi-GPU Support

```zig
var gpu = try abi.Gpu.init(allocator, .{
    .multi_gpu = true,
    .load_balance_strategy = .memory_aware,
});

// Distribute work across devices
const distributions = try gpu.distributeWork(total_elements);
defer allocator.free(distributions);

for (distributions) |dist| {
    // dist.device_id, dist.offset, dist.size
}
```

**Load balance strategies**: `round_robin`, `memory_aware`, `compute_aware`, `manual`

## Zig 0.16 Conventions

### I/O API

Zig 0.16 uses the unified `std.Io` API:

```zig
// Reader types
std.Io.Reader         // Generic reader interface
std.Io.File.Reader    // File reader (use .interface for delimiter methods)
std.Io.net.Stream.Reader  // Network stream reader

// HTTP Server - pass readers/writers directly, not via .interface
var reader = stream.reader(io, &recv_buffer);
var writer = stream.writer(io, &send_buffer);
var server: std.http.Server = .init(&reader, &writer);  // Direct reference
```

### Synchronous File I/O with std.Io.Threaded

For synchronous file operations (e.g., reading files outside async contexts), use `std.Io.Threaded`:

```zig
// Create I/O backend for synchronous file operations
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty,
});
defer io_backend.deinit();
const io = io_backend.io();

// Read file
const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024)) catch |err| {
    return err;
};
defer allocator.free(content);

// Write file
var file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch return error.Failed;
defer file.close(io);
var writer = file.writer(io);
try writer.writeAll(content);
```

**Note**: `std.fs.cwd()` does not exist in Zig 0.16. Use `std.Io.Dir.cwd()` (passing an `std.Io` context) instead.

### Windows-Specific File Operations

For memory-mapped files and direct file access on Windows, use kernel32 APIs:

```zig
const WindowsFile = struct {
    extern "kernel32" fn CreateFileA(...) callconv(.winapi) std.os.windows.HANDLE;
    extern "kernel32" fn GetFileSizeEx(...) callconv(.winapi) std.os.windows.BOOL;
    extern "kernel32" fn ReadFile(...) callconv(.winapi) std.os.windows.BOOL;
    extern "kernel32" fn CloseHandle(hObject: std.os.windows.HANDLE) callconv(.winapi) std.os.windows.BOOL;
};
```

See `src/features/ai/llm/io/mmap.zig` for a complete example.

### Timing and Measurement

Use `std.time.Timer` for high-precision timing (not `std.time.nanoTimestamp()`):

```zig
var timer = std.time.Timer.start() catch return error.TimerFailed;
// ... work ...
const elapsed_ns = timer.read();
```

### Sleep API

Use `std.Io`-based sleep instead of `std.time.sleep()`:

```zig
// Preferred - use the time utilities module
const time_utils = @import("src/shared/utils/time.zig");
time_utils.sleepMs(100);   // Sleep 100 milliseconds
time_utils.sleepSeconds(1); // Sleep 1 second
time_utils.sleepNs(50_000); // Sleep 50 microseconds

// Additional helpers available:
var watch = try time_utils.Stopwatch.start();
// ... work ...
const elapsed_ms = watch.elapsedMs();

// Direct Io usage (when you have an Io context)
const duration = std.Io.Clock.Duration{
    .clock = .awake,
    .raw = .fromNanoseconds(nanoseconds),
};
std.Io.Clock.Duration.sleep(duration, io) catch {};
```

### Aligned Memory Allocation

For aligned allocations, use `std.mem.Alignment`:

```zig
// Preferred - use Alignment enum
const page_size = 4096;
const data = try allocator.alignedAlloc(u8, comptime std.mem.Alignment.fromByteUnits(page_size), size);
defer allocator.free(data);
```

See `docs/migration/zig-0.16-migration.md` for full migration details.

### Memory Management

```zig
// Preferred - explicit allocator passing
var list = std.ArrayListUnmanaged(u8).empty;
try list.append(allocator, item);
list.deinit(allocator);

// Avoid - hidden allocator dependency
var list = std.ArrayList(u8).init(allocator);
```

### Format Specifiers

```zig
// Use modern format specifiers instead of manual conversions
std.debug.print("Status: {t}\n", .{status});     // {t} for enum/error values
std.debug.print("Size: {B}\n", .{size});         // {B} for byte sizes (SI)
std.debug.print("Size: {Bi}\n", .{size});        // {Bi} for byte sizes (binary)
std.debug.print("Duration: {D}\n", .{dur});      // {D} for durations
std.debug.print("Data: {b64}\n", .{data});       // {b64} for base64

// Avoid @tagName()
std.debug.print("Status: {s}\n", .{@tagName(status)});  // Don't do this
```

### Style

- 4 spaces, no tabs, lines under 100 chars, one blank line between functions
- **Types**: PascalCase (`Engine`, `TaskConfig`)
- **Functions/Variables**: snake_case (`create_engine`, `task_id`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_TASKS`, `DEFAULT_MAX_TASKS`)
- **Struct Fields**: `allocator` first, then config/state, collections, resources, flags
- Explicit imports only (no `usingnamespace` except for re-exports)
- Use specific error sets, not `anyerror`

## Environment Variables

**Connector Configuration Pattern** (`src/features/connectors/mod.zig`):
- Env vars prioritized: ABI-prefixed (`ABI_OPENAI_API_KEY`) checked before standard (`OPENAI_API_KEY`)
- Helpers: `getEnvOwned()` and `getFirstEnvOwned()` for flexible env lookup
- Auth: `buildBearerHeader()` for authentication headers
- Optional loading: `tryLoadOpenAI()` returns null if unavailable, `loadOpenAI()` returns error

Connector-specific:
- `ABI_OPENAI_API_KEY` / `OPENAI_API_KEY`
- `ABI_OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
- `ABI_OPENAI_MODE` (`responses`, `chat`, or `completions`)
- `ABI_HF_API_TOKEN` / `HF_API_TOKEN` / `HUGGING_FACE_HUB_TOKEN`
- `ABI_HF_BASE_URL` (default: `https://api-inference.huggingface.co`)
- `ABI_OLLAMA_HOST` / `OLLAMA_HOST` (default: `http://127.0.0.1:11434`)
- `ABI_OLLAMA_MODEL` (default: `llama3.2`)
- `DISCORD_BOT_TOKEN` - Discord bot authentication token (required for Discord features)
- `ABI_LOCAL_SCHEDULER_URL` / `LOCAL_SCHEDULER_URL` (default: `http://127.0.0.1:8081`)
- `ABI_LOCAL_SCHEDULER_ENDPOINT` (default: `/schedule`)

## Key API Notes

### Compute Engine Timeouts

When using `runWorkload(engine, workload, timeout_ms)`:
- `timeout_ms=0`: Immediately returns `EngineError.Timeout` if not ready
- `timeout_ms>0`: Waits specified milliseconds before timeout
- `timeout_ms=null`: Waits indefinitely

### WDBX Backup/Restore Security

Backup and restore operations are restricted to the `backups/` directory only. Filenames must not contain path traversal sequences (`..`), absolute paths, or Windows drive letters.

### WASM Build Constraints

When building for WASM (`zig build wasm`), these features are automatically disabled:
- `enable-database` - No `std.Io.Threaded` support
- `enable-network` - No socket support
- `enable-gpu` - Native GPU backends unavailable
- `enable-web` - WebGPU bindings simplified

WASM bindings are in `bindings/wasm/abi_wasm.zig`.

## Network Features

When `enable-network=true`, `src/features/network/` provides distributed systems primitives:
- **Service Discovery**: Consul, Kubernetes, and manual backends
- **Circuit Breaker**: Automatic fault isolation with configurable thresholds
- **Rate Limiting**: Token bucket, sliding window, fixed window algorithms
- **Connection Pooling**: Per-host pools with configurable sizes
- **Raft Consensus**: Full implementation for cluster consensus
- **Task Scheduling**: Priority levels with load balancing strategies

### Raft Consensus Usage

The Raft implementation (`src/features/network/raft.zig`) provides leader election and log replication:

```zig
var node = try RaftNode.init(allocator, "node-1", .{
    .election_timeout_min_ms = 150,
    .election_timeout_max_ms = 300,
    .heartbeat_interval_ms = 50,
});
defer node.deinit();

try node.addPeer("node-2");
try node.addPeer("node-3");

// Process timeouts (call periodically)
try node.tick(elapsed_ms);

// Append commands (leader only)
if (node.isLeader()) {
    const index = try node.appendCommand(command_data);
}
```

**Note**: Current implementation is in-memory only; add persistence interface for production use.

### Alerting System

The alerting module (`src/features/monitoring/alerting.zig`) provides configurable alert rules:

```zig
var manager = try AlertManager.init(allocator, .{
    .evaluation_interval_ms = 15_000,
    .default_for_duration_ms = 60_000,
});
defer manager.deinit();

try manager.addRule(.{
    .name = "high_error_rate",
    .metric = "errors_total",
    .condition = .greater_than,
    .threshold = 100,
    .severity = .critical,
    .for_duration_ms = 30_000,
});

// Register notification handler
try manager.addHandler(.{
    .callback = myAlertHandler,
    .min_severity = .warning,
});

// Evaluate rules against metrics
try manager.evaluate(metrics);
```

**Alert states**: `inactive` → `pending` → `firing` → `resolved`
**Severities**: `info`, `warning`, `critical`

## CLI Commands

**CLI Framework Pattern** (`tools/cli/mod.zig`):
- Commands are dispatched via string matching in `tools/cli/commands/`
- Feature-aware: Checks `framework.isFeatureEnabled()` before running feature-specific commands
- Subcommand pattern: Commands accept additional arguments (e.g., `db add`, `gpu list`)

The CLI (`zig build run -- <command>`) provides these subcommands:

- `db <subcommand>` - Database operations (add, query, stats, optimize, backup, restore, serve)
- `agent [--message]` - Run AI agent (interactive or one-shot mode)
- `discord [command]` - Discord bot operations (status, guilds, send, commands, webhook)
- `llm <subcommand>` - Local LLM inference (info, generate, chat, bench, list)
- `config [command]` - Configuration management (init, show, validate)
- `explore [options] <query>` - Search and explore codebase
- `gpu [subcommand]` - GPU info (backends, devices, summary, default)
- `network [command]` - Network registry (list, register, status)
- `simd` - Run SIMD performance demo
- `system-info` - Show system and framework status

### Database CLI Examples

```bash
zig build run -- db stats                              # Show database statistics
zig build run -- db add --id 1 --embed "text to embed" # Add vector via embedding
zig build run -- db add --id 2 --vector "1.0,2.0,3.0"  # Add raw vector
zig build run -- db backup --path mybackup.db          # Backup to file
zig build run -- db restore --path mybackup.db         # Restore from file
```

### Explore CLI Examples

```bash
zig build run -- explore "pub fn" --level quick        # Quick search (file pattern matching)
zig build run -- explore "fn init" --level thorough    # Thorough search (AST parsing, dependencies)
```

**Explore Levels**:
- `quick` - Fast file-pattern search, grep-based matching
- `medium` - AST parsing for accurate symbol resolution
- `thorough` - Full dependency graph + call graph analysis

**Capabilities**:
- AST-based symbol extraction (functions, types, constants)
- Call graph construction (caller/callee relationships)
- Dependency graph analysis (module imports, relationships)
- Parallel exploration across worker threads
- Query intent understanding (search vs. definition vs. usage)

### Discord CLI Examples

```bash
zig build run -- discord status                        # Show bot status and info
zig build run -- discord guilds                        # List connected guilds
zig build run -- discord send --channel 123 "Hello"   # Send message to channel
zig build run -- discord commands --guild 456         # List application commands
zig build run -- discord webhook --id WID --token T   # Execute webhook
```

**Prerequisites**:
- Set `DISCORD_BOT_TOKEN` environment variable with your bot token

**Capabilities**:
- Bot status and user information retrieval
- Guild (server) listing and management
- Message sending to channels
- Webhook execution
- Gateway connection info

### LLM CLI Examples

```bash
zig build run -- llm info model.gguf                   # Show model information
zig build run -- llm generate model.gguf --prompt "Hello"  # Generate text
zig build run -- llm chat model.gguf                   # Interactive chat mode
zig build run -- llm bench model.gguf                  # Benchmark performance
zig build run -- llm list                              # List available models
```

The LLM feature (`src/features/ai/llm/`) provides local GGUF model inference with BPE tokenization, quantized tensors (Q4_0, Q4_1, Q8_0), transformer ops (matmul, attention, RoPE, RMSNorm), KV cache, and sampling strategies (greedy, top-k, top-p, temperature).

## Running Examples

Individual examples can be run with:
```bash
zig build run-hello          # Run hello example
zig build run-database       # Run database example
zig build run-agent          # Run agent example
zig build run-compute        # Run compute example
zig build run-gpu            # Run GPU example
zig build run-network        # Run network example
zig build run-discord        # Run Discord example
```

## Testing Utilities

### Property-Based Testing

The `src/tests/proptest.zig` module provides property-based testing:

```zig
const proptest = @import("src/tests/mod.zig").proptest;

// Use generators for random test data
const gen = proptest.Generators;
const intGen = gen.int(i32, -1000, 1000);

// Property test with forAll
try proptest.forAll(allocator, intGen, struct {
    fn check(x: i32) bool {
        return x * 2 / 2 == x; // Property to verify
    }
}.check);
```

Available generators: `int`, `float`, `bool`, `slice`, `string`, `optional`, `oneOf`.

### Hardware-Gated Tests

Use `error.SkipZigTest` to gate hardware-specific tests:

```zig
test "gpu-feature" {
    if (!build_options.enable_gpu) return error.SkipZigTest;
    // test body
}
```

### Integration Testing Pattern

Tests in `src/tests/integration.zig` follow this pattern:
- Create full framework with selected features via `abi.init()`
- Check `framework.isFeatureEnabled()` before running feature-specific tests
- Print warnings for unavailable features instead of failing
- Use proper `defer` cleanup of handles and allocations

### WDBX Database Namespace

When `enable-database=true`, the `abi.wdbx` namespace provides vector database operations:

```zig
const db = try abi.wdbx.createDatabase(allocator, "vectors.db", .{});
defer abi.wdbx.closeDatabase(db);

try abi.wdbx.insertVector(db, 1, &vector, metadata);
const results = try abi.wdbx.searchVectors(db, &query, 10);
```

Features: HNSW indexing, hybrid search (vector + full-text + metadata filtering), batch operations, sharding, HTTP API. See `src/features/database/` for implementation files.

## Commit Guidelines

Use `<type>: <summary>` format. Keep summaries ≤ 72 chars. Focus commits; update docs when APIs change.

## Architecture References

- System overview: `docs/intro.md`
- API surface: `API_REFERENCE.md`
- Migration guide: `docs/migration/zig-0.16-migration.md`

## Contacts

`src/shared/contacts.zig` provides a centralized list of maintainer contacts extracted from the repository markdown files.
