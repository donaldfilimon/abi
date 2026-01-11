# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABI is a modern Zig 0.16.x framework for modular AI services, vector search, and high-performance systems tooling. It provides a layered architecture with feature-gated compilation.

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
`-Dgpu-cuda`, `-Dgpu-vulkan`, `-Dgpu-stdgpu`, `-Dgpu-metal`, `-Dgpu-webgpu`, `-Dgpu-opengl`, `-Dgpu-opengles`, `-Dgpu-webgl2`

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
│   ├── compute/             # Compute engine and concurrency
│   │   ├── concurrency/     # Lock-free queues, work-stealing, priority queues
│   │   ├── gpu/             # GPU integration layer
│   │   │   ├── backends/    # CUDA, Vulkan, Metal, WebGPU, OpenGL, stdgpu, simulated
│   │   │   ├── tensor/      # GPU tensor operations
│   │   │   ├── error_handling.zig   # Structured error tracking
│   │   │   ├── kernel_cache.zig     # Compiled kernel caching
│   │   │   ├── memory_pool_advanced.zig  # Size-class memory pooling
│   │   │   ├── metrics.zig          # Performance metrics collection
│   │   │   └── recovery.zig         # Device failure recovery
│   │   ├── memory/          # Arena allocators, pools
│   │   ├── network/         # Distributed compute (feature-gated)
│   │   ├── profiling/       # Metrics collection (feature-gated)
│   │   └── runtime/         # Engine, scheduler, NUMA, cancellation, futures
│   ├── features/            # Feature modules
│   │   ├── ai/              # AI features
│   │   │   ├── explore/     # Code exploration (AST, callgraph, dependency analysis)
│   │   │   ├── llm/         # Local LLM inference (GGUF, tokenizer, transformers)
│   │   │   ├── embeddings/  # Embedding models and caching
│   │   │   ├── rag/         # Retrieval-augmented generation
│   │   │   ├── streaming/   # Token streaming
│   │   │   └── memory/      # Conversation memory
│   │   ├── database/        # WDBX vector database (HNSW, hybrid search, batch ops)
│   │   ├── gpu/             # GPU backend stubs
│   │   ├── monitoring/      # Observability (OpenTelemetry)
│   │   ├── network/         # Network features (discovery, HA, circuit breaker)
│   │   └── connectors/      # API connectors (OpenAI, HuggingFace, Ollama)
│   ├── framework/           # Lifecycle and orchestration
│   ├── shared/              # Cross-cutting utilities
│   │   ├── logging/         # Logging infrastructure
│   │   ├── observability/   # Tracing, spans, contexts
│   │   ├── security/        # API keys, auth
│   │   └── utils/           # General utilities, HTTP, memory pools
│   └── tests/               # Test utilities (proptest)
├── tools/
│   └── cli/                 # CLI implementation (preferred over src/main.zig)
│       ├── commands/        # CLI command implementations
│       └── utils/           # CLI utilities
├── benchmarks/              # Modular benchmark suites
│   ├── framework.zig        # Statistical benchmarking framework
│   ├── main.zig             # Benchmark runner
│   └── <domain>.zig         # Domain-specific suites
├── examples/                # Example programs
├── bindings/
│   └── wasm/                # WASM bindings
└── docs/                    # Documentation
```

## Architecture

### Layered Structure

1. **Public API** (`src/abi.zig`) - Main entry point with curated re-exports. Use `abi.init()`, `abi.shutdown()`, `abi.version()`.

2. **Framework Layer** (`src/framework/`) - Lifecycle management, feature orchestration, runtime configuration, and plugin system.

3. **Compute Engine** (`src/compute/`) - Work-stealing scheduler, lock-free concurrent data structures, memory arena allocation, GPU integration with CPU fallback.

4. **Feature Stacks** (`src/features/`) - Vertical feature modules:
   - `ai/` - LLM inference (GGUF, transformers, KV cache), code exploration (AST parsing, callgraph, dependency analysis), embeddings, RAG, streaming, memory
   - `gpu/` - GPU backend implementations with fallback runtimes
   - `database/` - WDBX vector database with HNSW indexing, hybrid search, batch operations, full-text search, filtering
   - `web/` - HTTP client/server helpers
   - `monitoring/` - OpenTelemetry integration, logging, metrics, tracing
   - `network/` - Distributed task serialization, service discovery, circuit breaker, HA, rate limiting, retry logic

5. **Shared Utilities** (`src/shared/`) - Platform abstractions, SIMD acceleration, crypto, JSON, filesystem helpers.

### Key Patterns

- **Feature Gating with Stub Modules**: Compile-time feature enabling via build options checked with `build_options.enable_*`. When a feature is disabled, stub modules provide compile-time compatible placeholders that return `error.*Disabled` (e.g., `error.AiDisabled`, `error.NetworkDisabled`).

  **Stub Module Requirements**:
  - Must mirror the full API surface of the real module (all structs, functions, constants, type definitions)
  - Functions return appropriate `error.*Disabled` errors
  - Struct types must be defined (can be opaque or minimal)
  - Constants should have appropriate placeholder values
  - Import pattern: `const impl = if (build_options.enable_feature) @import("real.zig") else @import("stub.zig");`

  **Examples**: `src/features/ai/stub.zig`, `src/compute/network/stub.zig`, `src/features/ai/llm/stub.zig`, `src/compute/profiling/stub.zig`, `src/features/ai/explore/stub.zig`

- **GPU Backend Selection**: Multiple GPU backends with automatic fallback hierarchy. Vulkan devices are scored by type (discrete > integrated > virtual > cpu > other) during initialization. Available backends:
  - **CUDA** (NVIDIA) - Direct CUDA runtime + NVRTC JIT compilation with thread-safe initialization (`cuda.zig`, `cuda_nvrtc.zig`)
  - **Vulkan** (cross-platform) - Full Vulkan 1.3 support with compute pipelines, command buffer pooling. Key files: `vulkan.zig`, `vulkan_init.zig`, `vulkan_pipelines.zig`, `vulkan_buffers.zig`, `vulkan_command_pool.zig`
  - **Metal** (Apple) - Native Metal Shading Language support
  - **WebGPU** (web/native) - WebGPU compute API
  - **OpenGL/OpenGL ES** - Compute shaders via OpenGL 4.3+ / ES 3.1+
  - **stdgpu** (software fallback) - CPU-based SPIR-V interpreter, always available
  - **simulated** - Testing backend for CI/testing scenarios

  **Backend Infrastructure**:
  - **Device Scoring**: Intelligent GPU selection with scoring (discrete=1000, integrated=500, virtual=100, cpu=50, other=10 + API version bonus)
  - **Command Pool**: Vulkan command buffer pooling with state tracking, automatic recycling, and fence management (`vulkan_command_pool.zig`)
  - **Advanced Memory Pool**: Size-class based allocation (64B-4MB classes), automatic coalescing, fragmentation mitigation, memory pressure handling (`memory_pool_advanced.zig`)
  - **Metrics Collection**: Comprehensive performance tracking (kernel execution, memory transfers, device utilization, error rates, bandwidth) (`metrics.zig`)
  - **Recovery System**: Automatic device failure recovery with multiple strategies (retry with backoff, device switching, CPU fallback, simulation fallback) (`recovery.zig`)
  - **Error Handling**: Structured error tracking with specific error sets, health monitoring, and recovery event callbacks (`error_handling.zig`)

  See `src/compute/gpu/backends/vulkan_init.zig:selectPhysicalDevice()` and `src/compute/gpu/backends/stdgpu.zig:getDeviceInfo()`.

- **VTable Pattern**: Polymorphic workload execution for CPU and GPU variants (`WorkloadVTable`, `GPUWorkloadVTable`). Allows the same `WorkItem` to specify both CPU and GPU execution paths, selected at runtime based on availability and hints.

- **Allocator Ownership**: Explicit memory management throughout. Prefer `std.ArrayListUnmanaged` over `std.ArrayList` to make allocator passing explicit. All allocations must have corresponding cleanup in `deinit()` or via `defer`/`errdefer`.

- **Lifecycle Management**: Strict init/deinit patterns. Use `defer` for cleanup in success paths and `errdefer` for cleanup on error. Resources must be cleaned up in reverse order of initialization.

- **Memory Pooling**: GPU buffers use pooled allocation via `gpu.MemoryPool`. CPU-side memory uses arena allocation in compute contexts (`src/compute/memory/mod.zig`).

- **Error Context and Handling**: Use `ErrorContext` from `src/shared/utils/errors.zig` for structured error logging with operation context, category, and source location. Always use specific error sets instead of `anyerror`. Feature-disabled errors follow the pattern `error.<Feature>Disabled` (e.g., `error.AiDisabled`, `error.NetworkDisabled`, `error.ProfilingDisabled`).

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

When implementing or modifying GPU backends (`src/compute/gpu/backends/`):

**Standardized Backend Interface**:
All backends must implement this consistent interface:
```zig
// Lifecycle
pub fn init() !void
pub fn deinit() void

// Kernel operations
pub fn compileKernel(allocator, source) !*anyopaque
pub fn launchKernel(allocator, handle, config, args) !void
pub fn destroyKernel(allocator, handle) void

// Memory operations (explicit allocator parameter)
pub fn allocateDeviceMemory(allocator, size) !*anyopaque
pub fn freeDeviceMemory(allocator, ptr) void
pub fn memcpyHostToDevice(dst, src, size) !void
pub fn memcpyDeviceToHost(dst, src, size) !void
```

**Backend-Specific Patterns**:
- **Thread Safety**: Use mutex for initialization (see CUDA backend's `init_mutex`)
- **Graceful Fallback**: Separate simulation mode initialization for when hardware unavailable
- **Device Enumeration**: Use arena allocator for temporary allocations during device discovery
- **Symmetric Operations**: Every `allocate*` must have matching `free*` with same allocator
- **Format Specifiers**: Use `{t}` for enum/error values in logging (Zig 0.16 compliance)

**Command Buffer Management** (Vulkan):
- Use `CommandPool` for efficient command buffer allocation and recycling
- Track buffer state (available, recording, submitted, completed)
- Automatic fence management for synchronization
- Statistics collection via `getStats()`

**Memory Pool Integration**:
- Size classes: 64B, 256B, 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB
- Free-list management for fast reuse
- Automatic coalescing when utilization > 80%
- Memory pressure handling with high/low water marks (85%/70% default)

**Metrics Integration**:
- Record all kernel executions: `metrics.recordKernel(name, duration_ns)`
- Track memory transfers: `metrics.recordTransfer(direction, bytes, duration_ns)`
- Monitor allocations: `metrics.recordAllocation(bytes)` / `metrics.recordDeallocation(bytes)`
- Report errors: `metrics.recordError(error_type)`

**Recovery Integration**:
- Register devices: `recovery.registerDevice(backend_type, device_id)`
- Report failures: `recovery.reportError(backend_type, device_id, error_type)`
- Automatic fallback hierarchy: retry → switch_device → fallback_cpu → fallback_simulated

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

## CLI Commands

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

The LLM feature (`src/features/ai/llm/`) provides local model inference with:
- **Model I/O**: GGUF format loading, memory-mapped file access on Windows/POSIX (`io/gguf.zig`, `io/mmap.zig`, `io/tensor_loader.zig`)
- **Tokenization**: BPE (Byte-Pair Encoding) with vocab management and special tokens (`tokenizer/bpe.zig`, `tokenizer/vocab.zig`, `tokenizer/special_tokens.zig`)
- **Tensor Operations**: Quantized tensor support (Q4_0, Q4_1, Q8_0), tensor views and slicing (`tensor/quantized.zig`, `tensor/tensor.zig`, `tensor/view.zig`)
- **Transformer Operations**: Matrix multiplication (including quantized matmul), multi-head attention, RoPE (Rotary Position Embeddings), RMSNorm, SwiGLU/GELU activations, FFN layers (`ops/matmul.zig`, `ops/matmul_quant.zig`, `ops/attention.zig`, `ops/rope.zig`, `ops/rmsnorm.zig`, `ops/ffn.zig`, `ops/activations.zig`)
- **KV Cache**: Key-Value cache with ring buffer for efficient context management (`cache/kv_cache.zig`, `cache/ring_buffer.zig`)
- **Generation**: Batch generation, sampling strategies (greedy, top-k, top-p, temperature) (`generation/batch.zig`, `generation/generator.zig`, `generation/sampler.zig`)
- **Model Architectures**: LLaMA family support with configurable layers (`model/llama.zig`, `model/layer.zig`, `model/config.zig`, `model/weights.zig`)

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

### WDBX Database Namespace

When `enable-database=true`, the `abi.wdbx` namespace provides direct access to vector database operations:

```zig
const db = try abi.wdbx.createDatabase(allocator, "vectors.db", .{});
defer abi.wdbx.closeDatabase(db);

try abi.wdbx.insertVector(db, 1, &vector, metadata);
const results = try abi.wdbx.searchVectors(db, &query, 10);
```

**Database Features**:
- **HNSW Indexing**: Hierarchical Navigable Small World graphs for fast approximate nearest neighbor search
- **Hybrid Search**: Combined vector similarity + full-text search + metadata filtering
- **Batch Operations**: Bulk insert/update/delete for efficiency (`src/features/database/batch.zig`)
- **Full-Text Search**: Integrated text search capabilities (`src/features/database/fulltext.zig`)
- **Filtering**: Advanced metadata filtering during search (`src/features/database/filter.zig`)
- **Reindexing**: Background reindexing for optimization (`src/features/database/reindex.zig`)
- **Sharding**: Distributed sharding support (`src/features/database/shard.zig`)
- **HTTP API**: RESTful API for remote access (`src/features/database/http.zig`)

## Commit Guidelines

Use `<type>: <summary>` format. Keep summaries ≤ 72 chars. Focus commits; update docs when APIs change.

## Architecture References
## Contacts

`src/shared/contacts.zig` provides a centralized list of maintainer contacts extracted from the repository markdown files.

- System overview: `docs/intro.md`
- API surface: `API_REFERENCE.md`
- Migration guide: `docs/migration/zig-0.16-migration.md`
