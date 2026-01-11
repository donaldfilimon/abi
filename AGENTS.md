# AGENTS.md

This file provides guidance to AI agents (GitHub Copilot, Cursor, etc.) when working with code in this repository.

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

**Note**: See `build.zig` `Defaults` struct for current default values.

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

### Benchmark Suites

```bash
zig build bench-simd         # SIMD/Vector benchmarks
zig build bench-memory       # Memory allocator benchmarks
zig build bench-concurrency  # Concurrency benchmarks
zig build bench-database     # Database/HNSW benchmarks
zig build bench-network      # HTTP/network benchmarks
zig build bench-crypto       # Cryptography benchmarks
zig build bench-ai           # AI/ML inference benchmarks
zig build bench-quick        # Quick CI benchmarks
```

## Project Structure

```
abi/
├── src/
│   ├── abi.zig              # Public API entry point
│   ├── root.zig             # Root module
│   ├── compute/             # Compute engine and concurrency
│   │   ├── concurrency/     # Lock-free queues, work-stealing
│   │   ├── gpu/             # GPU integration layer
│   │   ├── memory/          # Arena allocators, pools
│   │   ├── network/         # Distributed compute (feature-gated)
│   │   ├── profiling/       # Metrics collection (feature-gated)
│   │   └── runtime/         # Engine, scheduler, NUMA
│   ├── features/            # Feature modules
│   │   ├── ai/              # AI features (agents, LLM, exploration)
│   │   ├── database/        # WDBX vector database
│   │   ├── gpu/             # GPU backend stubs
│   │   ├── monitoring/      # Observability
│   │   ├── network/         # Network features
│   │   └── connectors/      # API connectors (OpenAI, HF, Ollama)
│   ├── framework/           # Lifecycle and orchestration
│   ├── shared/              # Cross-cutting utilities
│   └── tests/               # Test utilities (proptest)
├── tools/cli/               # CLI implementation
├── benchmarks/              # Benchmark suites
├── examples/                # Example programs
├── bindings/wasm/           # WASM bindings
└── docs/                    # Documentation
```

## Architecture

### Layered Structure

1. **Public API** (`src/abi.zig`) - Main entry point with curated re-exports. Use `abi.init()`, `abi.shutdown()`, `abi.version()`.

2. **Framework Layer** (`src/framework/`) - Lifecycle management, feature orchestration, runtime configuration, and plugin system.

3. **Compute Engine** (`src/compute/`) - Work-stealing scheduler, lock-free concurrent data structures, memory arena allocation, GPU integration with CPU fallback.

4. **Feature Stacks** (`src/features/`) - Vertical feature modules (ai, gpu, database, web, monitoring, network).

5. **Shared Utilities** (`src/shared/`) - Platform abstractions, SIMD acceleration, crypto, JSON, filesystem helpers.

### Key Patterns

- **Feature Gating with Stub Modules**: Compile-time feature enabling via `build_options.enable_*`. When disabled, stub modules return `error.*Disabled`. Stubs must mirror the full API surface.
  - Import pattern: `const impl = if (build_options.enable_feature) @import("real.zig") else @import("stub.zig");`
  - Examples: `src/features/ai/stub.zig`, `src/compute/network/stub.zig`, `src/features/ai/llm/stub.zig`

- **GPU Backend Selection**: Vulkan devices scored by type (discrete > integrated > virtual > cpu > other). `stdgpu` provides CPU fallback.

- **VTable Pattern**: Polymorphic workload execution via `WorkloadVTable` and `GPUWorkloadVTable`.

- **Allocator Ownership**: Explicit memory management. Prefer `std.ArrayListUnmanaged` over `std.ArrayList`.

- **Lifecycle Management**: Strict init/deinit patterns. Use `defer`/`errdefer` for cleanup.

### Concurrency Primitives

- **WorkStealingQueue**: LIFO for owner, FIFO for thieves
- **LockFreeQueue/LockFreeStack**: Atomic CAS-based collections
- **ShardedMap**: Reduces lock contention via partitioning
- **Backoff**: Exponential backoff with spin-loop hints

### NUMA and CPU Affinity

Enable with `EngineConfig{ .numa_enabled = true, .cpu_affinity_enabled = true }`. See `src/compute/runtime/numa.zig`.

## Zig 0.16 Conventions

### Memory Management
```zig
// Preferred - use unmanaged for struct fields
pub const BenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(BenchmarkResult),
};
```

### Format Specifiers
```zig
std.debug.print("{t}: {t}\n", .{status, status});  // {t} for enums/errors
std.debug.print("Size: {B}\n", .{size});           // {B} for byte sizes
std.debug.print("Duration: {D}\n", .{dur});        // {D} for durations
```

### I/O API
```zig
// HTTP Server - pass readers/writers directly
var srv = std.http.Server.init(&reader, &writer);
```

### std.Io.Threaded
```zig
var io = std.Io.Threaded.init(alloc, .{});
defer io.deinit();
// Use io.io() for operations
```

**Note**: `std.fs.cwd()` does not exist in Zig 0.16. Use `std.Io.Dir.cwd()` with I/O context.

### Timing
```zig
var timer = std.time.Timer.start() catch return error.TimerFailed;
const elapsed_ns = timer.read();
```

## Coding Style

- 4 spaces, no tabs, max 100 chars/line, one blank line between functions
- **Types**: PascalCase (`Engine`, `TaskConfig`)
- **Functions/Variables**: snake_case (`create_engine`, `task_id`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_TASKS`, `DEFAULT_MAX_TASKS`)
- **Struct Fields**: `allocator` first, then config/state, collections, resources, flags
- Explicit imports only (no `usingnamespace` except for re-exports)
- Use specific error sets, not `anyerror`
- Prefer `defer`/`errdefer` for cleanup

## Testing Guidelines

Tests in `src/tests/` and inline `test "..."` blocks. Use `error.SkipZigTest` to gate hardware-specific tests:
```zig
test "gpu-feature" {
    if (!build_options.enable_gpu) return error.SkipZigTest;
    // test body
}
```

## Environment Variables

- `ABI_OPENAI_API_KEY` / `OPENAI_API_KEY`
- `ABI_OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
- `ABI_OPENAI_MODE` (`responses`, `chat`, or `completions`)
- `ABI_HF_API_TOKEN` / `HF_API_TOKEN` / `HUGGING_FACE_HUB_TOKEN`
- `ABI_OLLAMA_HOST` / `OLLAMA_HOST` (default: `http://127.0.0.1:11434`)
- `ABI_OLLAMA_MODEL` (default: `llama3.2`)

## Key API Notes

### Compute Engine Timeouts

- `timeout_ms=0`: Immediately returns `EngineError.Timeout` if not ready
- `timeout_ms>0`: Waits specified milliseconds before timeout
- `timeout_ms=null`: Waits indefinitely

### WDBX Backup/Restore Security

Restricted to `backups/` directory. No path traversal (`..`), absolute paths, or Windows drive letters.

### WASM Constraints

Auto-disabled: `enable-database`, `enable-network`, `enable-gpu`, `enable-web`.

## Architecture References

- System overview: `docs/intro.md`
- API surface: `API_REFERENCE.md`
- Migration guide: `docs/migration/zig-0.16-migration.md`

## Commit Guidelines

Use `<type>: <summary>` format. Keep summaries ≤ 72 chars. Focus commits; update docs when APIs change.
