# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
zig build                    # Build all modules
zig build test               # Run all tests
zig build test --summary all # Detailed test output
zig build run -- --help      # Run CLI with help
zig fmt .                    # Format code (only linter)
zig fmt --check .            # Check formatting
```

**Single-file testing (recommended for focused work):**
```bash
zig test src/compute/runtime/engine.zig      # Test specific file
zig test --test-filter "engine init"         # Filter by test name
```

**Feature flags:**
```bash
zig build -Denable-gpu=false -Denable-network=true
zig build test -Denable-gpu=true -Denable-network=true -Denable-profiling=true
```

Defaults: `enable-ai`, `enable-gpu`, `enable-web`, `enable-database` are true; `enable-network`, `enable-profiling` are false.

GPU backends: `-Dgpu-cuda`, `-Dgpu-vulkan`, `-Dgpu-metal`, `-Dgpu-webgpu`, `-Dgpu-opengl`

## Architecture

```
src/
├── abi.zig          # Public API surface (main import point)
├── cli.zig          # CLI implementation
├── core/            # Hardware helpers, cache alignment
├── compute/         # Runtime engine, concurrency, memory, GPU backends
│   ├── runtime/     # Work-stealing scheduler
│   ├── concurrency/ # Lock-free data structures
│   ├── memory/      # Pool allocators
│   ├── gpu/         # GPU backends (CUDA, Vulkan, Metal, WebGPU, OpenGL)
│   └── network/     # Task serialization for distributed compute
├── features/        # Vertical feature stacks (opt-in via build flags)
│   ├── ai/          # Agents, transformers, training, federated learning
│   ├── database/    # WDBX vector database
│   ├── web/         # HTTP client/server
│   ├── monitoring/  # Observability, tracing
│   └── connectors/  # External services (OpenAI, HuggingFace, Ollama)
├── framework/       # Lifecycle, config, feature orchestration
└── shared/          # Logging, utils, security, platform helpers
```

**Key principle:** Features are opt-in. Each feature in `src/features/` is self-contained.

## Code Patterns

### Framework Initialization
```zig
var framework = try abi.init(allocator, abi.FrameworkOptions{
    .enable_ai = true,
    .enable_database = true,
    .enable_gpu = false,
});
defer abi.shutdown(&framework);
```

### Feature Gating
```zig
if (!abi.gpu.moduleEnabled()) {
    std.debug.print("GPU module disabled\n", .{});
    return;
}
```

### Memory Management (Zig 0.16)
```zig
// Use Unmanaged for struct fields
pub const MyStruct = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(Item),  // NOT ArrayList
};

// Usage
try list.append(allocator, item);
list.deinit(allocator);
```

### HTTP/Async (Zig 0.16)
```zig
// std.Io.Threaded for async clients
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty,
});
defer io_backend.deinit();

// HTTP server - direct reader/writer references
var server: std.http.Server = .init(
    &connection_reader,  // Direct, not .interface
    &connection_writer,
);

// File.Reader delimiter methods use .interface
const line = reader.interface.takeDelimiter('\n');
```

### Format Specifiers
```zig
std.debug.print("Status: {t}\n", .{status});     // {t} for enums/errors
std.debug.print("Duration: {D}\n", .{duration}); // {D} for nanoseconds
```

## Naming Conventions

- Types: `PascalCase` (Engine, TaskConfig)
- Functions/variables: `snake_case` (createEngine, task_id)
- Constants: `UPPER_SNAKE_CASE` (MAX_TASKS)
- Docs: `//!` for modules, `///` for functions

## Requirements

- **Zig 0.16.0 or newer** (compile-time check in `src/abi.zig`)
- No external linter—only `zig fmt`

## Connector Environment Variables

- OpenAI: `ABI_OPENAI_API_KEY`, `ABI_OPENAI_BASE_URL`, `ABI_OPENAI_MODE`
- HuggingFace: `ABI_HF_API_TOKEN`, `ABI_HF_BASE_URL`
- Ollama: `ABI_OLLAMA_HOST`, `ABI_OLLAMA_MODEL`

## Key APIs

```zig
// Framework
abi.init(allocator, options)
abi.shutdown(&framework)
abi.version()

// Compute
abi.compute.createDefaultEngine(allocator)
abi.compute.runTask(&engine, ResultType, taskFn, timeout_ms)

// GPU
abi.gpu.moduleEnabled()
abi.gpu.availableBackends(allocator)
abi.gpu.createPool(allocator, size)

// Database (WDBX)
abi.database.createDatabase(allocator, name)
abi.database.insertVector(handle, id, vector, metadata)
abi.database.searchVectors(handle, query, k)

// SIMD
abi.simd.vectorAdd(a, b, result)
abi.simd.vectorDot(a, b)
abi.simd.cosineSimilarity(a, b)
```

## Documentation References

- Architecture overview: `docs/intro.md`
- API reference: `API_REFERENCE.md`
- Zig 0.16 migration: `docs/migration/zig-0.16-migration.md`
- Coding style: `AGENTS.md`
