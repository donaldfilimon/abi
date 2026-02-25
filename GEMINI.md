<<<<<<< Current (Your changes)
=======
---
title: "GEMINI"
tags: [ai, agents, gemini]
---
# GEMINI.md
> **Codebase Status:** Synced with repository as of 2026-01-31.

<p align="center">
  <img src="https://img.shields.io/badge/Gemini-Agent_Guide-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini Guide"/>
  <img src="https://img.shields.io/badge/Zig-0.16-F7A41D?style=for-the-badge&logo=zig&logoColor=white" alt="Zig"/>
</p>

This file gives the Gemini agent guidance on interacting with the ABI framework. It mirrors the structure of `AGENTS.md` and `CLAUDE.md` but is tuned for Gemini‑like prompts.

## Quick Start for Gemini

```bash
zig build                             # Build the framework
zig build test --summary all          # Run all tests (regression)
zig build cli-tests                   # Run CLI command smoke tests
zig build full-check                  # Format + tests + CLI smoke tests
zig fmt .                             # Format after edits
zig build run -- --help               # CLI help

# Example programs
zig build run-hello                    # Run hello example
zig build run-database                 # Run database example
zig build run-agent                    # Run agent example
```

Typical Gemini usage:

```bash
zig build run -- llm generate "Hello" --max 60
```

## Codebase Overview

**ABI Framework** is a Zig 0.16 framework for modular AI services, GPU compute, and vector databases.

**Key Features:**
- **Lock-free Concurrency**: Chase-Lev deque, epoch reclamation, MPMC queues, NUMA-aware stealing
- **Quantized Inference**: Q4/Q8 CUDA kernels with 4x/2x memory savings
- **Result Caching**: Sharded LRU with TTL support for task memoization
- **Parallel Search**: SIMD-accelerated HNSW batch queries

### Architecture

The codebase uses a domain-driven modular structure with unified configuration and compile-time feature gating:

```
src/
├── abi.zig              # Public API entry point: init(), shutdown(), version()
├── config.zig           # Unified configuration system (single Config struct)
├── config/              # Modular configuration system
├── framework.zig        # Framework orchestration with builder pattern
├── platform/            # Platform detection (NEW: mod.zig, detection.zig, cpu.zig)
├── runtime/             # Always-on infrastructure
├── gpu/                 # GPU acceleration
├── ai/                  # AI features (llm, embeddings, agents, training)
├── database/            # Vector database
├── network/             # Distributed networking
├── observability/       # Metrics, tracing, logging
├── web/                 # Web utilities
├── shared/              # Shared utilities (mod.zig, io.zig, security/, utils/)
├── connectors/          # External API connectors
├── cloud/               # Cloud function adapters
├── ha/                  # High availability
├── registry/            # Feature registry
├── tasks/               # Task management
└── tests/               # Test infrastructure
```

## Zig 0.16 Patterns (CRITICAL)

See `CLAUDE.md` for current Zig 0.16 patterns and examples.

### I/O Backend Initialization

Zig 0.16 requires explicit I/O backend initialization for file and network operations.

```zig
// Initialize once, use for all file/network operations
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty,  // .empty for library, .init() for CLI
});
defer io_backend.deinit();
const io = io_backend.io();

// File read
const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024));
defer allocator.free(content);
```

### Other Changes

```zig
// Error/enum formatting: use {t} instead of @errorName()/@tagName()
std.debug.print("Error: {t}", .{err});

// ArrayListUnmanaged: use .empty not .init()
var list = std.ArrayListUnmanaged(u8).empty;

// Timing: use Timer.start() not Instant.now()
var timer = std.time.Timer.start() catch return error.TimerFailed;

// HTTP server: use .interface for reader/writer
var server: std.http.Server = .init(
    &connection_reader.interface,
    &connection_writer.interface,
);
```

## Feature Management

Feature flags live in `build.zig` and correlate with `build_options`:

```zig
const build_options = struct{
    enable_ai: bool = true,
    enable_gpu: bool = true,
    enable_database: bool = true,
    // ... add others as needed
};
```

**Stub-Real Sync:** Changes to `src/<feature>/mod.zig` must be propagated to its `stub.zig`. The CI parity checker will report mismatches.

## New Runtime Types (2026-01-25)

### Lock-free Concurrency
```zig
const runtime = @import("abi").runtime;

// Chase-Lev work-stealing deque
var deque = runtime.ChaseLevDeque(Task).init(allocator);

// Epoch-based reclamation (ABA-safe)
var epoch = runtime.EpochReclamation(Node).init(allocator);

// MPMC bounded queue
var queue = try runtime.MpmcQueue(Msg).init(allocator, 1024);

// Result caching with TTL
var cache = try runtime.ResultCache(K, V).init(allocator, .{});

// NUMA-aware work stealing
var policy = try runtime.NumaStealPolicy.init(allocator, null, 8, .{});
```

### Quantized CUDA Kernels
```zig
const cuda = @import("abi").gpu.cuda;

var quant = try cuda.QuantizedKernelModule.init(allocator);
defer quant.deinit();

// Q4_0/Q8_0 matrix-vector multiplication
try quant.q4Matmul(a_ptr, x_ptr, y_ptr, m, k, stream);

// Configuration
const config = cuda.QuantConfig.forInference();
```

### Stream Error Recovery (2026-01-31)
```zig
const streaming = @import("abi").ai.streaming;

// Per-backend circuit breakers
var recovery = try streaming.StreamRecovery.init(allocator, .{});
defer recovery.deinit();

// Check if backend is available (circuit closed)
if (recovery.isBackendAvailable(.openai)) {
    // Safe to use
}

// Record outcomes to update circuit state
recovery.recordSuccess(.local);
recovery.recordFailure(.openai);  // Opens circuit after threshold

// Session caching for SSE Last-Event-ID reconnection
var cache = streaming.SessionCache.init(allocator, .{});
try cache.storeToken("session", 1, "Hello", .local, hash);
```

## Gotchas for Gemini
* **Stub‑Real Sync** – Changes to `src/<feature>/mod.zig` must be propagated to its `stub.zig`.
* **GPU Backends** – Use `-Dgpu-backend=auto` or `-Dgpu-backend=cuda,vulkan` (comma‑separated).
* **WASM Compatibility** – `database`, `network`, and `gpu` modules auto‑disable on WASM.
* **Compile‑time Flags** – `-D` options modify the compiled binary. `--enable-…` only toggles runtime behaviour.
* **Format Specifiers** – Always use `{t}` for printing errors and enums.

★ *Tip:* After modifications run `zig fmt .` followed by `zig build test --summary all` to verify nothing broke.
>>>>>>> Incoming (Background Agent changes)
