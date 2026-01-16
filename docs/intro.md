# Introduction

Welcome to **ABI**, a modern Zig 0.16.x framework for modular AI services, vector search, and high-performance systems tooling.

---

## Philosophy

ABI is built on three core pillars:

1. **Modularity** - Use only what you need. Core features are isolated, and advanced subsystems (AI, GPU, Database) are opt-in via build flags.

2. **Performance** - Written in Zig 0.16.x, leveraging a work-stealing compute runtime, lock-free data structures, and zero-copy patterns.

3. **Modernity** - Native support for vector embeddings, AI agents, GPU acceleration, and distributed compute.

---

## Architecture

The framework is organized into five layers:

```
┌─────────────────────────────────────────────────────────┐
│                    Public API (abi.zig)                 │
│         init(), shutdown(), version(), namespaces      │
├─────────────────────────────────────────────────────────┤
│                  Framework (src/framework/)             │
│      Lifecycle, Configuration, Feature Orchestration   │
├─────────────────────────────────────────────────────────┤
│                 Compute Engine (src/compute/)           │
│   Work-Stealing Scheduler, GPU Integration, Memory     │
├─────────────────────────────────────────────────────────┤
│                Feature Stacks (src/features/)           │
│        AI, Database, GPU, Network, Monitoring, Web     │
├─────────────────────────────────────────────────────────┤
│               Shared Utilities (src/shared/)            │
│     Platform Abstractions, SIMD, Crypto, Logging       │
└─────────────────────────────────────────────────────────┘
```

### Layer 1: Public API (`src/abi.zig`)

The entry point for all ABI applications. Provides:

- `abi.init(allocator, options)` - Initialize the framework
- `abi.shutdown(&framework)` - Clean shutdown
- `abi.version()` - Get version string
- Curated re-exports of feature namespaces

### Layer 2: Framework (`src/framework/`)

Manages the application lifecycle:

- **Initialization** - Set up allocators, configure features
- **Configuration** - Runtime options via `FrameworkOptions`
- **Feature Orchestration** - Enable/disable features at build and runtime
- **Plugin System** - Runtime-loadable extensions

### Layer 3: Compute Engine (`src/compute/`)

High-performance parallel execution:

- **Work-Stealing Scheduler** - Efficient task distribution across threads
- **GPU Integration** - Automatic GPU offloading with CPU fallback
- **Memory Management** - Arena allocation, pooling, NUMA awareness
- **Concurrency Primitives** - Lock-free queues, sharded maps, futures

### Layer 4: Feature Stacks (`src/features/`)

Domain-specific modules:

| Feature | Description |
|---------|-------------|
| **AI** | LLM connectors (OpenAI, Ollama, HuggingFace), agent runtime, training |
| **Database** | WDBX vector database, HNSW indexing, hybrid search |
| **GPU** | Multi-backend support (CUDA, Vulkan, Metal, WebGPU), unified API |
| **Network** | Distributed compute, node discovery, Raft consensus |
| **Monitoring** | Logging, metrics, alerting, tracing, profiling |
| **Web** | HTTP client/server, async I/O |
| **Connectors** | Discord, local scheduler integrations |

### Layer 5: Shared Utilities (`src/shared/`)

Cross-cutting concerns:

- **Platform** - OS abstractions, path handling
- **SIMD** - Vectorized operations
- **Crypto** - Hashing, secure random
- **Logging** - Structured log output
- **Filesystem** - File utilities

---

## Feature Gating

Features are enabled/disabled at build time:

```bash
# Enable specific features
zig build -Denable-ai=true -Denable-gpu=true

# Disable features to reduce binary size
zig build -Denable-network=false -Denable-profiling=false
```

When a feature is disabled, stub modules provide compile-time compatible placeholders that return `error.*Disabled` (e.g., `error.AiDisabled`).

---

## Quick Start

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize with default options
    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    // Check enabled features
    if (framework.isFeatureEnabled(.ai)) {
        std.debug.print("AI features available\n", .{});
    }

    std.debug.print("ABI v{s} initialized\n", .{abi.version()});
}
```

---

## Next Steps

- [Framework Guide](framework.md) - Configuration and lifecycle
- [Compute Engine](compute.md) - Task execution and scheduling
- [AI & Agents](ai.md) - LLM connectors and agent runtime
- [Database](database.md) - Vector database operations
- [GPU Acceleration](gpu.md) - GPU backends and unified API

---

## See Also

- [Documentation Index](index.md) - Full documentation listing
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Zig 0.16 Migration](migration/zig-0.16-migration.md) - API compatibility notes
 - [TODO List](TODO.md) - Pending implementations and Llama‑CPP parity tasks
