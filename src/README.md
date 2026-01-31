---
title: "Source Directory"
tags: [source, architecture, modules]
---
# Source Directory
> **Codebase Status:** Synced with repository as of 2026-01-30.

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Modular-blue?style=for-the-badge" alt="Modular"/>
  <img src="https://img.shields.io/badge/Modules-15+-green?style=for-the-badge" alt="15+ Modules"/>
  <img src="https://img.shields.io/badge/Zig-0.16-F7A41D?style=for-the-badge&logo=zig&logoColor=white" alt="Zig"/>
</p>

Core source modules of the ABI framework organized by function.

## Structure

The codebase uses a modular architecture with top-level modules that re-export
from their implementation directories while adding Framework integration via
Context structs.

| Directory | Description |
|-----------|-------------|
| `abi.zig` | Public API entry point with curated re-exports |
| `config.zig` | Unified configuration system (struct literal + builder APIs) |
| `framework.zig` | Framework orchestration and lifecycle management |
| `registry/` | Plugin registry system (comptime, runtime-toggle, dynamic modes) |
| `runtime/` | Always-on infrastructure (engine, scheduling, concurrency, memory) |
| `gpu/` | GPU acceleration with unified multi-backend API |
| `ai/` | AI module with sub-features (llm, embeddings, agents, training) |
| `database/` | Vector database (primary implementation) |
| `network/` | Distributed compute (primary implementation) |
| `observability/` | Metrics, tracing, profiling (consolidated from features/monitoring) |
| `web/` | Web/HTTP utilities (primary implementation) |
| `tasks/` | Task management system (roadmap, tracking) |
| `core/` | Core infrastructure, hardware helpers |
| `compute/` | Legacy re-exports (backward compat - use `runtime/` instead) |
| `features/` | Legacy feature implementations (ai, connectors, ha) |
| `shared/` | Cross-cutting utilities (logging, platform, utils) |
| `tests/` | Test utilities, property-based testing, stub parity verification |

## Module Hierarchy

```
src/
├── abi.zig              # Public API
├── config.zig           # Unified configuration
├── framework.zig        # Framework orchestration
│
├── registry/            # Feature registry system
│   ├── mod.zig          # Public API facade with Registry struct
│   ├── types.zig        # Core types (Feature, RegistrationMode, Error)
│   ├── registration.zig # registerComptime, registerRuntimeToggle, registerDynamic
│   └── lifecycle.zig    # initFeature, deinitFeature, enable/disable
│
├── runtime/             # Always-on infrastructure (CONSOLIDATED)
│   ├── mod.zig          # Unified entry point
│   ├── engine/          # Work-stealing task execution
│   ├── scheduling/      # Futures, cancellation, task groups
│   ├── concurrency/     # Lock-free data structures
│   ├── memory/          # Arena allocators, pools
│   └── workload.zig     # Workload detection
│
├── gpu/                 # GPU acceleration
│   ├── mod.zig          # Unified GPU API with backends, DSL, profiling
│   └── stub.zig         # Feature-disabled stub
│
├── ai/                  # AI module
│   ├── mod.zig          # Re-exports from features/ai + Context
│   ├── llm/             # LLM inference sub-feature
│   ├── embeddings/      # Embeddings generation sub-feature
│   ├── agents/          # Agent runtime sub-feature
│   └── training/        # Training pipelines sub-feature
│
├── database/            # Vector database
│   ├── mod.zig          # Primary implementation with Context
│   └── stub.zig         # Feature-disabled stub
│
├── network/             # Distributed compute
│   ├── mod.zig          # Primary implementation with Context
│   └── stub.zig         # Feature-disabled stub
│
├── observability/       # Metrics and tracing
│   ├── mod.zig          # Re-exports + Context
│   └── stub.zig         # Feature-disabled stub
│
├── web/                 # Web utilities
│   ├── mod.zig          # Primary implementation with Context
│   └── stub.zig         # Feature-disabled stub
│
├── tasks/               # Task management
│   ├── mod.zig          # Task manager, roadmap, tracking
│   └── types.zig        # Task and milestone types
│
├── compute/             # Legacy re-exports (backward compat)
│   └── mod.zig          # Re-exports from runtime/ for compatibility
│
├── features/            # Legacy feature implementations
│   ├── ai/              # AI (LLM, embeddings, RAG) - still active
│   ├── connectors/      # API connectors - still active
│   └── ha/              # High availability - still active
│
└── shared/              # Cross-cutting concerns
    ├── logging/         # Logging infrastructure
    ├── observability/   # Tracing, metrics
    ├── platform/        # OS abstractions
    ├── plugins/         # Plugin system
    ├── security/        # API keys, auth
    └── utils/           # General utilities
```

## Key Entry Points

- **Public API**: `abi.zig` - Use `abi.init()`, `abi.shutdown()`, `abi.version()`
- **Configuration**: `config.zig` - Unified `Config` struct with `Builder` API
- **Framework**: `framework.zig` - `Framework` struct manages feature lifecycle
- **Runtime**: `runtime/mod.zig` - Always-available scheduling and concurrency

## Module Pattern

Top-level modules (gpu, ai, database, network, observability, web) follow this pattern:

1. Re-export types and functions from implementation directories
2. Add a `Context` struct for Framework integration
3. Provide `isEnabled()` function for compile-time feature detection
4. Have a corresponding `stub.zig` for when the feature is disabled

```zig
// Example: src/gpu/mod.zig
const unified = @import("unified.zig");

// Re-exports
pub const Gpu = unified.Gpu;
pub const Buffer = unified.GpuBuffer;

// Context for Framework integration
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.GpuConfig,
    gpu: ?Gpu = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.GpuConfig) !*Context { ... }
    pub fn deinit(self: *Context) void { ... }
};

pub fn isEnabled() bool {
    return build_options.enable_gpu;
}
```

## See Also

- [CLAUDE.md](../CLAUDE.md) - Full project documentation
- [API Reference](../API_REFERENCE.md)
- [Docs Map](../docs/README.md) - Documentation layout and entry points
- [docs/intro.md](../docs/intro.md) - Architecture overview
