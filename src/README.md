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

The codebase uses a modular architecture with top-level modules that expose
stable APIs and Context structs for Framework integration.

| Directory | Description |
|-----------|-------------|
| `abi.zig` | Public API entry point with curated re-exports |
| `config.zig` | Unified configuration system (struct literal + builder APIs) |
| `framework.zig` | Framework orchestration and lifecycle management |
| `flags.zig` | Feature flag definitions |
| `config/` | Modular configuration per feature |
| `registry/` | Feature registry system (comptime, runtime-toggle, dynamic modes) |
| `runtime/` | Always-on infrastructure (engine, scheduling, concurrency, memory) |
| `platform/` | Platform detection and CPU feature abstraction |
| `shared/` | Cross-cutting utilities (logging, platform, utils) |
| `ai/` | AI module with sub-features (llm, embeddings, agents, training) |
| `gpu/` | GPU acceleration with unified multi-backend API |
| `database/` | Vector database |
| `network/` | Distributed compute |
| `observability/` | Metrics, tracing, profiling |
| `web/` | Web/HTTP utilities |
| `cloud/` | Cloud provider adapters |
| `connectors/` | External provider connectors (OpenAI, Ollama, etc.) |
| `ha/` | High availability (backup, PITR, replication) |
| `tasks/` | Task management system (roadmap, tracking) |

## Module Hierarchy

```
src/
├── abi.zig              # Public API
├── config.zig           # Unified configuration
├── framework.zig        # Framework orchestration
├── flags.zig            # Feature flags
│
├── config/              # Modular configuration
│   ├── mod.zig
│   ├── ai.zig
│   ├── cloud.zig
│   ├── database.zig
│   ├── gpu.zig
│   ├── network.zig
│   ├── observability.zig
│   ├── plugin.zig
│   └── web.zig
│
├── registry/            # Feature registry system
│   ├── mod.zig          # Public API facade with Registry struct
│   ├── types.zig        # Core types (Feature, RegistrationMode, Error)
│   ├── registration.zig # registerComptime, registerRuntimeToggle, registerDynamic
│   └── lifecycle.zig    # initFeature, deinitFeature, enable/disable
│
├── runtime/             # Always-on infrastructure
│   ├── mod.zig          # Unified entry point
│   ├── engine/          # Work-stealing task execution
│   ├── scheduling/      # Futures, cancellation, task groups
│   ├── concurrency/     # Lock-free data structures
│   └── memory/          # Allocators, pools
│
├── platform/            # Platform detection and abstraction
│   ├── mod.zig
│   ├── cpu.zig
│   ├── detection.zig
│   └── stub.zig
│
├── shared/              # Cross-cutting concerns
│   ├── mod.zig
│   ├── logging.zig
│   ├── platform.zig
│   ├── plugins.zig
│   ├── security/
│   └── utils/
│
├── ai/                  # AI module
│   ├── mod.zig
│   ├── llm/
│   ├── embeddings/
│   ├── agents/
│   ├── training/
│   ├── streaming/
│   ├── rag/
│   ├── documents/
│   └── ...
│
├── gpu/                 # GPU acceleration
│   ├── mod.zig
│   └── stub.zig
│
├── database/            # Vector database
│   ├── mod.zig
│   └── stub.zig
│
├── network/             # Distributed compute
│   ├── mod.zig
│   └── stub.zig
│
├── observability/       # Metrics and tracing
│   ├── mod.zig
│   └── stub.zig
│
├── web/                 # Web utilities
│   ├── mod.zig
│   └── stub.zig
│
├── cloud/               # Cloud provider adapters
│   ├── mod.zig
│   └── ...
│
├── connectors/          # External provider connectors
│   ├── mod.zig
│   └── ...
│
├── ha/                  # High availability
│   ├── mod.zig
│   └── ...
│
└── tasks/               # Task management
    ├── mod.zig
    └── types.zig
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
