---
title: "Source Directory"
tags: [source, architecture, modules]
---
# Source Directory
> **Codebase Status:** Synced with repository as of 2026-01-31.

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
| `flags.zig` | Compile-time feature flag definitions |
| `framework.zig` | Framework orchestration and lifecycle management |
| `flags.zig` | Feature flag definitions |
| `config/` | Modular configuration per feature |
| `registry/` | Feature registry system (comptime, runtime-toggle, dynamic modes) |
| `runtime/` | Always-on infrastructure (engine, scheduling, concurrency, memory) |
| `platform/` | Platform detection and SIMD capabilities |
| `shared/` | Cross-cutting utilities (logging, io, security, utils) |
| `ai/` | AI module with sub-features (llm, embeddings, agents, training) |
| `gpu/` | GPU acceleration with unified multi-backend API |
| `database/` | Vector database (primary implementation) |
| `network/` | Distributed compute (primary implementation) |
| `observability/` | Metrics, tracing, profiling |
| `web/` | Web/HTTP utilities (primary implementation) |
| `connectors/` | External API connectors (OpenAI, Ollama, Anthropic, etc.) |
| `cloud/` | Cloud function adapters (AWS, GCP, Azure) |
| `ha/` | High availability (backup, PITR, replication) |
| `tasks/` | Task management system (roadmap, tracking) |
| `tests/` | Test utilities, property-based testing, stub parity verification |

## Module Hierarchy

```
src/
├── abi.zig              # Public API
├── config.zig           # Unified configuration
├── flags.zig            # Compile-time feature flags
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
├── platform/            # Platform detection and SIMD capabilities
│   ├── mod.zig          # Platform entry point
│   ├── detection.zig    # OS/arch detection
│   └── cpu.zig          # CPU feature helpers
│
├── shared/              # Cross-cutting concerns
│   ├── logging.zig      # Logging infrastructure
│   ├── io.zig           # I/O helpers
│   ├── security/        # API keys, auth
│   └── utils/           # General utilities
│
├── ai/                  # AI module
│   ├── mod.zig          # Re-exports + Context
│   ├── llm/             # LLM inference sub-feature
│   ├── embeddings/      # Embeddings generation sub-feature
│   ├── agents/          # Agent runtime sub-feature
│   └── training/        # Training pipelines sub-feature
│
├── gpu/                 # GPU acceleration
│   ├── mod.zig          # Unified GPU API with backends, DSL, profiling
│   └── stub.zig         # Feature-disabled stub
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
├── connectors/          # External API connectors
├── cloud/               # Cloud function adapters
├── ha/                  # High availability
├── tasks/               # Task management
└── tests/               # Test infrastructure
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
- [Architecture](../docs/content/architecture.html) - System overview
