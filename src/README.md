---
title: "Source Directory"
tags: [source, architecture, modules]
---
# Source Directory
> **Codebase Status:** Synced with repository as of 2026-02-04.

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
| `api/` | Entry points (`main.zig`) |
| `core/` | Framework orchestration, config, flags, registry |
| `features/` | Feature modules (ai, gpu, database, network, observability, web) |
| `services/` | Shared infrastructure (runtime, platform, shared, connectors, cloud, ha, tasks, tests) |

## Module Hierarchy

```
src/
├── abi.zig              # Public API module root
├── api/                 # Entry points
│   └── main.zig         # CLI entrypoint fallback
│
├── core/                # Framework orchestration and config
│   ├── config/          # Unified configuration
│   ├── framework.zig    # Framework lifecycle
│   └── registry/        # Feature registry system
│
├── features/            # Feature modules
│   ├── ai/              # AI module (agents, llm, training, personas)
│   ├── analytics/       # Event tracking and experiments
│   ├── cloud/           # Cloud function adapters (AWS, GCP, Azure)
│   ├── database/        # Vector database
│   ├── gpu/             # GPU acceleration
│   ├── network/         # Distributed compute
│   ├── observability/   # Metrics and tracing
│   └── web/             # Web/HTTP utilities
│
└── services/            # Shared infrastructure
    ├── runtime/         # Scheduling, concurrency, memory
    ├── platform/        # Platform detection and SIMD capabilities
    ├── shared/          # Logging, io, utils, security
    ├── connectors/      # External API connectors
    ├── ha/              # High availability
    ├── tasks/           # Task management
    └── tests/           # Test infrastructure
```

## Key Entry Points

- **Public API**: `abi.zig` - `abi.initDefault()`, `Framework.builder()`, `Framework.deinit()`, `abi.version()`
- **Configuration**: `core/config/mod.zig` - Unified `Config` struct with `Builder` API
- **Framework**: `core/framework.zig` - `Framework` struct manages feature lifecycle
- **Runtime**: `services/runtime/mod.zig` - Always-available scheduling and concurrency

## Module Pattern

Top-level modules (gpu, ai, database, network, observability, web) follow this pattern:

1. Re-export types and functions from implementation directories
2. Add a `Context` struct for Framework integration
3. Provide `isEnabled()` function for compile-time feature detection
4. Have a corresponding `stub.zig` for when the feature is disabled

```zig
// Example: src/features/gpu/mod.zig
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
- [API Reference](../docs/api-reference.md)
- [Docs Map](../docs/README.md) - Documentation layout and entry points
- [docs/README.md](../docs/README.md) - Documentation site source
