---
title: "overview"
tags: []
---
# ABI Framework Architecture Overview
> **Codebase Status:** Synced with repository as of 2026-01-23.

This document provides a comprehensive overview of the ABI framework architecture after the 2026.01 migration.

## Directory Structure

```
src/
├── abi.zig              # Public API entry point: init(), shutdown(), version()
├── config.zig           # Unified configuration system
├── framework.zig        # Framework orchestration with builder pattern
│
├── runtime/             # Always-on infrastructure (task execution, scheduling)
│   └── mod.zig         # Runtime Context for Framework integration
│
├── gpu/                 # GPU acceleration [FULLY MIGRATED]
│   ├── mod.zig         # Module entry, exports unified API
│   ├── unified.zig     # Gpu struct, GpuConfig, high-level ops
│   ├── dsl/            # Kernel DSL compiler (builder, codegen, optimizer)
│   ├── backends/       # Backend implementations (cuda/, vulkan, metal, etc.)
│   ├── diagnostics.zig # GPU state debugging
│   ├── error_handling.zig # Structured error context
│   └── failover.zig    # Graceful degradation to CPU
│
├── database/           # Vector database [FULLY MIGRATED]
│   ├── mod.zig        # Module entry with Context struct
│   ├── stub.zig       # Feature-gated placeholder
│   ├── database.zig   # Core VectorDatabase implementation
│   ├── hnsw.zig       # HNSW indexing
│   ├── formats/       # File format handlers (8 files)
│   └── ...            # 24+ implementation files
│
├── network/            # Distributed compute [FULLY MIGRATED]
│   ├── mod.zig        # Module entry with Context struct
│   └── stub.zig       # Feature-gated placeholder
│
├── web/                # Web/HTTP utilities [FULLY MIGRATED]
│   ├── mod.zig        # Module entry with Context struct
│   ├── stub.zig       # Feature-gated placeholder
│   ├── client.zig     # HTTP client implementation
│   └── weather.zig    # Weather API client
│
├── ai/                 # AI module (thin wrappers)
│   ├── mod.zig        # Module entry with Context struct
│   ├── agents/        # Agent system wrapper
│   ├── embeddings/    # Embeddings wrapper
│   ├── llm/           # LLM inference wrapper
│   └── training/      # Training pipeline wrapper
│
├── observability/      # Metrics, tracing, profiling
│   ├── mod.zig        # Module entry with Context struct
│   └── stub.zig       # Feature-gated placeholder
│
├── shared/             # Cross-cutting utilities
│   ├── mod.zig        # Logging, plugins, platform exports
│   ├── simd.zig       # SIMD vector operations
│   ├── observability/ # Metrics primitives, tracing types (Tracer, Span)
│   └── utils/         # Memory, time, backoff utilities
│
├── compute/            # Legacy compute infrastructure
│   ├── mod.zig        # Concurrency primitives
│   └── runtime/       # Future, TaskGroup, CancellationToken
│
├── features/           # Implementation layer
│   ├── mod.zig        # Feature references
│   ├── ai/            # Full AI implementation (agent, training, embeddings, llm)
│   ├── connectors/    # API connectors (OpenAI, Ollama, Anthropic)
│   └── ha/            # High availability (backup, PITR, replication)
│
└── registry/           # Feature registry system [MODULARIZED]
    ├── mod.zig        # Public API facade with Registry struct
    ├── types.zig      # Core types (Feature, RegistrationMode, Error)
    ├── registration.zig # registerComptime, registerRuntimeToggle, registerDynamic
    └── lifecycle.zig  # initFeature, deinitFeature, enable/disable
```

## Design Patterns

### 1. Feature Gating Pattern

Compile-time feature selection via `build_options.enable_*`. Disabled features use stub modules:

```zig
const impl = if (build_options.enable_feature)
    @import("real.zig")
else
    @import("stub.zig");
```

**Stub Requirements:**
- Must mirror complete API (structs, functions, constants)
- Always return `error.<Feature>Disabled` for operations
- Include Context struct matching real module
- Include `isEnabled()` function returning `false`

**Stub Parity Testing:**
Automated parity verification in `src/tests/stub_parity.zig` ensures stubs stay synchronized:
```zig
// Verify all feature modules follow Context + isEnabled pattern
fn verifyContextPattern(comptime Module: type) !void {
    try testing.expect(@hasDecl(Module, "Context"));
    try testing.expect(@hasDecl(Module, "isEnabled"));
}
```

### 2. Context Pattern (Framework Integration)

Every feature module exposes a `Context` struct for Framework integration:

```zig
pub const Context = struct {
    allocator: Allocator,
    // feature-specific state...

    pub fn init(allocator: Allocator, config: ?Config) !Context {
        // Initialize feature
    }

    pub fn deinit(self: *Context) void {
        // Cleanup
    }
};
```

### 3. Wrapper Pattern

Thin wrapper modules in `src/` delegate to implementations in `src/features/`:

```zig
// src/ai/mod.zig (wrapper)
const impl = @import("../features/ai/mod.zig");
pub const Agent = impl.Agent;
```

This pattern allows gradual migration while maintaining stable imports.

### 4. Builder Pattern (Framework)

Fluent API for framework configuration:

```zig
var fw = try abi.Framework.builder(allocator)
    .withGpu(.{ .backend = .vulkan })
    .withDatabase(.{ .path = "./data" })
    .build();
```

### 5. Registry Pattern (Feature Management)

Three registration modes for different use cases:

```zig
const registry = @import("registry/mod.zig");

var reg = registry.Registry.init(allocator);
defer reg.deinit();

// Comptime-only: Zero overhead, resolved at compile time
try reg.registerComptime(.gpu);

// Runtime-toggle: Compiled in but can be enabled/disabled
try reg.registerRuntimeToggle(.ai, ai_mod.Context, &ai_config);
try reg.enableFeature(.ai);
try reg.initFeature(.ai);

// Query features
if (reg.isEnabled(.gpu)) {
    // Use GPU...
}
```

| Mode | Overhead | Runtime Toggle | Use Case |
|------|----------|----------------|----------|
| `comptime_only` | Zero | No | Static features |
| `runtime_toggle` | Minimal | Yes | Optional features |
| `dynamic` | Moderate | Yes | Plugin loading (future) |

## Module Responsibilities

| Module | Responsibility | Status |
|--------|---------------|--------|
| `runtime` | Task execution, scheduling | Migrated |
| `gpu` | GPU acceleration, kernel DSL | Migrated |
| `database` | Vector storage, HNSW indexing | Migrated |
| `network` | Distributed compute, RPC | Migrated |
| `web` | HTTP client, weather API | Migrated |
| `ai` | LLM, agents, training | Wrapper (impl in features/) |
| `observability` | Metrics, tracing | Migrated |
| `compute` | Concurrency primitives | Legacy |

## Import Guidance

**For application code:**
```zig
const abi = @import("abi");
var fw = try abi.init(allocator);
```

**For internal module access:**
```zig
// GPU (fully migrated)
const gpu = @import("src/gpu/mod.zig");

// Database (fully migrated)
const database = @import("src/database/mod.zig");

// Network (fully migrated)
const network = @import("src/network/mod.zig");

// Web (fully migrated)
const web = @import("src/web/mod.zig");

// AI (use wrapper, impl in features/)
const ai = @import("src/ai/mod.zig");
```

## Configuration Flow

```
User Code
    ↓
abi.init(allocator, config)
    ↓
Framework.init()
    ↓
┌─────────────────────────────────────┐
│  For each enabled feature:         │
│    1. Check build_options.enable_* │
│    2. Load real or stub module     │
│    3. Call Context.init()          │
└─────────────────────────────────────┘
    ↓
Framework ready for use
```

## Migration Status

### Complete (2026.01)
- GPU module: 74 files in `src/gpu/`
- Database module: 32 files in `src/database/`
- Web module: 5 files in `src/web/`
- Network module: Context struct added
- Runtime module: Context struct added
- Framework orchestration working

### Remaining Work
- AI module: Full implementation still in `src/features/ai/`
- Compute module: Legacy code, may be consolidated

## Testing

```bash
# Default build (all features)
zig build

# Feature-disabled build (tests stubs)
zig build -Denable-ai=false -Denable-gpu=false -Denable-database=false \
          -Denable-network=false -Denable-web=false

# Run all tests (includes stub parity verification)
zig build test --summary all

# Build all examples
zig build examples
```

### Test Categories

| Category | File | Description |
|----------|------|-------------|
| Framework | `src/tests/mod.zig` | Framework init, feature flags |
| Stub Parity | `src/tests/stub_parity.zig` | Verify stub/real API match |
| Registry | `src/registry/mod.zig` | Registration and lifecycle tests |
| Integration | `src/tests/test_matrix.zig` | Cross-module integration |

## Related Documentation

- [Framework Guide](../framework.md) - Detailed framework usage
- [Feature Flags](../feature-flags.md) - Build configuration
- [GPU Guide](../gpu.md) - GPU programming
- [Database Guide](../database.md) - Vector database operations
- [Troubleshooting](../troubleshooting.md) - Common issues

