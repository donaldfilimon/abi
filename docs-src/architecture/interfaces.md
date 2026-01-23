---
title: Interface Contracts
description: High-level interface boundaries, data shapes, and compile-time gating rules
---

# Interface Contracts

This document defines the interface boundaries for the ABI framework. These contracts serve as the foundation for modular refactoring while ensuring backward compatibility.

## Overview

The framework uses four key domain boundaries:
1. **Config** - Unified configuration with feature-specific sub-configs
2. **Tasks** - Task management with persistence, querying, and lifecycle
3. **Registry** - Feature registration with three registration modes
4. **AI/GPU Coupling** - Decoupled interfaces for GPU acceleration in AI workloads

## 1. Config Domain

### Public Surface

```zig
// src/config.zig

/// Main configuration struct - all features are optional
pub const Config = struct {
    gpu: ?GpuConfig = null,
    ai: ?AiConfig = null,
    database: ?DatabaseConfig = null,
    network: ?NetworkConfig = null,
    observability: ?ObservabilityConfig = null,
    web: ?WebConfig = null,
    plugins: PluginConfig = .{},

    pub fn defaults() Config;        // All compile-time enabled features
    pub fn minimal() Config;          // Empty config (all disabled)
    pub fn isEnabled(Feature) bool;   // Runtime feature check
    pub fn enabledFeatures(Allocator) ![]Feature;
};

/// Feature identifiers
pub const Feature = enum {
    gpu, ai, llm, embeddings, agents, training,
    database, network, observability, web,

    pub fn name() []const u8;
    pub fn description() []const u8;
    pub fn isCompileTimeEnabled() bool;  // Uses build_options
};

/// Fluent builder for Config
pub const Builder = struct {
    pub fn init(Allocator) Builder;
    pub fn withDefaults() *Builder;
    pub fn withGpu(GpuConfig) *Builder;
    pub fn withGpuDefaults() *Builder;
    pub fn withAi(AiConfig) *Builder;
    pub fn withAiDefaults() *Builder;
    pub fn withLlm(LlmConfig) *Builder;
    pub fn withDatabase(DatabaseConfig) *Builder;
    pub fn withDatabaseDefaults() *Builder;
    pub fn withNetwork(NetworkConfig) *Builder;
    pub fn withNetworkDefaults() *Builder;
    pub fn withObservability(ObservabilityConfig) *Builder;
    pub fn withObservabilityDefaults() *Builder;
    pub fn withWeb(WebConfig) *Builder;
    pub fn withWebDefaults() *Builder;
    pub fn withPlugins(PluginConfig) *Builder;
    pub fn build() Config;
};

/// Validation
pub fn validate(Config) ConfigError!void;
```

### Domain Sub-Configs

| Config | Nested Configs | Key Fields |
|--------|---------------|------------|
| `GpuConfig` | `RecoveryConfig` | backend, device_index, memory_limit, async_enabled |
| `AiConfig` | `LlmConfig`, `EmbeddingsConfig`, `AgentsConfig`, `TrainingConfig` | Sub-features are optional |
| `DatabaseConfig` | - | path, index_type, wal_enabled, cache_size |
| `NetworkConfig` | `UnifiedMemoryConfig`, `LinkingConfig` | bind_address, port, discovery_enabled, role |
| `ObservabilityConfig` | - | metrics_enabled, tracing_enabled, profiling_enabled |
| `WebConfig` | - | bind_address, port, cors_enabled, timeout_ms |
| `PluginConfig` | - | paths, auto_discover, load |

### Compile-Time Gating Rules

```zig
// defaults() respects compile-time flags
pub fn defaults() Config {
    return .{
        .gpu = if (build_options.enable_gpu) GpuConfig.defaults() else null,
        .ai = if (build_options.enable_ai) AiConfig.defaults() else null,
        // ... etc
    };
}

// validate() enforces compile-time constraints
pub fn validate(config: Config) ConfigError!void {
    if (config.gpu != null and !build_options.enable_gpu)
        return ConfigError.FeatureDisabled;
    // ... etc
}
```

### Domain Split Boundaries (Phase 2)

```
src/config.zig          # Shim: re-exports from config/*
src/config/
├── gpu.zig            # GpuConfig, RecoveryConfig
├── ai.zig             # AiConfig, LlmConfig, EmbeddingsConfig, etc.
├── database.zig       # DatabaseConfig
├── network.zig        # NetworkConfig, UnifiedMemoryConfig, LinkingConfig
├── observability.zig  # ObservabilityConfig
├── web.zig            # WebConfig
└── plugin.zig         # PluginConfig
```

---

## 2. Tasks Domain

### Public Surface

```zig
// src/tasks.zig

/// Core types
pub const Priority = enum(u8) { low, normal, high, critical };
pub const Status = enum(u8) { pending, in_progress, completed, cancelled, blocked };
pub const Category = enum(u8) { personal, roadmap, compute, bug, feature };
pub const SortBy = enum { created, updated, priority, due_date, status };

pub const Task = struct {
    id: u64,
    title: []const u8,
    description: ?[]const u8,
    status: Status,
    priority: Priority,
    category: Category,
    tags: []const []const u8,
    created_at: i64,
    updated_at: i64,
    due_date: ?i64,
    completed_at: ?i64,
    blocked_by: ?u64,
    parent_id: ?u64,

    pub fn isActionable() bool;
    pub fn isOverdue() bool;
};

pub const Filter = struct {
    status: ?Status,
    priority: ?Priority,
    category: ?Category,
    tag: ?[]const u8,
    overdue_only: bool,
    parent_id: ?u64,
    sort_by: SortBy,
    sort_descending: bool,
};

pub const Stats = struct {
    total: usize,
    pending: usize,
    in_progress: usize,
    completed: usize,
    cancelled: usize,
    blocked: usize,
    overdue: usize,
};

/// Manager configuration
pub const ManagerConfig = struct {
    storage_path: []const u8 = ".abi/tasks.json",
    auto_save: bool = true,
};

pub const AddOptions = struct {
    description: ?[]const u8,
    priority: Priority,
    category: Category,
    tags: []const []const u8,
    due_date: ?i64,
    parent_id: ?u64,
};

/// Task Manager
pub const Manager = struct {
    // Lifecycle
    pub fn init(Allocator, ManagerConfig) ManagerError!Manager;
    pub fn deinit() void;

    // CRUD operations (lifecycle)
    pub fn add(title: []const u8, AddOptions) ManagerError!u64;
    pub fn get(id: u64) ?Task;
    pub fn delete(id: u64) ManagerError!void;

    // Status operations (lifecycle)
    pub fn setStatus(id: u64, Status) ManagerError!void;
    pub fn complete(id: u64) ManagerError!void;
    pub fn start(id: u64) ManagerError!void;
    pub fn cancel(id: u64) ManagerError!void;

    // Property operations (lifecycle)
    pub fn setTitle(id: u64, []const u8) ManagerError!void;
    pub fn setDescription(id: u64, ?[]const u8) ManagerError!void;
    pub fn setPriority(id: u64, Priority) ManagerError!void;
    pub fn setCategory(id: u64, Category) ManagerError!void;
    pub fn setDueDate(id: u64, ?i64) ManagerError!void;
    pub fn setBlockedBy(id: u64, ?u64) ManagerError!void;

    // Querying
    pub fn list(Allocator, Filter) ManagerError![]Task;
    pub fn getStats() Stats;

    // Persistence
    pub fn save() ManagerError!void;
    pub fn load() ManagerError!void;
};

/// Roadmap integration
pub const roadmap = struct {
    pub const RoadmapItem = struct { ... };
    pub const incomplete_items: []const RoadmapItem;
    pub fn importAll(*Manager) !usize;
};
```

### Domain Split Boundaries (Phase 3)

```
src/tasks.zig           # Facade: routes to tasks/*
src/tasks/
├── types.zig          # Task, Priority, Status, Category, Filter, SortBy, Stats
├── persistence.zig    # save(), load(), JSON serialization
├── querying.zig       # list(), matchesFilter(), sorting
├── lifecycle.zig      # add, get, delete, setStatus, complete, start, etc.
└── roadmap.zig        # RoadmapItem, incomplete_items, importAll
```

---

## 3. Registry Domain

### Public Surface

```zig
// src/registry/mod.zig

pub const Feature = config_module.Feature;  // Re-exported

pub const RegistrationMode = enum {
    comptime_only,     // Zero overhead, resolved at compile time
    runtime_toggle,    // Compiled in, can be enabled/disabled at runtime
    dynamic,           // Loaded from shared libraries (future)
};

pub const FeatureRegistration = struct {
    feature: Feature,
    mode: RegistrationMode,
    context_ptr: ?*anyopaque,
    config_ptr: ?*const anyopaque,
    init_fn: ?*const fn(Allocator, *const anyopaque) anyerror!*anyopaque,
    deinit_fn: ?*const fn(*anyopaque) void,
    library_handle: ?*anyopaque,  // For dynamic mode
    library_path: ?[]const u8,
    enabled: bool,
    initialized: bool,
};

pub const Registry = struct {
    pub const Error = error{
        FeatureNotRegistered,
        FeatureAlreadyRegistered,
        FeatureNotCompiled,
        FeatureDisabled,
        InitializationFailed,
        AlreadyInitialized,
        NotInitialized,
        DynamicLoadingNotSupported,
        LibraryLoadFailed,
        SymbolNotFound,
        InvalidMode,
    } || Allocator.Error;

    // Lifecycle
    pub fn init(Allocator) Registry;
    pub fn deinit() void;

    // Registration API
    pub fn registerComptime(comptime Feature) Error!void;
    pub fn registerRuntimeToggle(comptime Feature, comptime ContextType, *const anyopaque) Error!void;
    pub fn registerDynamic(Feature, library_path: []const u8) Error!void;

    // Feature Lifecycle
    pub fn initFeature(Feature) Error!void;
    pub fn deinitFeature(Feature) Error!void;

    // Query API
    pub fn isRegistered(Feature) bool;
    pub fn isEnabled(Feature) bool;
    pub fn isInitialized(Feature) bool;
    pub fn getMode(Feature) ?RegistrationMode;
    pub fn getContext(Feature, comptime ContextType) Error!*ContextType;

    // Toggle API (runtime_toggle only)
    pub fn enableFeature(Feature) Error!void;
    pub fn disableFeature(Feature) Error!void;

    // Enumeration
    pub fn listFeatures(Allocator) Error![]Feature;
    pub fn count() usize;
};

// Compile-time helpers
pub fn isFeatureCompiledIn(comptime Feature) bool;
pub fn getParentFeature(Feature) ?Feature;
```

### Registration Mode Behavior

| Mode | Compile-time | Runtime Enable | Runtime Disable | Init Required |
|------|--------------|----------------|-----------------|---------------|
| `comptime_only` | Must be enabled | Always enabled | Cannot disable | No |
| `runtime_toggle` | Must be compiled | Yes | Yes | Yes |
| `dynamic` | N/A | Yes (after load) | Yes | Yes |

### Domain Split Boundaries (Phase 4) ✅ COMPLETE

```
src/registry/
├── mod.zig            # Public API facade with Registry struct
├── types.zig          # Core types: Feature, RegistrationMode, FeatureRegistration, Error
├── registration.zig   # registerComptime, registerRuntimeToggle, registerDynamic
├── lifecycle.zig      # initFeature, deinitFeature, enable/disable
└── plugins/           # Dynamic loading infrastructure (future)
    └── discovery.zig
```

**Implementation Notes:**
- `mod.zig` delegates to sub-modules while maintaining backward-compatible API
- `Registry.Error` alias provided for existing code using `Registry.Error.*`
- `types.zig` includes `isFeatureCompiledIn()` and `getParentFeature()` helpers

---

## 4. AI/GPU Interface

### Current Coupling

AI and GPU modules are currently loosely coupled:

```zig
// src/ai/mod.zig - Current pattern
pub const llm = if (build_options.enable_llm)
    @import("llm/mod.zig")
else
    @import("llm/stub.zig");

// LLM can optionally use GPU
pub const LlmConfig = struct {
    use_gpu: bool = true,  // Config option, not hard dependency
    // ...
};
```

```zig
// src/gpu/mod.zig - Current pattern
pub const Context = struct {
    pub fn init(Allocator, GpuConfig) !*Context;
    pub fn createBuffer(...) !UnifiedBuffer;
    pub fn vectorAdd(...) !ExecutionResult;
    pub fn matrixMultiply(...) !ExecutionResult;
};
```

### Decoupled Interface (Phase 5)

```zig
// src/ai/gpu_interface.zig - New abstraction layer

/// Lightweight GPU operations interface for AI workloads
pub const GpuOps = struct {
    pub const Error = error{
        GpuDisabled,
        AllocationFailed,
        ComputeFailed,
    };

    /// Check if GPU acceleration is available
    pub fn isAvailable() bool {
        return build_options.enable_gpu and gpu_module.isEnabled();
    }

    /// Matrix multiply with automatic GPU/CPU fallback
    pub fn matmul(a: []const f32, b: []const f32, result: []f32, m: usize, n: usize, k: usize) Error!void {
        if (comptime !build_options.enable_gpu) {
            return cpuMatmul(a, b, result, m, n, k);
        }
        // Try GPU, fallback to CPU on failure
    }

    /// Batch SIMD operations
    pub fn simdAdd(a: []const f32, b: []const f32, result: []f32) Error!void;
    pub fn simdMul(a: []const f32, b: []const f32, result: []f32) Error!void;
    pub fn softmax(input: []const f32, output: []f32) Error!void;
    pub fn layerNorm(input: []const f32, output: []f32, gamma: []const f32, beta: []const f32) Error!void;
};
```

```zig
// src/shared/gpu_ai_utils.zig - Shared utilities

/// Tensor operations used by both AI and GPU modules
pub const TensorOps = struct {
    pub fn reshape(data: []f32, old_shape: []const usize, new_shape: []const usize) !void;
    pub fn transpose(data: []f32, shape: []const usize, axes: []const usize) !void;
    pub fn slice(data: []const f32, shape: []const usize, starts: []const usize, ends: []const usize) ![]f32;
};

/// Memory layout utilities
pub const MemoryLayout = struct {
    pub const Order = enum { row_major, col_major };
    pub fn stride(shape: []const usize, order: Order) []usize;
    pub fn offset(indices: []const usize, shape: []const usize, order: Order) usize;
};
```

### Compile-Time Gating Assertions

```zig
// In AI modules that use GPU

comptime {
    // Ensure GPU operations go through the interface
    if (build_options.enable_gpu) {
        // Verify interface is available
        _ = @import("gpu_interface.zig").GpuOps;
    }
}

// Usage in LLM inference
pub fn forward(self: *Model, input: []const f32) ![]f32 {
    const gpu_ops = @import("gpu_interface.zig").GpuOps;

    if (self.config.use_gpu and gpu_ops.isAvailable()) {
        return try gpu_ops.matmul(input, self.weights, self.output, ...);
    } else {
        return try self.cpuForward(input);
    }
}
```

---

## 5. Stub Parity Contract

Every feature module must have a corresponding stub with identical public API:

### Pattern

```zig
// src/<feature>/mod.zig - Real implementation
pub const Context = struct {
    pub fn init(Allocator, Config) !*Context;
    pub fn deinit(*Context) void;
    pub fn doWork(*Context, input: []const u8) ![]u8;
};

pub fn isEnabled() bool {
    return build_options.enable_<feature>;
}

// src/<feature>/stub.zig - Stub implementation
pub const Context = struct {
    pub fn init(_: Allocator, _: Config) !*Context {
        return error.<Feature>Disabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn doWork(_: *Context, _: []const u8) ![]u8 {
        return error.<Feature>Disabled;
    }
};

pub fn isEnabled() bool {
    return false;
}
```

### Parity Rules

1. **Same public declarations**: All `pub` types, functions, and constants must exist in both
2. **Same function signatures**: Parameters, return types, and error sets must match
3. **Stub behavior**: Return appropriate `error.*Disabled` or no-op
4. **Compile-time safety**: Stub must compile independently

### Automated Parity Check (Phase 6) ✅ COMPLETE

Implemented in `src/tests/stub_parity.zig`:

```zig
// Verify modules follow Context + isEnabled pattern
fn verifyContextPattern(comptime Module: type) !void {
    try testing.expect(@hasDecl(Module, "Context"));
    try testing.expect(@hasDecl(Module, "isEnabled"));
}

test "all feature modules have consistent API surface" {
    try verifyContextPattern(abi.database);
    try verifyContextPattern(abi.gpu);
    try verifyContextPattern(abi.network);
    try verifyContextPattern(abi.web);
    try verifyContextPattern(abi.observability);
    if (build_options.enable_ai) {
        try verifyContextPattern(abi.ai);
    }
}
```

**Test Coverage:**
- Database: Context, SearchResult, DatabaseHandle, sub-modules (wdbx, fulltext, hybrid, filter, batch, clustering, formats)
- GPU: Context, isEnabled
- Network: Context, isEnabled
- Web: Context, isEnabled
- Observability: Context, isEnabled
- AI: Context, isEnabled (when enabled)
- AI sub-modules: llm, agents, embeddings, training (when enabled)

---

## 6. Error Contracts

### Standard Error Sets

```zig
// Config errors
pub const ConfigError = error{
    FeatureDisabled,
    InvalidConfig,
    MissingRequired,
    ConflictingConfig,
};

// Task errors
pub const ManagerError = error{
    TaskNotFound,
    InvalidOperation,
    PersistenceFailed,
    ParseError,
} || Allocator.Error || Io.Error;

// Registry errors
pub const RegistryError = error{
    FeatureNotRegistered,
    FeatureAlreadyRegistered,
    FeatureNotCompiled,
    FeatureDisabled,
    InitializationFailed,
    AlreadyInitialized,
    NotInitialized,
    DynamicLoadingNotSupported,
    InvalidMode,
} || Allocator.Error;

// GPU interface errors
pub const GpuInterfaceError = error{
    GpuDisabled,
    AllocationFailed,
    ComputeFailed,
    KernelFailed,
    SynchronizationFailed,
};
```

---

## Summary

| Domain | Entry Point | Key Types | Split Target |
|--------|-------------|-----------|--------------|
| Config | `src/config.zig` | `Config`, `Feature`, `Builder` | `src/config/*.zig` |
| Tasks | `src/tasks.zig` | `Manager`, `Task`, `Filter` | `src/tasks/*.zig` |
| Registry | `src/registry/mod.zig` | `Registry`, `RegistrationMode` | `src/registry/*.zig` |
| AI/GPU | `src/ai/gpu_interface.zig` | `GpuOps` | New file + `shared/gpu_ai_utils.zig` |

These interfaces form the contract for Phases 2-9. Changes to public surfaces require updating this document and ensuring backward compatibility.
