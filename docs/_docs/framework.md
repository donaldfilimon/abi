---
title: Framework Lifecycle
description: Deep dive into Framework initialization, state machine, builder pattern, and feature registry
section: Core
order: 3
---

# Framework Lifecycle

The `Framework` struct (`src/core/framework.zig`) is the central coordinator for ABI.
It manages the lifecycle of all enabled feature modules, provides typed access to their
contexts, and enforces a strict state machine for initialization and shutdown.

## State Machine

The framework transitions through six states:

```
uninitialized --> initializing --> running --> stopping --> stopped
                      |                                      ^
                      +---------> failed --------------------+
```

| State | Description | Transitions To |
|-------|-------------|----------------|
| `uninitialized` | Before any `init` call. Default state for a zero-initialized `Framework`. | `initializing` |
| `initializing` | Features are being started in dependency order. Config is validated, the registry is populated, and each feature context is allocated. | `running`, `failed` |
| `running` | Normal operation. All requested features are available through getter methods. | `stopping` |
| `stopping` | Graceful shutdown in progress. Feature contexts are deinitialized in reverse order. | `stopped` |
| `stopped` | All resources released. The `Framework` instance should not be used after this. | -- |
| `failed` | Initialization encountered an error. Calling `deinit()` transitions to `stopped`. | `stopped` |

The state is stored as a `Framework.State` enum and can be queried at any time:

```zig
const state = fw.getState();
if (fw.isRunning()) {
    // Safe to access feature contexts
}
```

## Initialization Patterns

ABI provides four initialization methods, from simplest to most flexible.

### 1. Default Initialization

Enables all compile-time features with their default configurations. This is the
quickest way to get a fully-featured framework instance.

```zig
const abi = @import("abi");

var fw = try abi.initDefault(allocator);
defer fw.deinit();

// All compile-time enabled features are now available
if (fw.isEnabled(.gpu)) {
    const gpu_ctx = try fw.getGpu();
    _ = gpu_ctx;
}
```

`abi.initDefault(allocator)` is a convenience wrapper around `Framework.initDefault(allocator)`.

### 2. Custom Configuration (Struct Literal)

Pass a `Config` struct literal to select specific features and tune their settings.
Only the fields you set are enabled; everything else stays at its default (or null,
meaning "not explicitly configured").

```zig
var fw = try abi.Framework.init(allocator, .{
    .gpu = .{ .backend = .vulkan },
    .ai = .{ .llm = .{ .model_path = "./model.gguf" } },
    .database = .{ .path = "./data" },
});
defer fw.deinit();
```

### 3. Builder Pattern

The builder provides a fluent API that is especially useful when configuration is
assembled incrementally or conditionally:

```zig
var builder_inst = abi.Framework.builder(allocator);

// Always enable GPU
_ = builder_inst.withGpu(.{ .backend = .metal });

// Conditionally enable AI
if (enable_ai) {
    _ = builder_inst.withAiDefaults();
}

// Provide a shared I/O backend for file/network operations
_ = builder_inst.withIo(io);

// Enable more features
_ = builder_inst.withDatabaseDefaults();
_ = builder_inst.withCache(.{
    .max_entries = 50_000,
    .eviction_policy = .lru,
});

var fw = try builder_inst.build();
defer fw.deinit();
```

Every feature has both a typed method (`.withGpu(config)`) and a defaults method
(`.withGpuDefaults()`). The full list of builder methods:

| Method | Defaults Variant | Feature |
|--------|-----------------|---------|
| `.withGpu(GpuConfig)` | `.withGpuDefaults()` | GPU |
| `.withAi(AiConfig)` | `.withAiDefaults()` | AI |
| `.withLlm(LlmConfig)` | -- | LLM only (subset of AI) |
| `.withDatabase(DatabaseConfig)` | `.withDatabaseDefaults()` | Database |
| `.withNetwork(NetworkConfig)` | `.withNetworkDefaults()` | Network |
| `.withObservability(ObservabilityConfig)` | `.withObservabilityDefaults()` | Observability |
| `.withWeb(WebConfig)` | `.withWebDefaults()` | Web |
| `.withCloud(CloudConfig)` | `.withCloudDefaults()` | Cloud |
| `.withAnalytics(AnalyticsConfig)` | `.withAnalyticsDefaults()` | Analytics |
| `.withAuth(AuthConfig)` | `.withAuthDefaults()` | Auth |
| `.withMessaging(MessagingConfig)` | `.withMessagingDefaults()` | Messaging |
| `.withCache(CacheConfig)` | `.withCacheDefaults()` | Cache |
| `.withStorage(StorageConfig)` | `.withStorageDefaults()` | Storage |
| `.withSearch(SearchConfig)` | `.withSearchDefaults()` | Search |
| `.withGateway(GatewayConfig)` | `.withGatewayDefaults()` | Gateway |
| `.withPages(PagesConfig)` | `.withPagesDefaults()` | Pages |
| `.withBenchmarks(BenchmarksConfig)` | `.withBenchmarksDefaults()` | Benchmarks |
| `.withMobile(MobileConfig)` | `.withMobileDefaults()` | Mobile |
| `.withPlugins(PluginConfig)` | -- | Plugin system |
| `.withIo(std.Io)` | -- | Shared I/O backend |
| `.withDefaults()` | -- | All defaults at once |

Call `.build()` to finalize. This validates the configuration and returns a `Framework`
in the `running` state (or returns an error).

### 4. Minimal Initialization

Creates a framework with no optional features enabled. Only the runtime context is
initialized. Useful for testing or when you want full control over what is active.

```zig
var fw = try abi.Framework.initMinimal(allocator);
defer fw.deinit();

// Only runtime is available
const runtime = fw.getRuntime();

// All feature getters return null
std.debug.assert(fw.gpu == null);
std.debug.assert(fw.ai == null);
```

## The Framework Struct

The `Framework` struct holds:

- **`allocator`** -- the memory allocator passed at init, used for all framework allocations.
- **`io`** -- an optional shared I/O backend (`?std.Io`) for file and network access across features.
- **`config`** -- the `Config` struct used to initialize this instance.
- **`state`** -- the current lifecycle `State` enum value.
- **`registry`** -- a `Registry` for runtime feature management and introspection.
- **`runtime`** -- a pointer to the always-available `runtime_mod.Context`.
- **21 feature context slots** -- each is `?*FeatureModule.Context`, set to `null` if the feature is disabled.

## Feature Registry

The `Registry` (`src/core/registry/`) tracks which features are registered at runtime.
It supports three registration modes:

| Mode | Description | Overhead |
|------|-------------|----------|
| `comptime_only` | Features resolved at compile time only. Cannot be toggled at runtime. | Zero |
| `runtime_toggle` | Compiled in but can be enabled/disabled at runtime. | Small (state check) |
| `dynamic` | Loaded from shared libraries at runtime (future). | Highest |

Registry operations:

```zig
// Check if a feature is registered
if (fw.isFeatureRegistered(.gpu)) {
    // ...
}

// List all registered features
const features = try fw.listRegisteredFeatures(allocator);
defer allocator.free(features);

// Get the raw registry for advanced operations
const reg = fw.getRegistry();
```

Compile-time checking is also available:

```zig
const types = @import("core/registry/types.zig");

// Check if a feature was compiled in via build_options
const gpu_compiled = comptime types.isFeatureCompiledIn(.gpu);
```

## Feature Access

Each feature has a typed getter that returns a pointer to the feature's `Context`,
or `error.FeatureDisabled` if that feature was not initialized:

```zig
// Fallible access (returns error if disabled)
const gpu_ctx = try fw.getGpu();
const ai_ctx = try fw.getAi();
const db_ctx = try fw.getDatabase();
const net_ctx = try fw.getNetwork();
const obs_ctx = try fw.getObservability();
const cache_ctx = try fw.getCache();

// Check-then-access pattern
if (fw.isEnabled(.gpu)) {
    const ctx = try fw.getGpu();
    _ = ctx;
}

// Direct field access (null check)
if (fw.gpu) |gpu_ctx| {
    _ = gpu_ctx;
}

// Runtime is always available (never null)
const runtime = fw.getRuntime();
_ = runtime;
```

The complete list of getters: `getGpu`, `getAi`, `getDatabase`, `getNetwork`,
`getObservability`, `getWeb`, `getCloud`, `getAnalytics`, `getAuth`, `getMessaging`,
`getCache`, `getStorage`, `getSearch`, `getGateway`, `getPages`, `getBenchmarks`,
`getMobile`, `getAiCore`, `getAiInference`, `getAiTraining`, `getAiReasoning`.

## Shutdown and Cleanup

Calling `fw.deinit()` triggers an orderly shutdown:

1. State transitions to `stopping`.
2. Feature contexts are deinitialized in **reverse** order of initialization.
3. The registry is cleaned up.
4. The runtime context is released.
5. State transitions to `stopped`.

`deinit()` is idempotent -- calling it multiple times is safe.

```zig
var fw = try abi.initDefault(allocator);
// ... use framework ...
fw.deinit();  // Clean up all resources
```

For time-bounded shutdown (future async support):

```zig
const clean = fw.shutdownWithTimeout(5000); // 5 second timeout
if (!clean) {
    // Timeout expired -- some resources may not be fully cleaned up
}
```

## I/O Backend Integration

Features that need file or network access use a shared I/O backend. Pass it through
the builder:

```zig
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty,
});
defer io_backend.deinit();
const io = io_backend.io();

var fw = try abi.Framework.builder(allocator)
    .withIo(io)
    .withDatabaseDefaults()
    .withStorageDefaults()
    .build();
defer fw.deinit();
```

The I/O backend is stored in `fw.io` and can be retrieved by sub-systems as needed.

## Error Handling

Framework initialization can fail with errors from the `Framework.Error` set
(defined in `src/core/errors.zig`):

| Error | Cause |
|-------|-------|
| `ConfigError.FeatureDisabled` | Config enables a feature that is disabled at compile time |
| `error.OutOfMemory` | Memory allocation failed during context creation |
| `error.FeatureInitFailed` | A feature module's `init()` returned an error |
| `error.FeatureDisabled` | Calling a getter for a feature that is not initialized |

The recommended pattern is to use `errdefer` for cleanup in fallible init paths:

```zig
var fw = abi.Framework.init(allocator, config) catch |err| {
    std.log.err("Framework init failed: {t}", .{err});
    return err;
};
defer fw.deinit();
```

## Further Reading

- [Architecture](architecture.html) -- module hierarchy and comptime gating
- [Configuration](configuration.html) -- all build flags, environment variables, and config types
- [CLI](cli.html) -- `config init`, `config show`, `config validate` commands

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
