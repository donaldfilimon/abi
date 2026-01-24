# framework API Reference

> Framework orchestration and lifecycle management

**Source:** [`src/framework.zig`](../../src/framework.zig)

---

Framework Orchestration Layer

This module provides the central orchestration for the ABI framework, managing
the lifecycle of all feature modules, coordinating initialization and shutdown,
and maintaining runtime state.

## Overview

The `Framework` struct is the primary entry point for using ABI. It:

- Initializes and manages feature contexts (GPU, AI, Database, etc.)
- Maintains a feature registry for runtime feature management
- Provides typed access to enabled features
- Handles graceful shutdown and resource cleanup

## Initialization Patterns

### Default Initialization

```zig
const abi = @import("abi");

var fw = try abi.Framework.initDefault(allocator);
defer fw.deinit();

// All compile-time enabled features are now available
```

### Custom Configuration

```zig
var fw = try abi.Framework.init(allocator, .{
.gpu = .{ .backend = .vulkan },
.ai = .{ .llm = .{ .model_path = "./model.gguf" } },
.database = .{ .path = "./data" },
});
defer fw.deinit();
```

### Builder Pattern

```zig
var fw = try abi.Framework.builder(allocator)
.withGpu(.{ .backend = .vulkan })
.withAiDefaults()
.withDatabaseDefaults()
.build();
defer fw.deinit();
```

## Feature Access

```zig
// Check if a feature is enabled
if (fw.isEnabled(.gpu)) {
// Get the feature context
const gpu_ctx = try fw.getGpu();
// Use GPU features...
}

// Runtime context is always available
const runtime = fw.getRuntime();
```

## State Management

The framework transitions through the following states:
- `uninitialized`: Initial state before `init()`
- `initializing`: During feature initialization
- `running`: Normal operation state
- `stopping`: During shutdown
- `stopped`: After `deinit()` completes
- `failed`: If initialization fails

---

## API

### `pub const Framework`

<sup>**type**</sup>

Framework orchestration handle.

The Framework struct is the central coordinator for the ABI framework. It manages
the lifecycle of all enabled feature modules, provides access to their contexts,
and maintains the framework's runtime state.

## Thread Safety

The Framework itself is not thread-safe. If you need to access the framework from
multiple threads, you should use external synchronization or ensure each thread
has its own Framework instance.

## Memory Management

The Framework allocates memory for feature contexts during initialization. All
allocated memory is released when `deinit()` is called. The caller must ensure
the provided allocator remains valid for the lifetime of the Framework.

## Example

```zig
var fw = try Framework.init(allocator, Config.defaults());
defer fw.deinit();

// Check state
if (fw.isRunning()) {
// Access features
if (fw.gpu) |gpu_ctx| {
// Use GPU...
}
}
```

### `pub const State`

<sup>**type**</sup>

Framework lifecycle states.

### `pub fn init(allocator: std.mem.Allocator, cfg: Config) Error!Framework`

<sup>**fn**</sup>

Initialize the framework with the given configuration.

This is the primary initialization method for the Framework. It validates the
configuration, initializes all enabled feature modules, and transitions the
framework to the `running` state.

## Parameters

- `allocator`: Memory allocator for framework resources. Must remain valid for
the lifetime of the Framework.
- `cfg`: Configuration specifying which features to enable and their settings.

## Returns

A fully initialized Framework instance in the `running` state.

## Errors

- `ConfigError.FeatureDisabled`: A feature is enabled in config but disabled at compile time
- `error.OutOfMemory`: Memory allocation failed
- `error.FeatureInitFailed`: A feature module failed to initialize

## Example

```zig
var fw = try Framework.init(allocator, .{
.gpu = .{ .backend = .vulkan },
.database = .{ .path = "./data" },
});
defer fw.deinit();
```

### `pub fn initWithIo(allocator: std.mem.Allocator, cfg: Config, io: std.Io) Error!Framework`

<sup>**fn**</sup>

Initialize the framework with the given configuration **and** an I/O backend.
This method is used by the builder when `withIo` is supplied.

### `pub fn initDefault(allocator: std.mem.Allocator) Error!Framework`

<sup>**fn**</sup>

Create a framework with default configuration.

This is a convenience method that creates a framework with all compile-time
enabled features also enabled at runtime with their default settings.

## Parameters

- `allocator`: Memory allocator for framework resources

## Returns

A Framework instance with default configuration.

## Example

```zig
var fw = try Framework.initDefault(allocator);
defer fw.deinit();
```

### `pub fn initMinimal(allocator: std.mem.Allocator) Error!Framework`

<sup>**fn**</sup>

Create a framework with minimal configuration (no features enabled).

This creates a framework with no optional features enabled. Only the
runtime context is initialized. Useful for testing or when you want
to explicitly enable specific features.

## Parameters

- `allocator`: Memory allocator for framework resources

## Returns

A Framework instance with minimal configuration.

## Example

```zig
var fw = try Framework.initMinimal(allocator);
defer fw.deinit();

// Only runtime is available, no features enabled
try std.testing.expect(fw.gpu == null);
try std.testing.expect(fw.ai == null);
```

### `pub fn builder(allocator: std.mem.Allocator) FrameworkBuilder`

<sup>**fn**</sup>

Start building a framework configuration.

Returns a FrameworkBuilder that provides a fluent API for configuring
and initializing the framework.

## Parameters

- `allocator`: Memory allocator for framework resources

## Returns

A FrameworkBuilder instance for configuring the framework.

## Example

```zig
var fw = try Framework.builder(allocator)
.withGpuDefaults()
.withAi(.{ .llm = .{} })
.build();
defer fw.deinit();
```

### `pub fn deinit(self: *Framework) void`

<sup>**fn**</sup>

Shutdown and cleanup the framework.

This method transitions the framework to the `stopping` state, deinitializes
all feature contexts in reverse order of initialization, cleans up the registry,
and finally transitions to `stopped`.

After calling `deinit()`, the framework instance should not be used. Any
pointers to feature contexts become invalid.

This method is idempotent - calling it multiple times is safe.

## Example

```zig
var fw = try Framework.initDefault(allocator);
// ... use framework ...
fw.deinit();  // Clean up all resources
```

### `pub fn isRunning(self: *const Framework) bool`

<sup>**fn**</sup>

Check if the framework is running.

### `pub fn isEnabled(self: *const Framework, feature: Feature) bool`

<sup>**fn**</sup>

Check if a feature is enabled.

### `pub fn getState(self: *const Framework) State`

<sup>**fn**</sup>

Get the current framework state.

### `pub fn getGpu(self: *Framework) Error!*gpu_mod.Context`

<sup>**fn**</sup>

Get GPU context (returns error if not enabled).

### `pub fn getAi(self: *Framework) Error!*ai_mod.Context`

<sup>**fn**</sup>

Get AI context (returns error if not enabled).

### `pub fn getDatabase(self: *Framework) Error!*database_mod.Context`

<sup>**fn**</sup>

Get database context (returns error if not enabled).

### `pub fn getNetwork(self: *Framework) Error!*network_mod.Context`

<sup>**fn**</sup>

Get network context (returns error if not enabled).

### `pub fn getObservability(self: *Framework) Error!*observability_mod.Context`

<sup>**fn**</sup>

Get observability context (returns error if not enabled).

### `pub fn getWeb(self: *Framework) Error!*web_mod.Context`

<sup>**fn**</sup>

Get web context (returns error if not enabled).

### `pub fn getRuntime(self: *Framework) *runtime_mod.Context`

<sup>**fn**</sup>

Get runtime context (always available).

### `pub fn getRegistry(self: *Framework) *Registry`

<sup>**fn**</sup>

Get the feature registry for runtime feature management.

### `pub fn isFeatureRegistered(self: *const Framework, feature: Feature) bool`

<sup>**fn**</sup>

Check if a feature is registered in the registry.

### `pub fn listRegisteredFeatures(self: *const Framework, allocator: std.mem.Allocator) RegistryError![]Feature`

<sup>**fn**</sup>

List all registered features.

### `pub const FrameworkBuilder`

<sup>**type**</sup>

Fluent builder for Framework initialization.

### `pub fn withDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Start with default configuration.

### `pub fn withGpu(self: *FrameworkBuilder, gpu_config: config_module.GpuConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable GPU with configuration.

### `pub fn withGpuDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Enable GPU with defaults.

### `pub fn withIo(self: *FrameworkBuilder, io: std.Io) *FrameworkBuilder`

<sup>**fn**</sup>

Provide a shared I/O backend for the framework.
Pass the `std.Io` obtained from `IoBackend.init`.

### `pub fn withAi(self: *FrameworkBuilder, ai_config: config_module.AiConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable AI with configuration.

### `pub fn withAiDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Enable AI with defaults.

### `pub fn withLlm(self: *FrameworkBuilder, llm_config: config_module.LlmConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable LLM only.

### `pub fn withDatabase(self: *FrameworkBuilder, db_config: config_module.DatabaseConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable database with configuration.

### `pub fn withDatabaseDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Enable database with defaults.

### `pub fn withNetwork(self: *FrameworkBuilder, net_config: config_module.NetworkConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable network with configuration.

### `pub fn withNetworkDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Enable network with defaults.

### `pub fn withObservability(self: *FrameworkBuilder, obs_config: config_module.ObservabilityConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable observability with configuration.

### `pub fn withObservabilityDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Enable observability with defaults.

### `pub fn withWeb(self: *FrameworkBuilder, web_config: config_module.WebConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable web with configuration.

### `pub fn withWebDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Enable web with defaults.

### `pub fn withPlugins(self: *FrameworkBuilder, plugin_config: config_module.PluginConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Configure plugins.

### `pub fn build(self: *FrameworkBuilder) Framework.Error!Framework`

<sup>**fn**</sup>

Build and initialize the framework.
If an I/O backend was supplied via `withIo`, it will be stored in the
resulting `Framework` instance.

### `pub const FrameworkOptions`

<sup>**type**</sup>

Legacy FrameworkOptions for backward compatibility.
@deprecated Use Config directly.

### `pub fn toConfig(self: FrameworkOptions) Config`

<sup>**fn**</sup>

Convert to new Config format.

### `pub const FrameworkConfiguration`

<sup>**const**</sup>

Legacy FrameworkConfiguration for backward compatibility.
@deprecated Use Config directly.

### `pub const RuntimeConfig`

<sup>**const**</sup>

Legacy RuntimeConfig for backward compatibility.
@deprecated Use Config directly.

### `pub fn runtimeConfigFromOptions(allocator: std.mem.Allocator, options: FrameworkOptions) !Config`

<sup>**fn**</sup>

Convert legacy options to new config.
@deprecated Use Config directly.

---

*Generated automatically by `zig build gendocs`*
