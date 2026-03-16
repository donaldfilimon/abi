---
title: app API
purpose: Generated API reference for app
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# app

> Framework Orchestration Layer

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

var fw = try abi.App.initDefault(allocator);
defer fw.deinit();

// All compile-time enabled features are now available
```

### Custom Configuration

```zig
var fw = try abi.App.init(allocator, .{
.gpu = .{ .backend = .vulkan },
.ai = .{ .llm = .{ .model_path = "./model.gguf" } },
.database = .{ .path = "./data" },
});
defer fw.deinit();
```

### Builder Pattern

```zig
var fw = try abi.App.builder(allocator)
.with(.gpu, abi.config.GpuConfig{ .backend = .vulkan })
.withDefault(.ai)
.withDefault(.database)
.build();
defer fw.deinit();
```

## Feature Access

```zig
// Check if a feature is enabled
if (fw.isEnabled(.gpu)) {
// Get the feature context
const gpu_ctx = try fw.get(.gpu);
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

**Source:** [`src/core/framework.zig`](../../src/core/framework.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-framework"></a>`pub const Framework`

<sup>**const**</sup> | [source](../../src/core/framework.zig#L143)

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

### <a id="pub-const-state"></a>`pub const State`

<sup>**const**</sup> | [source](../../src/core/framework.zig#L203)

Framework lifecycle states.

### <a id="pub-const-error"></a>`pub const Error`

<sup>**const**</sup> | [source](../../src/core/framework.zig#L207)

Composable framework error set.
See `core/errors.zig` for the full hierarchy.

### <a id="pub-fn-init-allocator-std-mem-allocator-cfg-config-error-framework"></a>`pub fn init(allocator: std.mem.Allocator, cfg: Config) Error!Framework`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L240)

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

### <a id="pub-fn-initwithio-allocator-std-mem-allocator-cfg-config-io-std-io-error-framework"></a>`pub fn initWithIo(allocator: std.mem.Allocator, cfg: Config, io: std.Io) Error!Framework`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L246)

Initialize the framework with the given configuration **and** an I/O backend.
This method is used by the builder when `withIo` is supplied.

### <a id="pub-fn-initdefault-allocator-std-mem-allocator-error-framework"></a>`pub fn initDefault(allocator: std.mem.Allocator) Error!Framework`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L269)

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

### <a id="pub-fn-initminimal-allocator-std-mem-allocator-error-framework"></a>`pub fn initMinimal(allocator: std.mem.Allocator) Error!Framework`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L297)

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

### <a id="pub-fn-builder-allocator-std-mem-allocator-frameworkbuilder"></a>`pub fn builder(allocator: std.mem.Allocator) FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L323)

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
.withDefault(.gpu)
.with(.ai, abi.config.AiConfig{ .llm = .{} })
.build();
defer fw.deinit();
```

### <a id="pub-fn-deinit-self-framework-void"></a>`pub fn deinit(self: *Framework) void`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L345)

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

### <a id="pub-fn-shutdownwithtimeout-self-framework-timeout-ms-u64-bool"></a>`pub fn shutdownWithTimeout(self: *Framework, timeout_ms: u64) bool`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L351)

Shutdown with timeout. Currently synchronous (timeout reserved for
future async cleanup). Returns true if clean shutdown completed.

### <a id="pub-fn-isrunning-self-const-framework-bool"></a>`pub fn isRunning(self: *const Framework) bool`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L360)

Check if the framework is running.

### <a id="pub-fn-isenabled-self-const-framework-feature-feature-bool"></a>`pub fn isEnabled(self: *const Framework, feature: Feature) bool`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L365)

Check if a feature is enabled.

### <a id="pub-fn-getstate-self-const-framework-state"></a>`pub fn getState(self: *const Framework) State`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L370)

Get the current framework state.

### <a id="pub-fn-getruntime-self-framework-runtime-mod-context"></a>`pub fn getRuntime(self: *Framework) *runtime_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L375)

Get runtime context (always available).

### <a id="pub-fn-getregistry-self-framework-registry"></a>`pub fn getRegistry(self: *Framework) *Registry`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L380)

Get the feature registry for runtime feature management.

### <a id="pub-fn-isfeatureregistered-self-const-framework-feature-feature-bool"></a>`pub fn isFeatureRegistered(self: *const Framework, feature: Feature) bool`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L385)

Check if a feature is registered in the registry.

### <a id="pub-fn-listregisteredfeatures-self-const-framework-allocator-std-mem-allocator-registryerror-feature"></a>`pub fn listRegisteredFeatures(self: *const Framework, allocator: std.mem.Allocator) RegistryError![]Feature`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L390)

List all registered features.

### <a id="pub-const-frameworkbuilder"></a>`pub const FrameworkBuilder`

<sup>**const**</sup> | [source](../../src/core/framework.zig#L396)

Fluent builder for Framework initialization.

### <a id="pub-fn-withdefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L408)

Start with default configuration.

### <a id="pub-fn-withio-self-frameworkbuilder-io-std-io-frameworkbuilder"></a>`pub fn withIo(self: *FrameworkBuilder, io: std.Io) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L414)

Provide a shared I/O backend for the framework.
Pass the `std.Io` obtained from `IoBackend.init`.

### <a id="pub-fn-with-self-frameworkbuilder-comptime-feature-feature-cfg-anytype-frameworkbuilder"></a>`pub fn with(self: *FrameworkBuilder, comptime feature: Feature, cfg: anytype) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L429)

Enable a feature with explicit configuration.

## Example
```zig
var fw = try Framework.builder(allocator)
.with(.gpu, abi.config.GpuConfig{ .backend = .vulkan })
.with(.database, abi.config.DatabaseConfig{ .path = "./data" })
.build();
```

### <a id="pub-fn-withdefault-self-frameworkbuilder-comptime-feature-feature-frameworkbuilder"></a>`pub fn withDefault(self: *FrameworkBuilder, comptime feature: Feature) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L442)

Enable a feature with its default configuration.

## Example
```zig
var fw = try Framework.builder(allocator)
.withDefault(.gpu)
.withDefault(.database)
.build();
```

### <a id="pub-fn-withplugins-self-frameworkbuilder-plugin-config-config-module-pluginconfig-frameworkbuilder"></a>`pub fn withPlugins(self: *FrameworkBuilder, plugin_config: config_module.PluginConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L447)

Configure plugins.

### <a id="pub-fn-build-self-frameworkbuilder-framework-error-framework"></a>`pub fn build(self: *FrameworkBuilder) Framework.Error!Framework`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L454)

Build and initialize the framework.
If an I/O backend was supplied via `withIo`, it will be stored in the
resulting `Framework` instance.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
