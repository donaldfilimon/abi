# Framework Guide

This guide covers the initialization, configuration, and lifecycle management of an ABI application.

## Architecture Overview

The framework uses a flat domain structure with clear separation of concerns:

```
src/
├── abi.zig              # Public API entry point
├── config.zig           # Unified configuration system
├── framework.zig        # Framework orchestration
├── runtime/             # Always-on infrastructure (memory, scheduling)
├── gpu/                 # GPU acceleration (moved from compute/)
├── ai/                  # AI module with sub-features
├── database/            # Vector database (WDBX)
├── network/             # Distributed compute
├── observability/       # Metrics, tracing, profiling
├── web/                 # Web/HTTP utilities
└── internal/            # Shared utilities
```

## Initialization

The entry point for any ABI application is the `abi.init` function. It establishes the runtime environment, sets up the memory allocator, and configures enabled features.

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize with default configuration
    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    // Framework is now ready
    std.debug.print("ABI v{s} initialized\n", .{abi.version()});
}
```

## Configuration

### Unified Config System

The new `Config` struct in `config.zig` provides a unified configuration system with a builder pattern for fluent configuration.

```zig
const abi = @import("abi");

// Using the builder pattern
var config = abi.Config.init()
    .enableAi(true)
    .enableGpu(true)
    .enableDatabase(true)
    .enableNetwork(true)
    .enableProfiling(true)
    .setPluginPaths(&.{
        "./plugins",
        "/usr/local/abi/plugins",
    })
    .build();

var framework = try abi.init(allocator, config);
defer abi.shutdown(&framework);
```

### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_ai` | `bool` | build default | Enable AI features and connectors |
| `enable_gpu` | `bool` | build default | Enable GPU acceleration |
| `enable_web` | `bool` | build default | Enable HTTP utilities |
| `enable_database` | `bool` | build default | Enable WDBX vector database |
| `enable_network` | `bool` | build default | Enable distributed compute |
| `enable_profiling` | `bool` | build default | Enable profiling metrics |
| `disabled_features` | `[]const Feature` | `&.{}` | Features to explicitly disable |
| `plugin_paths` | `[]const []const u8` | `&.{}` | Plugin search paths |
| `auto_discover_plugins` | `bool` | `false` | Auto-discover plugins |

### Legacy Configuration

For backward compatibility, `abi.FrameworkOptions` is still supported:

```zig
const options = abi.FrameworkOptions{
    .enable_ai = true,
    .enable_gpu = true,
    .enable_web = true,
    .enable_database = true,
    .enable_network = true,
    .enable_profiling = true,
    .disabled_features = &.{},
    .plugin_paths = &.{
        "./plugins",
        "/usr/local/abi/plugins",
    },
    .auto_discover_plugins = false,
};
```

## Framework Orchestration

The `Framework` struct in `framework.zig` manages feature lifecycles and orchestration.

### Framework Lifecycle

```zig
var framework = try abi.init(allocator, config);
defer abi.shutdown(&framework);

// Access framework state
const is_ai_enabled = framework.isFeatureEnabled(.ai);
const is_gpu_ready = framework.isFeatureReady(.gpu);

// Get feature handles
if (framework.getFeature(.ai)) |ai_feature| {
    // Use AI feature
}
```

### Feature States

The framework tracks feature states through their lifecycle:

| State | Description |
|-------|-------------|
| `disabled` | Feature not enabled in config |
| `initializing` | Feature is starting up |
| `ready` | Feature is operational |
| `degraded` | Feature operational with reduced capability |
| `failed` | Feature failed to initialize |
| `shutdown` | Feature is shutting down |

## Feature Flags

ABI uses build-time feature flags to minimize binary size and compilation time. These are passed to `zig build`.

| Flag                 | Default | Description                                |
| -------------------- | ------- | ------------------------------------------ |
| `-Denable-ai`        | `true`  | Enables AI agents and connectors           |
| `-Denable-gpu`       | `true`  | Enables GPU acceleration support           |
| `-Denable-database`  | `true`  | Enables WDBX vector database               |
| `-Denable-web`       | `true`  | Enables HTTP client/server utilities       |
| `-Denable-network`   | `true`  | Enables distributed compute                |
| `-Denable-profiling` | `true`  | Enables performance profiling              |

## Lifecycle

Always ensure `abi.shutdown(&framework)` is called to release resources, stop worker threads, and flush logs.

```zig
// Proper lifecycle management
var framework = try abi.init(allocator, .{});
defer abi.shutdown(&framework);

// Or manual shutdown with error handling
errdefer abi.shutdown(&framework);
// ... use framework ...
abi.shutdown(&framework);
```

## Plugin System

The framework supports runtime-loadable plugins for extending functionality.

### Plugin Registration

```zig
const plugins = @import("abi").plugins;

var registry = plugins.PluginRegistry.init(allocator);
defer registry.deinit();

// Register a plugin
try registry.register(
    "my-connector",           // Plugin name
    "./plugins/connector.so", // Plugin path
    "ai",                     // Associated feature
);

// Find a plugin by name
if (registry.findByName("my-connector")) |plugin| {
    std.debug.print("Found plugin: {s}\n", .{plugin.path});
}
```

### Plugin Descriptor

| Field | Type | Description |
|-------|------|-------------|
| `name` | `[]const u8` | Unique plugin identifier |
| `path` | `[]const u8` | Path to plugin file |
| `feature` | `[]const u8` | Associated feature (ai, gpu, database, etc.) |

### Auto-Discovery

Enable automatic plugin discovery by setting `auto_discover_plugins = true` in the config. The framework will scan all paths in `plugin_paths` for compatible plugins.

---

## CLI Commands

```bash
# Configuration management
zig build run -- config init       # Initialize configuration
zig build run -- config show       # Show current configuration
zig build run -- config validate   # Validate configuration

# System information
zig build run -- system-info       # Show framework status
zig build run -- --version         # Show version
```

---

## See Also

- [Introduction](intro.md) - Architecture overview
- [Monitoring](monitoring.md) - Logging and metrics configuration
- [Compute Engine](compute.md) - Engine configuration
- [GPU Acceleration](gpu.md) - GPU module (now at top-level)
- [Troubleshooting](troubleshooting.md) - Feature disabled errors
