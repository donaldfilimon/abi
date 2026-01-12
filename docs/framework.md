# Framework Guide

This guide covers the initialization, configuration, and lifecycle management of an ABI application.

## Initialization

The entry point for any ABI application is the `abi.init` function. It establishes the runtime environment, sets up the memory allocator, and configures enabled features.

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // Initialize with default options
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // Framework is now ready
    std.debug.print("ABI v{s} initialized\n", .{abi.version()});
}
```

## Configuration

`abi.FrameworkOptions` allows you to customize the runtime behavior.

```zig
const options = abi.FrameworkOptions{
    .enable_ai = true,              // Enable AI features
    .enable_gpu = true,             // Enable GPU acceleration
    .enable_web = true,             // Enable web utilities
    .enable_database = true,        // Enable WDBX vector database
    .enable_network = true,         // Enable distributed compute
    .enable_profiling = true,       // Enable profiling/metrics
    .disabled_features = &.{},      // Explicitly disable specific features
    .plugin_paths = &.{             // Paths to plugin directories
        "./plugins",
        "/usr/local/abi/plugins",
    },
    .auto_discover_plugins = false, // Auto-discover plugins in paths
};
```

### FrameworkOptions Fields

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

Enable automatic plugin discovery by setting `auto_discover_plugins = true` in `FrameworkOptions`. The framework will scan all paths in `plugin_paths` for compatible plugins.

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
- [Troubleshooting](troubleshooting.md) - Feature disabled errors
