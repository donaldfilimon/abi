# Plugin System API

This document provides comprehensive API documentation for the plugin system
exposed via `abi.framework` and `abi.shared.*`. The runtime owns discovery and
registration while the shared layer exposes registries, loaders, and plugin
interfaces.

## Table of Contents

- [Overview](#overview)
- [Core Types](#core-types)
- [Functions](#functions)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Overview

Dynamic plugin system for extending framework functionality.

### Plugin Types

- **Database plugins**: Custom storage backends registered through
  `abi.shared.types.PluginType.vector_database`
- **AI/ML plugins**: Model implementations exposed under
  `abi.shared.types.PluginType.neural_network`
- **Processing plugins**: Data transformers and pipelines
- **I/O plugins**: Custom protocols and connector bridges

## Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var framework = try abi.init(std.heap.page_allocator, .{
        .plugin_paths = &.{ "./plugins" },
        .auto_discover_plugins = true,
        .auto_register_plugins = true,
    });
    defer framework.deinit();

    try framework.refreshPlugins();

    const registry = framework.pluginRegistry();
    std.log.info("Loaded plugins: {d}", .{registry.getPluginCount()});
}
```

