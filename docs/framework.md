# Framework Guide

This guide covers the initialization, configuration, and lifecycle management of an ABI application.

## Initialization

The entry point for any ABI application is the `abi.init` function. It establishes the runtime environment, sets up the memory allocator, and configures enabled features.

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    // Initialize with default options
    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // Framework is now ready
    std.debug.print("ABI v{s} initialized\n", .{abi.version()});
}
```

## Configuration

`abi.FrameworkOptions` allows you to customize the runtime behavior.

```zig
const options = abi.FrameworkOptions{
    .worker_threads = 4,           // Override default thread count
    .enable_metrics = true,        // Enable internal metrics collection
    .log_level = .info,           // Set logging verbosity
};
```

## Feature Flags

ABI uses build-time feature flags to minimize binary size and compilation time. These are passed to `zig build`.

| Flag                 | Default | Description                                |
| -------------------- | ------- | ------------------------------------------ |
| `-Denable-ai`        | `true`  | Enables AI agents and connectors           |
| `-Denable-gpu`       | `true`  | Enables GPU acceleration support           |
| `-Denable-database`  | `true`  | Enables WDBX vector database               |
| `-Denable-web`       | `true`  | Enables HTTP client/server utilities       |
| `-Denable-network`   | `false` | Enables distributed compute (Experimental) |
| `-Denable-profiling` | `false` | Enables performance profiling              |

## Lifecycle

Always ensure `abi.shutdown(&framework)` is called to release resources, stop worker threads, and flush logs.
