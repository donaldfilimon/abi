---
title: "todo"
tags: []
---
# Development TODO & Zig 0.16 Patterns
> **Codebase Status:** Synced with repository as of 2026-01-24.

> **Developer Guide**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for coding patterns and [CLAUDE.md](../CLAUDE.md) for comprehensive guidance.
>
> **Last Updated:** January 23, 2026
> **Zig Version:** 0.16.x

## Zig 0.16 Environment Initialization

Zig 0.16 introduces a unified I/O system that requires explicit environment configuration. This section documents the patterns for initializing the environment correctly.

### Basic Environment Setup

The `std.Io.Threaded` backend provides synchronous I/O operations with environment access:

```zig
const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize I/O backend with environment configuration
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,  // Use empty for no env access
    });
    defer io_backend.deinit();

    // Get the I/O interface for operations
    const io = io_backend.io();

    // Now you can use io for file operations
    const content = try std.Io.Dir.cwd().readFileAlloc(
        io,
        "config.json",
        allocator,
        .limited(10 * 1024 * 1024),  // 10MB limit
    );
    defer allocator.free(content);
}
```

### Environment Options

The `std.Io.Threaded.init()` accepts an options struct with these fields:

| Option | Type | Description |
|--------|------|-------------|
| `environ` | `std.process.Environ` | Environment variable access configuration |

**Environment configurations:**

```zig
// No environment access (most common for library code)
.environ = std.process.Environ.empty

// Full environment access (for CLI applications)
.environ = std.process.Environ.init()

// Custom environment (for testing)
.environ = std.process.Environ.fromMap(&.{
    .{ "HOME", "/home/user" },
    .{ "PATH", "/usr/bin" },
})
```

### File Operations Pattern

```zig
fn readConfigFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    return std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(10 * 1024 * 1024),
    );
}

fn writeConfigFile(allocator: std.mem.Allocator, path: []const u8, content: []const u8) !void {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);

    var writer = file.writer(io);
    try writer.writeAll(content);
}
```

### Directory Operations

```zig
fn ensureDirectory(allocator: std.mem.Allocator, path: []const u8) !void {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    std.Io.Dir.cwd().makePath(io, path) catch |err| switch (err) {
        error.PathAlreadyExists => {},  // Directory exists, that's fine
        else => return err,
    };
}

fn listDirectory(allocator: std.mem.Allocator, path: []const u8) !void {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        std.debug.print("{s}\n", .{entry.name});
    }
}
```

### HTTP Server with Environment

```zig
fn startHttpServer(allocator: std.mem.Allocator, stream: anytype) !void {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    var recv_buffer: [8192]u8 = undefined;
    var send_buffer: [8192]u8 = undefined;

    var connection_reader = stream.reader(io, &recv_buffer);
    var connection_writer = stream.writer(io, &send_buffer);

    // Use .interface to get the correct type for std.http.Server
    var server: std.http.Server = .init(
        &connection_reader.interface,
        &connection_writer.interface,
    );

    // Handle requests...
}
```

### Best Practices

1. **Scope the I/O backend narrowly**: Create the `io_backend` only when needed and `defer` its cleanup immediately.

2. **Use `Environ.empty` for library code**: Unless you specifically need environment variables, use `std.process.Environ.empty`.

3. **Pass `io` to operations**: All `std.Io.Dir` operations require the `io` parameter.

4. **Handle file limits**: Use `.limited(size)` to prevent reading excessively large files.

5. **Close files explicitly**: Always `defer file.close(io)` after opening.

### Migration from Zig 0.15

| Old Pattern (Zig 0.15) | New Pattern (Zig 0.16) |
|------------------------|------------------------|
| `std.fs.cwd()` | `std.Io.Dir.cwd()` (requires `io` param) |
| `std.fs.openFile()` | `std.Io.Dir.cwd().openFile(io, ...)` |
| `std.fs.createFile()` | `std.Io.Dir.cwd().createFile(io, ...)` |
| `std.io.AnyReader` | `std.Io.Reader` |
| `std.time.sleep()` | `std.Io.Clock.Duration.sleep(duration, io)` |

## Current Development Tasks

See [TODO.md](../TODO.md) for the comprehensive project task list.

### Quick Links

- [Zig 0.16 Migration Guide](migration/zig-0.16-migration.md) - Full migration documentation
- [Framework Guide](framework.md) - Configuration and initialization
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

---

<p align="center">
  <a href="docs-index.md">&larr; Documentation Index</a>
</p>

