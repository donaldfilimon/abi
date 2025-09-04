# Plugin Development Guide

## Overview

This guide provides comprehensive instructions for developing plugins for the Abi AI Framework. The framework's plugin system offers a stable Application Binary Interface (ABI) that ensures compatibility across versions and languages.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Plugin Structure](#plugin-structure)
3. [Development Workflow](#development-workflow)
4. [Cross-Language Plugins](#cross-language-plugins)
5. [Testing & Validation](#testing--validation)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Zig compiler (0.11.0 or later) for Zig plugins
- C/C++ compiler for C-based plugins
- Understanding of the [Plugin ABI Specification](PLUGIN_ABI_SPECIFICATION.md)

### Quick Start

1. **Clone the example plugin**:
   ```bash
   cp examples/plugins/example_plugin.zig my_plugin.zig
   ```

2. **Modify the plugin information**:
   ```zig
   const PLUGIN_INFO = PluginInfo{
       .name = "my_plugin",
       .version = PluginVersion.init(1, 0, 0),
       .author = "Your Name",
       .description = "My custom plugin",
       .plugin_type = .custom,
       .abi_version = PluginVersion.init(1, 0, 0),
   };
   ```

3. **Build the plugin**:
   ```bash
   zig build-lib -dynamic -O ReleaseFast my_plugin.zig
   ```

## Plugin Structure

### Minimal Plugin

A minimal plugin requires only three functions:

```zig
const std = @import("std");

// Plugin information
const PLUGIN_INFO = PluginInfo{
    .name = "minimal_plugin",
    .version = PluginVersion.init(1, 0, 0),
    .author = "Developer",
    .description = "Minimal plugin example",
    .plugin_type = .custom,
    .abi_version = PluginVersion.init(1, 0, 0),
};

// Required functions
fn getInfo() callconv(.c) *const PluginInfo {
    return &PLUGIN_INFO;
}

fn initPlugin(context: *PluginContext) callconv(.c) c_int {
    _ = context;
    // Initialize your plugin
    return 0; // Success
}

fn deinitPlugin(context: *PluginContext) callconv(.c) void {
    _ = context;
    // Cleanup resources
}

// Plugin interface
const PLUGIN_INTERFACE = PluginInterface{
    .get_info = getInfo,
    .init = initPlugin,
    .deinit = deinitPlugin,
};

// Entry point
export fn abi_plugin_create() callconv(.c) ?*const PluginInterface {
    return &PLUGIN_INTERFACE;
}
```

### Full-Featured Plugin

For a complete example with all optional features, see:
- [Advanced Plugin Example](../examples/plugins/advanced_plugin_example.zig)

Key features include:
- State management
- Configuration handling
- Event processing
- Performance metrics
- Extended APIs
- Thread safety

## Development Workflow

### 1. Design Phase

Define your plugin's:
- **Purpose**: What problem does it solve?
- **Type**: Choose from available plugin types
- **Dependencies**: What other plugins/services are needed?
- **Capabilities**: What functionality will it provide?

### 2. Implementation

Follow this structure:

```zig
// 1. Import dependencies
const std = @import("std");
const plugin_types = @import("plugin_types.zig");

// 2. Define plugin state
var plugin_state = struct {
    initialized: bool = false,
    // ... your state fields
}{};

// 3. Implement required functions
fn getInfo() callconv(.c) *const PluginInfo { ... }
fn initPlugin(ctx: *PluginContext) callconv(.c) c_int { ... }
fn deinitPlugin(ctx: *PluginContext) callconv(.c) void { ... }

// 4. Implement optional functions as needed
fn startPlugin(ctx: *PluginContext) callconv(.c) c_int { ... }
fn processData(ctx: *PluginContext, in: ?*anyopaque, out: ?*anyopaque) callconv(.c) c_int { ... }

// 5. Create interface vtable
const PLUGIN_INTERFACE = PluginInterface{
    .get_info = getInfo,
    .init = initPlugin,
    .deinit = deinitPlugin,
    .start = startPlugin,
    .process = processData,
    // ... other functions
};

// 6. Export entry point
export fn abi_plugin_create() callconv(.c) ?*const PluginInterface {
    return &PLUGIN_INTERFACE;
}
```

### 3. Building

#### Using Zig:
```bash
# Debug build
zig build-lib -dynamic my_plugin.zig

# Release build
zig build-lib -dynamic -O ReleaseFast my_plugin.zig

# Cross-compilation
zig build-lib -dynamic -target x86_64-windows my_plugin.zig
```

#### Using the build system:
```bash
# Add to build_plugins.zig
zig build plugins
```

### 4. Testing

Use the provided tools:

```bash
# Validate ABI compliance
./plugin_validator my_plugin.so --verbose

# Run comprehensive tests
./plugin_test_harness my_plugin.so --stress --benchmark

# Run specific tests
./plugin_test_harness my_plugin.so --filter lifecycle
```

## Cross-Language Plugins

### C/C++ Plugin

```c
#include <stdint.h>
#include <string.h>

// Plugin info structure
typedef struct {
    const char* name;
    struct { uint32_t major, minor, patch; } version;
    const char* author;
    const char* description;
    // ... other fields
} PluginInfo;

// Plugin interface
typedef struct {
    const PluginInfo* (*get_info)(void);
    int (*init)(void* context);
    void (*deinit)(void* context);
    // ... other function pointers
} PluginInterface;

// Implementation
static const PluginInfo plugin_info = {
    .name = "c_plugin",
    .version = {1, 0, 0},
    .author = "Developer",
    .description = "C plugin example"
};

static const PluginInfo* get_info(void) {
    return &plugin_info;
}

static int init_plugin(void* context) {
    // Initialize
    return 0;
}

static void deinit_plugin(void* context) {
    // Cleanup
}

static const PluginInterface plugin_interface = {
    .get_info = get_info,
    .init = init_plugin,
    .deinit = deinit_plugin
};

// Entry point
__attribute__((visibility("default")))
const PluginInterface* abi_plugin_create(void) {
    return &plugin_interface;
}
```

### Rust Plugin

```rust
use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
struct PluginInfo {
    name: *const c_char,
    version: PluginVersion,
    author: *const c_char,
    description: *const c_char,
    // ... other fields
}

#[repr(C)]
struct PluginVersion {
    major: u32,
    minor: u32,
    patch: u32,
}

#[repr(C)]
struct PluginInterface {
    get_info: extern "C" fn() -> *const PluginInfo,
    init: extern "C" fn(*mut c_void) -> c_int,
    deinit: extern "C" fn(*mut c_void),
    // ... other function pointers
}

static PLUGIN_INFO: PluginInfo = PluginInfo {
    name: b"rust_plugin\0".as_ptr() as *const c_char,
    version: PluginVersion { major: 1, minor: 0, patch: 0 },
    author: b"Developer\0".as_ptr() as *const c_char,
    description: b"Rust plugin example\0".as_ptr() as *const c_char,
};

extern "C" fn get_info() -> *const PluginInfo {
    &PLUGIN_INFO
}

extern "C" fn init_plugin(_context: *mut c_void) -> c_int {
    // Initialize
    0 // Success
}

extern "C" fn deinit_plugin(_context: *mut c_void) {
    // Cleanup
}

static PLUGIN_INTERFACE: PluginInterface = PluginInterface {
    get_info,
    init: init_plugin,
    deinit: deinit_plugin,
};

#[no_mangle]
pub extern "C" fn abi_plugin_create() -> *const PluginInterface {
    &PLUGIN_INTERFACE
}
```

## Testing & Validation

### 1. Unit Testing

Create tests for your plugin logic:

```zig
test "plugin initialization" {
    var context = TestContext.init();
    defer context.deinit();
    
    const result = initPlugin(&context);
    try std.testing.expectEqual(0, result);
    try std.testing.expect(plugin_state.initialized);
}
```

### 2. Integration Testing

Test with the framework:

```zig
test "plugin lifecycle" {
    var loader = try PluginLoader.init(allocator);
    defer loader.deinit();
    
    const handle = try loader.loadPlugin("./my_plugin.so");
    var plugin = try loader.getPlugin(handle);
    
    try plugin.initialize(&config);
    try plugin.start();
    // ... test operations
    try plugin.stop();
}
```

### 3. Validation Checklist

- [ ] Plugin loads without errors
- [ ] Entry point is exported correctly
- [ ] All required functions are implemented
- [ ] ABI version is compatible
- [ ] Memory is properly managed
- [ ] Error codes are returned correctly
- [ ] Thread safety (if applicable)
- [ ] Performance meets requirements

## Best Practices

### 1. Error Handling

Always return appropriate error codes:

```zig
fn processData(ctx: *PluginContext, input: ?*anyopaque, output: ?*anyopaque) callconv(.c) c_int {
    if (input == null or output == null) {
        return -1; // Invalid parameters
    }
    
    // Process data
    doProcess(input, output) catch |err| {
        logError(ctx, "Processing failed: {}", .{err});
        return -2; // Processing error
    };
    
    return 0; // Success
}
```

### 2. Memory Management

Track all allocations:

```zig
const Allocation = struct {
    ptr: [*]u8,
    size: usize,
};

var allocations = std.ArrayList(Allocation).init(allocator);

fn allocateMemory(size: usize) !*anyopaque {
    const mem = try allocator.alloc(u8, size);
    try allocations.append(.{ .ptr = mem.ptr, .size = size });
    return mem.ptr;
}

fn cleanup() void {
    for (allocations.items) |alloc| {
        allocator.free(alloc.ptr[0..alloc.size]);
    }
    allocations.deinit();
}
```

### 3. Thread Safety

Use appropriate synchronization:

```zig
var mutex = std.Thread.Mutex{};
var shared_data: SharedData = .{};

fn threadSafeOperation() !void {
    mutex.lock();
    defer mutex.unlock();
    
    // Modify shared data safely
    shared_data.update();
}
```

### 4. Performance

- Use atomic operations for counters
- Implement caching where appropriate
- Batch operations when possible
- Profile and optimize hot paths

### 5. Logging

Use the provided logging interface:

```zig
fn logMessage(ctx: *PluginContext, level: u8, comptime fmt: []const u8, args: anytype) void {
    var buffer: [1024]u8 = undefined;
    const msg = std.fmt.bufPrint(&buffer, fmt, args) catch return;
    
    if (ctx.log_fn) |log_fn| {
        log_fn(ctx, level, msg);
    }
}
```

## Troubleshooting

### Common Issues

1. **Plugin fails to load**
   - Check export symbol: `nm -D my_plugin.so | grep abi_plugin_create`
   - Verify ABI version compatibility
   - Check for missing dependencies: `ldd my_plugin.so`

2. **Initialization fails**
   - Check error codes returned
   - Verify required resources are available
   - Check logs for error messages

3. **Crashes or segfaults**
   - Use address sanitizer: `zig build-lib -fsanitize=address`
   - Check for null pointer access
   - Verify memory alignment

4. **Memory leaks**
   - Ensure all allocations are freed in deinit
   - Use valgrind for detection
   - Implement allocation tracking

5. **Performance issues**
   - Profile with perf or similar tools
   - Check for unnecessary allocations
   - Optimize hot paths
   - Consider caching

### Debug Tips

1. Enable verbose logging:
   ```zig
   const DEBUG = true;
   
   fn debug(comptime fmt: []const u8, args: anytype) void {
       if (DEBUG) {
           std.debug.print("[PLUGIN] " ++ fmt ++ "\n", args);
       }
   }
   ```

2. Use the test harness:
   ```bash
   ./plugin_test_harness my_plugin.so --filter specific_test
   ```

3. Validate frequently during development:
   ```bash
   ./plugin_validator my_plugin.so --verbose
   ```

## Resources

- [Plugin ABI Specification](PLUGIN_ABI_SPECIFICATION.md)
- [Plugin System Documentation](PLUGIN_SYSTEM.md)
- [Example Plugins](../examples/plugins/)
- [Test Suite](../tests/test_plugin_abi.zig)
- [Validation Tools](../tools/)

## Support

For questions and support:
- Check the [FAQ](FAQ.md)
- Review [existing plugins](../examples/plugins/)
- Open an issue on [GitHub](https://github.com/donaldfilimon/abi)