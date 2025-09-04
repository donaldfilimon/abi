# Abi AI Framework - Plugin ABI Specification

## Overview

The Abi AI Framework provides a stable Application Binary Interface (ABI) for plugin development, ensuring binary compatibility across different plugin implementations and framework versions.

## ABI Version

Current ABI Version: **1.0.0**

The ABI follows semantic versioning:
- **Major**: Breaking changes to the binary interface
- **Minor**: Backward-compatible additions
- **Patch**: Bug fixes with no interface changes

## Core Components

### 1. Plugin Entry Point

All plugins must export a C-compatible factory function:

```zig
// Standard plugin entry point
pub const PLUGIN_ENTRY_POINT = "abi_plugin_create";

// Function signature
pub const PluginFactoryFn = *const fn () callconv(.c) ?*const PluginInterface;
```

### 2. Plugin Interface Structure

The `PluginInterface` is an extern struct with C-compatible function pointers:

```zig
pub const PluginInterface = extern struct {
    // Required functions
    get_info: *const fn () callconv(.c) *const PluginInfo,
    init: *const fn (context: *PluginContext) callconv(.c) c_int,
    deinit: *const fn (context: *PluginContext) callconv(.c) void,

    // Optional lifecycle functions
    start: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    stop: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    pause: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    plugin_resume: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,

    // Processing and configuration
    process: ?*const fn (context: *PluginContext, input: ?*anyopaque, output: ?*anyopaque) callconv(.c) c_int = null,
    configure: ?*const fn (context: *PluginContext, config: *const PluginConfig) callconv(.c) c_int = null,
    get_config: ?*const fn (context: *PluginContext) callconv(.c) ?*const PluginConfig = null,

    // Status and diagnostics
    get_status: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    get_metrics: ?*const fn (context: *PluginContext, buffer: [*]u8, buffer_size: usize) callconv(.c) c_int = null,

    // Event handling
    on_event: ?*const fn (context: *PluginContext, event_type: u32, event_data: ?*anyopaque) callconv(.c) c_int = null,

    // Extended API access
    get_api: ?*const fn (api_name: [*:0]const u8) callconv(.c) ?*anyopaque = null,
};
```

### 3. Plugin Information Structure

```zig
pub const PluginInfo = struct {
    // Identity
    name: []const u8,
    version: PluginVersion,
    author: []const u8,
    description: []const u8,

    // Type and compatibility
    plugin_type: PluginType,
    abi_version: PluginVersion,

    // Dependencies and capabilities
    dependencies: []const []const u8 = &.{},
    provides: []const []const u8 = &.{},
    requires: []const []const u8 = &.{},

    // Optional metadata
    license: ?[]const u8 = null,
    homepage: ?[]const u8 = null,
    repository: ?[]const u8 = null,
};
```

## Plugin Types

The framework supports various plugin types:

- **Database Plugins**: vector_database, indexing_algorithm, compression_algorithm
- **AI/ML Plugins**: neural_network, embedding_generator, training_algorithm, inference_engine
- **Processing Plugins**: text_processor, image_processor, audio_processor, data_transformer
- **I/O Plugins**: data_loader, data_exporter, protocol_handler
- **Utility Plugins**: logger, metrics_collector, security_provider, configuration_provider
- **Custom Plugins**: For domain-specific extensions

## Plugin Lifecycle

### State Transitions

```
unloaded -> loaded -> initializing -> initialized -> running
                                    -> stopped
                                    -> paused -> running
                                    -> error_state
```

### Required Implementation

1. **get_info()**: Return static plugin information
2. **init()**: Initialize plugin with provided context
3. **deinit()**: Clean up resources and release memory

### Optional Implementation

- **start/stop**: For plugins with runtime services
- **pause/resume**: For suspendable operations
- **process**: For data transformation plugins
- **configure**: For runtime configuration changes
- **on_event**: For event-driven plugins

## Memory Management

- Plugins receive an allocator through `PluginContext`
- All allocations must be freed in `deinit()`
- The framework handles plugin instance lifecycle

## Error Handling

Return codes follow C conventions:
- `0`: Success
- Non-zero: Error (specific codes defined by plugin)

## Binary Compatibility Rules

1. **Stable ABI Guarantees**:
   - Function pointer signatures remain unchanged within major version
   - Struct layouts are fixed for extern structs
   - Optional functions can be null

2. **Version Compatibility**:
   - Plugins must declare their required ABI version
   - Framework checks compatibility before loading
   - Minor version increments are backward compatible

3. **Calling Conventions**:
   - All functions use `callconv(.c)` for cross-language compatibility
   - Parameters use C-compatible types or opaque pointers

## Best Practices

1. **Version Management**:
   - Always specify exact ABI version requirements
   - Test plugins against minimum and maximum supported versions

2. **Error Handling**:
   - Return appropriate error codes
   - Log detailed error information through plugin context

3. **Resource Management**:
   - Track all allocations
   - Implement proper cleanup in deinit()
   - Handle partial initialization failures

4. **Thread Safety**:
   - Assume plugin functions may be called from multiple threads
   - Use appropriate synchronization for shared state

## Example Implementation

See `/examples/plugins/example_plugin.zig` for a complete plugin implementation demonstrating:
- Proper ABI compliance
- Error handling
- Resource management
- Optional function implementation

## Migration Guide

When upgrading between ABI versions:

1. Check the changelog for breaking changes
2. Update plugin version requirements
3. Implement new required functions
4. Test thoroughly with new framework version

## Testing

The framework provides ABI compatibility tests:
- Version negotiation tests
- Function pointer validation
- State transition tests
- Error handling verification

Run tests with: `zig build test-plugins`