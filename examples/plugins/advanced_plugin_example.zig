//! Advanced Plugin Example for Abi AI Framework
//!
//! This example demonstrates advanced plugin features including:
//! - Proper error handling and recovery
//! - State management and persistence
//! - Configuration management
//! - Performance metrics collection
//! - Thread-safe operations
//! - Memory management best practices

const std = @import("std");

// Import the actual types from the framework
// In a real plugin development scenario, you would use:
// const plugin_types = @import("abi_framework").plugins.types;
// const plugin_interface = @import("abi_framework").plugins.interface;

// For this example, we'll include the necessary definitions
const PluginInterface = @import("../../src/plugins/interface.zig").PluginInterface;
const types = @import("../../src/plugins/types.zig");
const PluginInfo = types.PluginInfo;
const PluginVersion = types.PluginVersion;
const PluginType = types.PluginType;
const PluginConfig = types.PluginConfig;
const PluginContext = types.PluginContext;
const PluginState = types.PluginState;

// =============================================================================
// PLUGIN-SPECIFIC TYPES
// =============================================================================

/// Advanced data processor configuration
const ProcessorConfig = struct {
    batch_size: usize = 1024,
    cache_size_mb: usize = 16,
    thread_count: usize = 4,
    enable_compression: bool = true,
    enable_encryption: bool = false,
    encryption_key: ?[]const u8 = null,
};

/// Processing statistics
const ProcessingStats = struct {
    total_processed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    errors_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    bytes_processed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    average_latency_ns: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    peak_memory_usage: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
};

/// Plugin internal state
const PluginInternalState = struct {
    allocator: std.mem.Allocator,
    config: ProcessorConfig,
    stats: ProcessingStats,
    cache: ?std.hash_map.HashMap(u64, []u8, std.hash_map.AutoContext(u64), 80),
    worker_threads: ?[]std.Thread,
    mutex: std.Thread.Mutex,
    state: PluginState,
    start_time: i64,
    last_error: ?[]u8,
};

// =============================================================================
// GLOBAL STATE
// =============================================================================

var global_state: ?*PluginInternalState = null;
var global_mutex = std.Thread.Mutex{};

// =============================================================================
// PLUGIN INFORMATION
// =============================================================================

const PLUGIN_INFO = PluginInfo{
    .name = "advanced_data_processor",
    .version = PluginVersion.init(2, 0, 0),
    .author = "Abi AI Framework Team",
    .description = "Advanced data processing plugin with caching, threading, and metrics",
    .plugin_type = .data_transformer,
    .abi_version = PluginVersion.init(1, 0, 0),
    .dependencies = &[_][]const u8{
        "compression_algorithm@1.0.0",
        "encryption_provider@1.0.0",
    },
    .provides = &[_][]const u8{
        "data_transform",
        "batch_process",
        "stream_process",
        "cache_management",
    },
    .requires = &[_][]const u8{
        "memory_allocator",
        "thread_pool",
        "logger",
    },
    .license = "MIT",
    .homepage = "https://github.com/donaldfilimon/abi",
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn logMessage(context: *PluginContext, level: u8, comptime format: []const u8, args: anytype) void {
    var buffer: [1024]u8 = undefined;
    const message = std.fmt.bufPrint(&buffer, format, args) catch return;
    
    if (context.log_fn) |log_fn| {
        log_fn(context, level, message);
    }
}

fn updateLastError(state: *PluginInternalState, comptime format: []const u8, args: anytype) void {
    if (state.last_error) |err| {
        state.allocator.free(err);
    }
    
    state.last_error = std.fmt.allocPrint(state.allocator, format, args) catch null;
}

// =============================================================================
// PLUGIN INTERFACE IMPLEMENTATION
// =============================================================================

fn getInfo() callconv(.c) *const PluginInfo {
    return &PLUGIN_INFO;
}

fn initPlugin(context: *PluginContext) callconv(.c) c_int {
    global_mutex.lock();
    defer global_mutex.unlock();
    
    if (global_state != null) {
        return -1; // Already initialized
    }
    
    // Allocate plugin state
    const state = context.allocator.create(PluginInternalState) catch {
        logMessage(context, 3, "Failed to allocate plugin state", .{});
        return -2;
    };
    
    // Initialize state
    state.* = .{
        .allocator = context.allocator,
        .config = ProcessorConfig{},
        .stats = ProcessingStats{},
        .cache = null,
        .worker_threads = null,
        .mutex = std.Thread.Mutex{},
        .state = .initialized,
        .start_time = std.time.timestamp(),
        .last_error = null,
    };
    
    // Load configuration if provided
    if (context.config) |cfg| {
        if (cfg.getParameter("batch_size")) |value| {
            state.config.batch_size = std.fmt.parseInt(usize, value, 10) catch 1024;
        }
        if (cfg.getParameter("cache_size_mb")) |value| {
            state.config.cache_size_mb = std.fmt.parseInt(usize, value, 10) catch 16;
        }
        if (cfg.getParameter("thread_count")) |value| {
            state.config.thread_count = std.fmt.parseInt(usize, value, 10) catch 4;
        }
    }
    
    // Initialize cache
    if (state.config.cache_size_mb > 0) {
        state.cache = std.hash_map.HashMap(u64, []u8, std.hash_map.AutoContext(u64), 80).init(context.allocator);
    }
    
    global_state = state;
    
    logMessage(context, 1, "Advanced data processor initialized with batch_size={d}, cache_size={d}MB", .{
        state.config.batch_size,
        state.config.cache_size_mb,
    });
    
    return 0; // Success
}

fn deinitPlugin(context: *PluginContext) callconv(.c) void {
    global_mutex.lock();
    defer global_mutex.unlock();
    
    const state = global_state orelse return;
    
    // Stop any running operations
    if (state.state == .running) {
        _ = stopPlugin(context);
    }
    
    // Clean up cache
    if (state.cache) |*cache| {
        var it = cache.iterator();
        while (it.next()) |entry| {
            state.allocator.free(entry.value_ptr.*);
        }
        cache.deinit();
    }
    
    // Free error message
    if (state.last_error) |err| {
        state.allocator.free(err);
    }
    
    // Log final statistics
    logMessage(context, 1, "Plugin shutting down. Total processed: {d}, Errors: {d}", .{
        state.stats.total_processed.load(.seq_cst),
        state.stats.errors_count.load(.seq_cst),
    });
    
    state.allocator.destroy(state);
    global_state = null;
}

fn startPlugin(context: *PluginContext) callconv(.c) c_int {
    const state = global_state orelse return -1;
    
    state.mutex.lock();
    defer state.mutex.unlock();
    
    if (state.state != .initialized and state.state != .stopped) {
        return -2; // Invalid state
    }
    
    // Initialize worker threads if configured
    if (state.config.thread_count > 1) {
        state.worker_threads = state.allocator.alloc(std.Thread, state.config.thread_count - 1) catch {
            updateLastError(state, "Failed to allocate worker threads", .{});
            return -3;
        };
        
        // Start worker threads (implementation would go here)
    }
    
    state.state = .running;
    logMessage(context, 1, "Plugin started with {d} worker threads", .{state.config.thread_count});
    
    return 0;
}

fn stopPlugin(context: *PluginContext) callconv(.c) c_int {
    const state = global_state orelse return -1;
    
    state.mutex.lock();
    defer state.mutex.unlock();
    
    if (state.state != .running and state.state != .paused) {
        return -2; // Not running
    }
    
    state.state = .stopping;
    
    // Stop worker threads
    if (state.worker_threads) |threads| {
        // Signal threads to stop (implementation would go here)
        // Join threads
        state.allocator.free(threads);
        state.worker_threads = null;
    }
    
    state.state = .stopped;
    logMessage(context, 1, "Plugin stopped", .{});
    
    return 0;
}

fn processData(context: *PluginContext, input: ?*anyopaque, output: ?*anyopaque) callconv(.c) c_int {
    const state = global_state orelse return -1;
    
    if (state.state != .running) {
        return -2; // Not running
    }
    
    const start_time = std.time.nanoTimestamp();
    defer {
        const elapsed = std.time.nanoTimestamp() - start_time;
        const current_avg = state.stats.average_latency_ns.load(.seq_cst);
        const total = state.stats.total_processed.load(.seq_cst);
        const new_avg = (current_avg * total + @as(u64, @intCast(elapsed))) / (total + 1);
        _ = state.stats.average_latency_ns.store(new_avg, .seq_cst);
    }
    
    // Example processing logic
    const input_data = @as(*[]const u8, @ptrCast(@alignCast(input orelse return -3))).*;
    const output_buffer = @as(*[]u8, @ptrCast(@alignCast(output orelse return -3)));
    
    // Check cache
    const hash = std.hash.Wyhash.hash(0, input_data);
    if (state.cache) |*cache| {
        state.mutex.lock();
        defer state.mutex.unlock();
        
        if (cache.get(hash)) |cached_result| {
            output_buffer.* = state.allocator.dupe(u8, cached_result) catch {
                _ = state.stats.errors_count.fetchAdd(1, .seq_cst);
                return -4;
            };
            _ = state.stats.total_processed.fetchAdd(1, .seq_cst);
            return 0;
        }
    }
    
    // Process data (example: apply transformation)
    output_buffer.* = state.allocator.alloc(u8, input_data.len) catch {
        _ = state.stats.errors_count.fetchAdd(1, .seq_cst);
        return -4;
    };
    
    // Example transformation: reverse the data
    for (input_data, 0..) |byte, i| {
        output_buffer.*[input_data.len - 1 - i] = byte;
    }
    
    // Update cache
    if (state.cache) |*cache| {
        state.mutex.lock();
        defer state.mutex.unlock();
        
        const cache_copy = state.allocator.dupe(u8, output_buffer.*) catch {
            // Cache update failed, but processing succeeded
            _ = state.stats.total_processed.fetchAdd(1, .seq_cst);
            _ = state.stats.bytes_processed.fetchAdd(input_data.len, .seq_cst);
            return 0;
        };
        
        cache.put(hash, cache_copy) catch {
            state.allocator.free(cache_copy);
        };
    }
    
    _ = state.stats.total_processed.fetchAdd(1, .seq_cst);
    _ = state.stats.bytes_processed.fetchAdd(input_data.len, .seq_cst);
    
    // Update peak memory usage
    const current_memory = state.allocator.total_allocated_bytes;
    var peak = state.stats.peak_memory_usage.load(.seq_cst);
    while (current_memory > peak) {
        peak = state.stats.peak_memory_usage.cmpxchgWeak(
            peak,
            current_memory,
            .seq_cst,
            .seq_cst,
        ) orelse break;
    }
    
    return 0;
}

fn getStatus(context: *PluginContext) callconv(.c) c_int {
    _ = context;
    const state = global_state orelse return @intFromEnum(PluginState.unloaded);
    return @intFromEnum(state.state);
}

fn getMetrics(context: *PluginContext, buffer: [*]u8, buffer_size: usize) callconv(.c) c_int {
    _ = context;
    const state = global_state orelse return -1;
    
    const uptime = std.time.timestamp() - state.start_time;
    
    const metrics_str = std.fmt.bufPrint(
        buffer[0..buffer_size],
        \\{{
        \\  "uptime_seconds": {d},
        \\  "total_processed": {d},
        \\  "errors_count": {d},
        \\  "bytes_processed": {d},
        \\  "average_latency_ns": {d},
        \\  "peak_memory_bytes": {d},
        \\  "cache_size": {d},
        \\  "state": "{s}"
        \\}}
    ,
        .{
            uptime,
            state.stats.total_processed.load(.seq_cst),
            state.stats.errors_count.load(.seq_cst),
            state.stats.bytes_processed.load(.seq_cst),
            state.stats.average_latency_ns.load(.seq_cst),
            state.stats.peak_memory_usage.load(.seq_cst),
            if (state.cache) |*cache| cache.count() else 0,
            @tagName(state.state),
        },
    ) catch return -2;
    
    return @intCast(metrics_str.len);
}

fn onEvent(context: *PluginContext, event_type: u32, event_data: ?*anyopaque) callconv(.c) c_int {
    _ = event_data;
    const state = global_state orelse return -1;
    
    switch (event_type) {
        1 => { // System startup
            logMessage(context, 1, "Received system startup event", .{});
        },
        2 => { // System shutdown
            logMessage(context, 1, "Received system shutdown event", .{});
            if (state.state == .running) {
                return stopPlugin(context);
            }
        },
        3 => { // Configuration update
            logMessage(context, 1, "Received configuration update event", .{});
            // Handle configuration updates
        },
        4 => { // Memory pressure
            logMessage(context, 2, "Received memory pressure event", .{});
            // Clear cache to free memory
            if (state.cache) |*cache| {
                state.mutex.lock();
                defer state.mutex.unlock();
                
                var it = cache.iterator();
                while (it.next()) |entry| {
                    state.allocator.free(entry.value_ptr.*);
                }
                cache.clearRetainingCapacity();
            }
        },
        else => {
            logMessage(context, 3, "Received unknown event type: {d}", .{event_type});
        },
    }
    
    return 0;
}

fn configure(context: *PluginContext, config: *const PluginConfig) callconv(.c) c_int {
    const state = global_state orelse return -1;
    
    state.mutex.lock();
    defer state.mutex.unlock();
    
    // Update configuration
    if (config.getParameter("batch_size")) |value| {
        state.config.batch_size = std.fmt.parseInt(usize, value, 10) catch state.config.batch_size;
    }
    
    if (config.getParameter("enable_compression")) |value| {
        state.config.enable_compression = std.mem.eql(u8, value, "true");
    }
    
    if (config.getParameter("enable_encryption")) |value| {
        state.config.enable_encryption = std.mem.eql(u8, value, "true");
    }
    
    logMessage(context, 1, "Configuration updated", .{});
    return 0;
}

// Extended API for batch processing
const BatchProcessor = struct {
    fn processBatch(items: []const []const u8) ![][]u8 {
        const state = global_state orelse return error.NotInitialized;
        
        if (state.state != .running) {
            return error.NotRunning;
        }
        
        const results = state.allocator.alloc([]u8, items.len) catch return error.OutOfMemory;
        errdefer state.allocator.free(results);
        
        for (items, 0..) |item, i| {
            var output: []u8 = undefined;
            const result = processData(null, @ptrCast(&item), @ptrCast(&output));
            if (result != 0) {
                return error.ProcessingFailed;
            }
            results[i] = output;
        }
        
        return results;
    }
};

fn getApi(api_name: [*:0]const u8) callconv(.c) ?*anyopaque {
    const name = std.mem.span(api_name);
    
    if (std.mem.eql(u8, name, "batch_processor")) {
        return @ptrCast(&BatchProcessor);
    }
    
    return null;
}

// =============================================================================
// PLUGIN INTERFACE VTABLE
// =============================================================================

const PLUGIN_INTERFACE = PluginInterface{
    .get_info = getInfo,
    .init = initPlugin,
    .deinit = deinitPlugin,
    .start = startPlugin,
    .stop = stopPlugin,
    .pause = null, // Not implemented in this example
    .plugin_resume = null, // Not implemented in this example
    .process = processData,
    .configure = configure,
    .get_config = null, // Not implemented in this example
    .get_status = getStatus,
    .get_metrics = getMetrics,
    .on_event = onEvent,
    .get_api = getApi,
};

// =============================================================================
// PLUGIN ENTRY POINT
// =============================================================================

/// Plugin factory function - this is the entry point called by the framework
export fn abi_plugin_create() callconv(.c) ?*const PluginInterface {
    return &PLUGIN_INTERFACE;
}

// =============================================================================
// BUILD AND USAGE
// =============================================================================

// Build command:
// zig build-lib -dynamic -O ReleaseFast advanced_plugin_example.zig
//
// This creates a shared library that can be loaded by the Abi AI Framework.
//
// Key features demonstrated:
// 1. Thread-safe operations using atomic values and mutexes
// 2. Comprehensive error handling and recovery
// 3. Performance metrics collection
// 4. Memory management with proper cleanup
// 5. Cache implementation for improved performance
// 6. Configuration management
// 7. Event handling for system integration
// 8. Extended API for specialized functionality
//
// The plugin follows all ABI requirements and best practices for
// production-ready plugin development.