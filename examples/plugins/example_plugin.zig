//! Example Plugin for Abi AI Framework
//!
//! This example demonstrates how to create a plugin that can be dynamically loaded
//! into the Abi AI Framework. It implements a simple text processor plugin.

const std = @import("std");

// Note: In a real plugin, you would import these from the framework
// For this example, we define the types locally
const PluginInfo = struct {
    name: []const u8,
    version: PluginVersion,
    author: []const u8,
    description: []const u8,
    plugin_type: PluginType,
    abi_version: PluginVersion,
    dependencies: []const []const u8 = &.{},
    provides: []const []const u8 = &.{},
    requires: []const []const u8 = &.{},
    license: ?[]const u8 = null,
    homepage: ?[]const u8 = null,
    repository: ?[]const u8 = null,

    pub fn isCompatible(self: PluginInfo, framework_abi: PluginVersion) bool {
        return self.abi_version.isCompatible(framework_abi);
    }
};

const PluginVersion = struct {
    major: u32,
    minor: u32,
    patch: u32,
    pre_release: ?[]const u8 = null,

    pub fn init(major: u32, minor: u32, patch: u32) PluginVersion {
        return .{ .major = major, .minor = minor, .patch = patch };
    }

    pub fn isCompatible(self: PluginVersion, required: PluginVersion) bool {
        if (self.major != required.major) return false;
        if (self.minor < required.minor) return false;
        return true;
    }
};

const PluginType = enum {
    text_processor,
    neural_network,
    vector_database,
    custom,
    // ... other types
};

const PluginConfig = struct {
    enabled: bool = true,
    auto_load: bool = true,
    priority: i32 = 0,
    max_memory_mb: ?u32 = null,
    max_cpu_time_ms: ?u32 = null,
    sandboxed: bool = false,
    permissions: []const []const u8 = &.{},
    parameters: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator) PluginConfig {
        return .{
            .parameters = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *PluginConfig) void {
        self.parameters.deinit();
    }
};

const PluginContext = struct {
    allocator: std.mem.Allocator,
    logger: ?*anyopaque = null,
    config: *PluginConfig,
    log_fn: ?*const fn (level: u8, message: []const u8) void = null,
    get_service_fn: ?*const fn (service_name: []const u8) ?*anyopaque = null,

    pub fn log(self: *PluginContext, level: u8, message: []const u8) void {
        if (self.log_fn) |log_func| {
            log_func(level, message);
        }
    }
};

const PluginInterface = extern struct {
    get_info: *const fn () callconv(.c) *const PluginInfo,
    init: *const fn (context: *PluginContext) callconv(.c) c_int,
    deinit: *const fn (context: *PluginContext) callconv(.c) void,
    start: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    stop: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    pause: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    plugin_resume: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    process: ?*const fn (context: *PluginContext, input: ?*anyopaque, output: ?*anyopaque) callconv(.c) c_int = null,
    configure: ?*const fn (context: *PluginContext, config: *const PluginConfig) callconv(.c) c_int = null,
    get_config: ?*const fn (context: *PluginContext) callconv(.c) ?*const PluginConfig = null,
    get_status: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    get_metrics: ?*const fn (context: *PluginContext, buffer: [*]u8, buffer_size: usize) callconv(.c) c_int = null,
    on_event: ?*const fn (context: *PluginContext, event_type: u32, event_data: ?*anyopaque) callconv(.c) c_int = null,
    get_api: ?*const fn (api_name: [*:0]const u8) callconv(.c) ?*anyopaque = null,
};

// =============================================================================
// PLUGIN IMPLEMENTATION
// =============================================================================

/// Plugin state
var plugin_initialized = false;
var plugin_running = false;
var processed_count: u64 = 0;

/// Plugin information
const PLUGIN_INFO = PluginInfo{
    .name = "example_text_processor",
    .version = PluginVersion.init(1, 0, 0),
    .author = "Abi AI Framework Team",
    .description = "Example text processing plugin that demonstrates the plugin API",
    .plugin_type = .text_processor,
    .abi_version = PluginVersion.init(1, 0, 0),
    .provides = &[_][]const u8{ "text_uppercase", "text_lowercase", "text_reverse" },
    .license = "MIT",
    .homepage = "https://github.com/abi-ai/plugins",
    .repository = "https://github.com/abi-ai/plugins/tree/main/examples",
};

/// Text processing input/output structures
const TextInput = struct {
    text: []const u8,
    operation: []const u8, // "uppercase", "lowercase", "reverse"
};

const TextOutput = struct {
    result: []u8,
    length: usize,
};

// =============================================================================
// PLUGIN INTERFACE IMPLEMENTATION
// =============================================================================

/// Get plugin information
fn getInfo() callconv(.c) *const PluginInfo {
    return &PLUGIN_INFO;
}

/// Initialize the plugin
fn initPlugin(context: *PluginContext) callconv(.c) c_int {
    if (plugin_initialized) {
        return -1; // Already initialized
    }

    context.log(1, "Initializing example text processor plugin");
    plugin_initialized = true;
    processed_count = 0;

    return 0; // Success
}

/// Deinitialize the plugin
fn deinitPlugin(context: *PluginContext) callconv(.c) void {
    if (!plugin_initialized) return;

    context.log(1, "Deinitializing example text processor plugin");
    plugin_initialized = false;
    plugin_running = false;
}

/// Start the plugin
fn startPlugin(context: *PluginContext) callconv(.c) c_int {
    if (!plugin_initialized) {
        return -1; // Not initialized
    }

    if (plugin_running) {
        return -2; // Already running
    }

    context.log(1, "Starting example text processor plugin");
    plugin_running = true;

    return 0; // Success
}

/// Stop the plugin
fn stopPlugin(context: *PluginContext) callconv(.c) c_int {
    if (!plugin_running) {
        return -1; // Not running
    }

    context.log(1, "Stopping example text processor plugin");
    plugin_running = false;

    return 0; // Success
}

/// Process text data
fn processText(context: *PluginContext, input: ?*anyopaque, output: ?*anyopaque) callconv(.c) c_int {
    if (!plugin_running) {
        return -1; // Plugin not running
    }

    const text_input: *TextInput = @ptrCast(@alignCast(input orelse return -2));
    const text_output: *TextOutput = @ptrCast(@alignCast(output orelse return -2));

    // Allocate output buffer (caller must free)
    text_output.result = context.allocator.alloc(u8, text_input.text.len) catch return -3;
    text_output.length = text_input.text.len;

    // Process based on operation
    if (std.mem.eql(u8, text_input.operation, "uppercase")) {
        for (text_input.text, 0..) |char, i| {
            text_output.result[i] = std.ascii.toUpper(char);
        }
    } else if (std.mem.eql(u8, text_input.operation, "lowercase")) {
        for (text_input.text, 0..) |char, i| {
            text_output.result[i] = std.ascii.toLower(char);
        }
    } else if (std.mem.eql(u8, text_input.operation, "reverse")) {
        for (text_input.text, 0..) |char, i| {
            text_output.result[text_input.text.len - 1 - i] = char;
        }
    } else {
        // Unknown operation, just copy
        @memcpy(text_output.result, text_input.text);
    }

    processed_count += 1;

    return 0; // Success
}

/// Get plugin status
fn getStatus(context: *PluginContext) callconv(.c) c_int {
    _ = context;

    if (!plugin_initialized) return 0; // Uninitialized
    if (!plugin_running) return 1; // Initialized but not running
    return 2; // Running
}

/// Get plugin metrics
fn getMetrics(context: *PluginContext, buffer: [*]u8, buffer_size: usize) callconv(.c) c_int {
    _ = context;

    const metrics = std.fmt.bufPrint(buffer[0..buffer_size], "{{\"processed_count\":{d},\"status\":\"{s}\",\"version\":\"1.0.0\"}}", .{ processed_count, if (plugin_running) "running" else "stopped" }) catch return -1;

    return @intCast(metrics.len);
}

/// Handle events
fn onEvent(context: *PluginContext, event_type: u32, event_data: ?*anyopaque) callconv(.c) c_int {
    _ = event_data;

    switch (event_type) {
        1 => { // System startup
            context.log(1, "Received system startup event");
        },
        2 => { // System shutdown
            context.log(1, "Received system shutdown event");
        },
        else => {
            context.log(2, "Received unknown event");
        },
    }

    return 0; // Success
}

/// Get plugin-specific API
fn getApi(api_name: [*:0]const u8) callconv(.c) ?*anyopaque {
    const name = std.mem.span(api_name);

    if (std.mem.eql(u8, name, "text_processor")) {
        // Return a function pointer to our text processing API
        return @ptrCast(&processText);
    }

    return null; // API not found
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
    .process = processText,
    .get_status = getStatus,
    .get_metrics = getMetrics,
    .on_event = onEvent,
    .get_api = getApi,
};

// =============================================================================
// PLUGIN ENTRY POINT
// =============================================================================

/// Plugin factory function - this is the entry point called by the framework
export fn abi_plugin_create() ?*const PluginInterface {
    return &PLUGIN_INTERFACE;
}

// =============================================================================
// BUILDING INSTRUCTIONS
// =============================================================================

// To build this plugin as a shared library:
//
// Windows: zig build-lib -dynamic example_plugin.zig
// Linux:   zig build-lib -dynamic example_plugin.zig
// macOS:   zig build-lib -dynamic example_plugin.zig
//
// This will create:
// - Windows: example_plugin.dll
// - Linux:   libexample_plugin.so
// - macOS:   libexample_plugin.dylib
//
// The resulting library can be loaded by the Abi AI Framework using:
//
// ```zig
// var registry = try plugins.init(allocator);
// defer registry.deinit();
//
// try registry.addPluginPath("./plugins");
// try registry.loadPlugin("./plugins/example_plugin.dll"); // or .so/.dylib
// try registry.initializePlugin("example_text_processor", null);
// try registry.startPlugin("example_text_processor");
//
// // Use the plugin
// const plugin = registry.getPlugin("example_text_processor");
// var input = TextInput{ .text = "Hello World", .operation = "uppercase" };
// var output: TextOutput = undefined;
// try plugin.process(&input, &output);
// // output.result now contains "HELLO WORLD"
// ```
