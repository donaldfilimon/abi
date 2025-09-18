//! Plugin Interface Definition
//!
//! This module defines the standard interface that all plugins must implement.
//! It uses C-compatible function pointers to ensure cross-language compatibility.

const std = @import("std");
const types = @import("types.zig");

const PluginInfo = types.PluginInfo;
const PluginConfig = types.PluginConfig;
const PluginContext = types.PluginContext;
const PluginError = types.PluginError;
const PluginState = types.PluginState;

/// Standard plugin interface using C-compatible function pointers
/// This vtable approach ensures compatibility across different compilation units
pub const PluginInterface = extern struct {
    // Required functions (must be implemented by all plugins)
    get_info: *const fn () callconv(.c) *const PluginInfo,
    init: *const fn (context: *PluginContext) callconv(.c) c_int,
    deinit: *const fn (context: *PluginContext) callconv(.c) void,

    // Lifecycle functions
    start: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    stop: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    pause: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    plugin_resume: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,

    // Generic processing function
    process: ?*const fn (context: *PluginContext, input: ?*anyopaque, output: ?*anyopaque) callconv(.c) c_int = null,

    // Configuration functions
    configure: ?*const fn (context: *PluginContext, config: *const PluginConfig) callconv(.c) c_int = null,
    get_config: ?*const fn (context: *PluginContext) callconv(.c) ?*const PluginConfig = null,

    // Status and diagnostics
    get_status: ?*const fn (context: *PluginContext) callconv(.c) c_int = null,
    get_metrics: ?*const fn (context: *PluginContext, buffer: [*]u8, buffer_size: usize) callconv(.c) c_int = null,

    // Event handling
    on_event: ?*const fn (context: *PluginContext, event_type: u32, event_data: ?*anyopaque) callconv(.c) c_int = null,

    // Plugin-specific API (type-safe through casting)
    get_api: ?*const fn (api_name: [*:0]const u8) callconv(.c) ?*anyopaque = null,

    pub fn isValid(self: *const PluginInterface) bool {
        _ = self;
        return true; // In this version, all fields are mandatory so always valid
    }
};

/// Plugin wrapper that provides a safer Zig API around the C interface
pub const Plugin = struct {
    interface: *const PluginInterface,
    handle: ?*anyopaque = null, // Platform-specific library handle
    state: PluginState = .unloaded,
    info: ?*const PluginInfo = null,
    context: ?*PluginContext = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, interface: *const PluginInterface) !Plugin {
        if (!interface.isValid()) {
            return PluginError.InvalidPlugin;
        }

        return Plugin{
            .interface = interface,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Plugin) void {
        if (self.context) |ctx| {
            if (self.state == .running or self.state == .initialized) {
                _ = self.interface.deinit(ctx);
            }
            self.allocator.destroy(ctx);
        }
        self.state = .unloaded;
    }

    pub fn getInfo(self: *Plugin) *const PluginInfo {
        if (self.info == null) {
            self.info = self.interface.get_info();
        }
        return self.info.?;
    }

    pub fn initialize(self: *Plugin, config: *PluginConfig) !void {
        // Allow initialization from unloaded or loaded state
        if (self.state != .unloaded and self.state != .loaded and !self.state.canTransitionTo(.initializing)) {
            return PluginError.InvalidParameters;
        }

        self.state = .initializing;

        // Create context
        if (self.context == null) {
            self.context = try self.allocator.create(PluginContext);
            self.context.?.* = PluginContext{
                .allocator = self.allocator,
                .config = config,
            };
        }

        const result = self.interface.init(self.context.?);
        if (result != 0) {
            self.state = .error_state;
            return PluginError.InitializationFailed;
        }

        self.state = .initialized;
    }

    pub fn start(self: *Plugin) !void {
        if (!self.state.canTransitionTo(.running)) {
            return PluginError.InvalidParameters;
        }

        if (self.interface.start) |start_fn| {
            const result = start_fn(self.context.?);
            if (result != 0) {
                self.state = .error_state;
                return PluginError.ExecutionFailed;
            }
        }

        self.state = .running;
    }

    pub fn stop(self: *Plugin) !void {
        if (!self.state.canTransitionTo(.stopping)) {
            return PluginError.InvalidParameters;
        }

        self.state = .stopping;

        if (self.interface.stop) |stop_fn| {
            const result = stop_fn(self.context.?);
            if (result != 0) {
                self.state = .error_state;
                return PluginError.ExecutionFailed;
            }
        }

        self.state = .stopped;
    }

    pub fn pause(self: *Plugin) !void {
        if (!self.state.canTransitionTo(.paused)) {
            return PluginError.InvalidParameters;
        }

        if (self.interface.pause) |pause_fn| {
            const result = pause_fn(self.context.?);
            if (result != 0) {
                return PluginError.ExecutionFailed;
            }
        }

        self.state = .paused;
    }

    pub fn resumePlugin(self: *Plugin) !void {
        if (!self.state.canTransitionTo(.running)) {
            return PluginError.InvalidParameters;
        }

        if (self.interface.plugin_resume) |resume_fn| {
            const result = resume_fn(self.context.?);
            if (result != 0) {
                return PluginError.ExecutionFailed;
            }
        }

        self.state = .running;
    }

    pub fn process(self: *Plugin, input: ?*anyopaque, output: ?*anyopaque) !void {
        if (self.state != .running) {
            return PluginError.InvalidParameters;
        }

        if (self.interface.process) |process_fn| {
            const result = process_fn(self.context.?, input, output);
            if (result != 0) {
                return PluginError.ExecutionFailed;
            }
        }
    }

    pub fn configure(self: *Plugin, config: *const PluginConfig) !void {
        if (self.interface.configure) |configure_fn| {
            const result = configure_fn(self.context.?, config);
            if (result != 0) {
                return PluginError.InvalidParameters;
            }
        }
    }

    pub fn getStatus(self: *Plugin) i32 {
        if (self.interface.get_status) |get_status_fn| {
            return get_status_fn(self.context.?);
        }
        return @intFromEnum(self.state);
    }

    pub fn getMetrics(self: *Plugin, buffer: []u8) !usize {
        if (self.interface.get_metrics) |get_metrics_fn| {
            const result = get_metrics_fn(self.context.?, buffer.ptr, buffer.len);
            if (result < 0) {
                return PluginError.ExecutionFailed;
            }
            return @intCast(result);
        }
        return 0;
    }

    pub fn onEvent(self: *Plugin, event_type: u32, event_data: ?*anyopaque) !void {
        if (self.interface.on_event) |on_event_fn| {
            const result = on_event_fn(self.context.?, event_type, event_data);
            if (result != 0) {
                return PluginError.ExecutionFailed;
            }
        }
    }

    pub fn getApi(self: *Plugin, api_name: [:0]const u8) ?*anyopaque {
        if (self.interface.get_api) |get_api_fn| {
            return get_api_fn(api_name.ptr);
        }
        return null;
    }

    pub fn getState(self: *const Plugin) PluginState {
        return self.state;
    }

    pub fn setState(self: *Plugin, new_state: PluginState) !void {
        if (!self.state.canTransitionTo(new_state)) {
            return PluginError.InvalidParameters;
        }
        self.state = new_state;
    }
};

/// Plugin factory function type
pub const PluginFactoryFn = *const fn () callconv(.c) ?*const PluginInterface;

/// Standard plugin entry point function name
pub const PLUGIN_ENTRY_POINT = "abi_plugin_create";

/// ABI version for plugin compatibility
pub const PLUGIN_ABI_VERSION = types.PluginVersion.init(1, 0, 0);

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Create a plugin from a loaded interface
pub fn createPlugin(allocator: std.mem.Allocator, interface: *const PluginInterface) !*Plugin {
    const plugin = try allocator.create(Plugin);
    plugin.* = try Plugin.init(allocator, interface);
    return plugin;
}

/// Destroy a plugin instance
pub fn destroyPlugin(allocator: std.mem.Allocator, plugin: *Plugin) void {
    plugin.deinit();
    allocator.destroy(plugin);
}

// =============================================================================
// TESTS
// =============================================================================

// Mock plugin interface for testing
const MockPluginInfo = PluginInfo{
    .name = "test_plugin",
    .version = types.PluginVersion.init(1, 0, 0),
    .author = "Test Author",
    .description = "Test plugin for unit testing",
    .plugin_type = .custom,
    .abi_version = PLUGIN_ABI_VERSION,
};

fn mockGetInfo() callconv(.c) *const PluginInfo {
    return &MockPluginInfo;
}

fn mockInit(context: *PluginContext) callconv(.c) c_int {
    _ = context;
    return 0; // Success
}

fn mockDeinit(context: *PluginContext) callconv(.c) void {
    _ = context;
}

const MockInterface = PluginInterface{
    .get_info = mockGetInfo,
    .init = mockInit,
    .deinit = mockDeinit,
};

test "Plugin interface validation" {
    try std.testing.expect(MockInterface.isValid());

    // Since all fields are mandatory in extern struct, we test basic validation
    var plugin = try Plugin.init(std.testing.allocator, &MockInterface);
    defer plugin.deinit();

    try std.testing.expect(MockInterface.isValid());
}

test "Plugin lifecycle" {
    var plugin = try Plugin.init(std.testing.allocator, &MockInterface);
    defer plugin.deinit();

    const info = plugin.getInfo();
    try std.testing.expectEqualStrings("test_plugin", info.name);

    var config = types.PluginConfig.init(std.testing.allocator);
    defer config.deinit();

    try plugin.initialize(&config);
    try std.testing.expectEqual(types.PluginState.initialized, plugin.getState());

    try plugin.start();
    try std.testing.expectEqual(types.PluginState.running, plugin.getState());

    try plugin.stop();
    try std.testing.expectEqual(types.PluginState.stopped, plugin.getState());
}
