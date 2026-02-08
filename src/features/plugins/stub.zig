//! Plugins Stub Module
//!
//! API-compatible no-op implementations when plugins feature is disabled.
//! All operations return `PluginsDisabled` or defaults.
//!
//! To enable the real implementation, build with `-Denable-plugins=true`.

const std = @import("std");

// ============================================================================
// Plugin Types (matching mod.zig signatures)
// ============================================================================

pub const PluginInfo = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8 = "",
    author: []const u8 = "",
};

pub const Event = struct {
    name: []const u8,
    data: ?*const anyopaque = null,
};

pub const PluginContext = struct {
    allocator: std.mem.Allocator,
    framework_ptr: ?*anyopaque = null,
};

pub const PluginHooks = struct {
    on_init: ?*const fn (*PluginContext) anyerror!void = null,
    on_deinit: ?*const fn (*PluginContext) void = null,
    on_event: ?*const fn (*PluginContext, Event) anyerror!void = null,
};

pub const Plugin = struct {
    info: PluginInfo,
    hooks: PluginHooks = .{},
    enabled: bool = true,
};

pub const PluginError = error{
    PluginsDisabled,
    PluginNotFound,
    PluginAlreadyRegistered,
    PluginInitFailed,
    RegistryFull,
    OutOfMemory,
};

// ============================================================================
// Stub Registry
// ============================================================================

pub const Registry = struct {
    allocator: std.mem.Allocator,
    plugins: std.ArrayListUnmanaged(Plugin) = .empty,
    plugin_ctx: PluginContext,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .allocator = allocator,
            .plugin_ctx = .{ .allocator = allocator },
        };
    }

    pub fn deinit(self: *Registry) void {
        _ = self;
    }

    pub fn register(_: *Registry, _: Plugin) PluginError!void {
        return PluginError.PluginsDisabled;
    }

    pub fn unregister(_: *Registry, _: []const u8) PluginError!void {
        return PluginError.PluginsDisabled;
    }

    pub fn get(_: *const Registry, _: []const u8) ?*const Plugin {
        return null;
    }

    pub fn getMut(_: *Registry, _: []const u8) ?*Plugin {
        return null;
    }

    pub fn enable(_: *Registry, _: []const u8) PluginError!void {
        return PluginError.PluginsDisabled;
    }

    pub fn disable(_: *Registry, _: []const u8) PluginError!void {
        return PluginError.PluginsDisabled;
    }

    pub fn list(_: *const Registry) []const Plugin {
        return &.{};
    }

    pub fn count(_: *const Registry) usize {
        return 0;
    }

    pub fn enabledCount(_: *const Registry) usize {
        return 0;
    }

    pub fn dispatch(_: *Registry, _: Event) PluginError!void {
        return PluginError.PluginsDisabled;
    }

    pub fn initPlugins(_: *Registry) PluginError!void {
        return PluginError.PluginsDisabled;
    }

    pub fn setFrameworkPtr(_: *Registry, _: *anyopaque) void {}
};

// ============================================================================
// Config re-export
// ============================================================================

pub const PluginConfig = @import("../../core/config/plugin.zig").PluginConfig;

// ============================================================================
// Stub Context
// ============================================================================

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: PluginConfig,
    registry: Registry,

    pub fn init(allocator: std.mem.Allocator, config: PluginConfig) !*Context {
        _ = allocator;
        _ = config;
        return PluginError.PluginsDisabled;
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }

    pub fn getRegistry(_: *Context) *Registry {
        unreachable;
    }

    pub fn register(_: *Context, _: Plugin) PluginError!void {
        return PluginError.PluginsDisabled;
    }

    pub fn emit(_: *Context, _: Event) PluginError!void {
        return PluginError.PluginsDisabled;
    }
};

// ============================================================================
// Module Lifecycle
// ============================================================================

var initialized: bool = false;

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    return PluginError.PluginsDisabled;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return initialized;
}
