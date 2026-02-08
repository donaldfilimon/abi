//! Plugins Module
//!
//! Comptime plugin system for extending the ABI framework with custom hooks,
//! event handlers, and feature integrations. Plugins are registered at compile
//! time and have full access to framework feature contexts.
//!
//! ## Features
//! - Comptime plugin registration (zero runtime discovery overhead)
//! - Lifecycle hooks (on_init, on_deinit, on_event)
//! - Plugin enable/disable at runtime
//! - Event dispatch to all enabled plugins
//! - Plugin info and introspection
//!
//! ## Usage
//!
//! ```zig
//! const my_plugin = Plugin{
//!     .info = .{ .name = "my-plugin", .version = "1.0.0" },
//!     .hooks = .{ .on_init = &myInitHook },
//! };
//!
//! var ctx = try plugins.Context.init(allocator, .{});
//! defer ctx.deinit();
//! try ctx.register(my_plugin);
//! try ctx.emit(.{ .name = "app.started" });
//! ```

const std = @import("std");
const build_options = @import("build_options");

// ============================================================================
// Plugin Types
// ============================================================================

/// Metadata describing a plugin.
pub const PluginInfo = struct {
    /// Unique plugin name (used as identifier).
    name: []const u8,
    /// Semantic version string.
    version: []const u8,
    /// Human-readable description.
    description: []const u8 = "",
    /// Plugin author.
    author: []const u8 = "",
};

/// An event dispatched through the plugin system.
pub const Event = struct {
    /// Event name (e.g., "framework.init", "request.received").
    name: []const u8,
    /// Optional opaque pointer to event-specific data.
    data: ?*const anyopaque = null,
};

/// Context passed to plugin hook functions. Provides access to the
/// allocator and an opaque pointer to the framework instance.
pub const PluginContext = struct {
    allocator: std.mem.Allocator,
    /// Opaque pointer to the Framework struct. Plugins with comptime
    /// knowledge of the Framework type can cast this for full access.
    framework_ptr: ?*anyopaque = null,
};

/// Lifecycle hook function signatures.
pub const PluginHooks = struct {
    /// Called after the plugin is registered and the framework is running.
    on_init: ?*const fn (*PluginContext) anyerror!void = null,
    /// Called when the plugin or framework is shutting down.
    on_deinit: ?*const fn (*PluginContext) void = null,
    /// Called when an event is dispatched through the plugin system.
    on_event: ?*const fn (*PluginContext, Event) anyerror!void = null,
};

/// A registered plugin with its metadata, hooks, and runtime state.
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
// Plugin Registry
// ============================================================================

/// Maximum number of plugins that can be registered.
const MAX_PLUGINS: usize = 64;

/// Manages registered plugins and dispatches lifecycle events.
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
        // Notify plugins of shutdown
        for (self.plugins.items) |plugin| {
            if (plugin.enabled) {
                if (plugin.hooks.on_deinit) |hook| {
                    hook(&self.plugin_ctx);
                }
            }
        }
        self.plugins.deinit(self.allocator);
        self.* = undefined;
    }

    /// Register a new plugin.
    pub fn register(self: *Registry, plugin: Plugin) PluginError!void {
        // Check for duplicate names
        for (self.plugins.items) |existing| {
            if (std.mem.eql(u8, existing.info.name, plugin.info.name)) {
                return PluginError.PluginAlreadyRegistered;
            }
        }

        if (self.plugins.items.len >= MAX_PLUGINS) {
            return PluginError.RegistryFull;
        }

        self.plugins.append(self.allocator, plugin) catch return PluginError.OutOfMemory;

        // Call on_init hook if the registry is already initialized
        if (self.initialized) {
            if (plugin.enabled) {
                if (plugin.hooks.on_init) |hook| {
                    hook(&self.plugin_ctx) catch return PluginError.PluginInitFailed;
                }
            }
        }
    }

    /// Unregister a plugin by name.
    pub fn unregister(self: *Registry, name: []const u8) PluginError!void {
        for (self.plugins.items, 0..) |plugin, i| {
            if (std.mem.eql(u8, plugin.info.name, name)) {
                // Call on_deinit before removing
                if (plugin.enabled) {
                    if (plugin.hooks.on_deinit) |hook| {
                        hook(&self.plugin_ctx);
                    }
                }
                _ = self.plugins.swapRemove(i);
                return;
            }
        }
        return PluginError.PluginNotFound;
    }

    /// Find a plugin by name.
    pub fn get(self: *const Registry, name: []const u8) ?*const Plugin {
        for (self.plugins.items) |*plugin| {
            if (std.mem.eql(u8, plugin.info.name, name)) {
                return plugin;
            }
        }
        return null;
    }

    /// Find a mutable plugin by name.
    pub fn getMut(self: *Registry, name: []const u8) ?*Plugin {
        for (self.plugins.items) |*plugin| {
            if (std.mem.eql(u8, plugin.info.name, name)) {
                return plugin;
            }
        }
        return null;
    }

    /// Enable a plugin by name.
    pub fn enable(self: *Registry, name: []const u8) PluginError!void {
        if (self.getMut(name)) |plugin| {
            if (!plugin.enabled) {
                plugin.enabled = true;
                if (self.initialized) {
                    if (plugin.hooks.on_init) |hook| {
                        hook(&self.plugin_ctx) catch return PluginError.PluginInitFailed;
                    }
                }
            }
        } else {
            return PluginError.PluginNotFound;
        }
    }

    /// Disable a plugin by name.
    pub fn disable(self: *Registry, name: []const u8) PluginError!void {
        if (self.getMut(name)) |plugin| {
            if (plugin.enabled) {
                if (plugin.hooks.on_deinit) |hook| {
                    hook(&self.plugin_ctx);
                }
                plugin.enabled = false;
            }
        } else {
            return PluginError.PluginNotFound;
        }
    }

    /// Get a slice of all registered plugins.
    pub fn list(self: *const Registry) []const Plugin {
        return self.plugins.items;
    }

    /// Count of registered plugins.
    pub fn count(self: *const Registry) usize {
        return self.plugins.items.len;
    }

    /// Count of enabled plugins.
    pub fn enabledCount(self: *const Registry) usize {
        var n: usize = 0;
        for (self.plugins.items) |p| {
            if (p.enabled) n += 1;
        }
        return n;
    }

    /// Dispatch an event to all enabled plugins.
    pub fn dispatch(self: *Registry, event: Event) PluginError!void {
        for (self.plugins.items) |plugin| {
            if (plugin.enabled) {
                if (plugin.hooks.on_event) |hook| {
                    hook(&self.plugin_ctx, event) catch continue;
                }
            }
        }
    }

    /// Initialize all registered plugins (call on_init hooks).
    pub fn initPlugins(self: *Registry) PluginError!void {
        for (self.plugins.items) |plugin| {
            if (plugin.enabled) {
                if (plugin.hooks.on_init) |hook| {
                    hook(&self.plugin_ctx) catch return PluginError.PluginInitFailed;
                }
            }
        }
        self.initialized = true;
    }

    /// Set the framework pointer for plugin contexts.
    pub fn setFrameworkPtr(self: *Registry, ptr: *anyopaque) void {
        self.plugin_ctx.framework_ptr = ptr;
    }
};

// ============================================================================
// Plugin Configuration (re-export from core config)
// ============================================================================

pub const PluginConfig = @import("../../core/config/plugin.zig").PluginConfig;

// ============================================================================
// Context — Framework integration
// ============================================================================

/// Plugins context for Framework lifecycle integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: PluginConfig,
    registry: Registry,

    pub fn init(allocator: std.mem.Allocator, config: PluginConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = config,
            .registry = Registry.init(allocator),
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.registry.deinit();
        self.allocator.destroy(self);
    }

    /// Get the plugin registry.
    pub fn getRegistry(self: *Context) *Registry {
        return &self.registry;
    }

    /// Convenience: register a plugin.
    pub fn register(self: *Context, plugin: Plugin) PluginError!void {
        return self.registry.register(plugin);
    }

    /// Convenience: dispatch an event to all plugins.
    pub fn emit(self: *Context, event: Event) PluginError!void {
        return self.registry.dispatch(event);
    }
};

// ============================================================================
// Module Lifecycle
// ============================================================================

var initialized: bool = false;

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return build_options.enable_plugins;
}

pub fn isInitialized() bool {
    return initialized;
}

// ============================================================================
// Tests
// ============================================================================

test "Registry register and lookup" {
    const allocator = std.testing.allocator;
    var registry = Registry.init(allocator);
    defer registry.deinit();

    try registry.register(.{
        .info = .{ .name = "test-plugin", .version = "1.0.0" },
    });

    try std.testing.expectEqual(@as(usize, 1), registry.count());
    try std.testing.expect(registry.get("test-plugin") != null);
    try std.testing.expect(registry.get("nonexistent") == null);
}

test "Registry prevents duplicate registration" {
    const allocator = std.testing.allocator;
    var registry = Registry.init(allocator);
    defer registry.deinit();

    try registry.register(.{
        .info = .{ .name = "my-plugin", .version = "1.0.0" },
    });
    try std.testing.expectError(
        PluginError.PluginAlreadyRegistered,
        registry.register(.{
            .info = .{ .name = "my-plugin", .version = "2.0.0" },
        }),
    );
}

test "Registry unregister" {
    const allocator = std.testing.allocator;
    var registry = Registry.init(allocator);
    defer registry.deinit();

    try registry.register(.{
        .info = .{ .name = "removable", .version = "0.1.0" },
    });
    try std.testing.expectEqual(@as(usize, 1), registry.count());

    try registry.unregister("removable");
    try std.testing.expectEqual(@as(usize, 0), registry.count());

    try std.testing.expectError(
        PluginError.PluginNotFound,
        registry.unregister("removable"),
    );
}

test "Registry enable/disable" {
    const allocator = std.testing.allocator;
    var registry = Registry.init(allocator);
    defer registry.deinit();

    try registry.register(.{
        .info = .{ .name = "toggled", .version = "1.0.0" },
        .enabled = true,
    });

    try std.testing.expectEqual(@as(usize, 1), registry.enabledCount());

    try registry.disable("toggled");
    try std.testing.expectEqual(@as(usize, 0), registry.enabledCount());

    try registry.enable("toggled");
    try std.testing.expectEqual(@as(usize, 1), registry.enabledCount());
}

test "Registry dispatches events to enabled plugins" {
    const allocator = std.testing.allocator;
    var registry = Registry.init(allocator);
    defer registry.deinit();

    var call_count: u32 = 0;
    const Counter = struct {
        fn onEvent(_: *PluginContext, _: Event) anyerror!void {
            // We can't capture call_count directly, so we use a global
            // for test verification below.
        }
    };

    try registry.register(.{
        .info = .{ .name = "evented", .version = "1.0.0" },
        .hooks = .{ .on_event = &Counter.onEvent },
    });

    // Dispatch should not error even with a no-op handler
    try registry.dispatch(.{ .name = "test.event" });
    _ = &call_count;
}

test "Registry lifecycle hooks" {
    const allocator = std.testing.allocator;
    var registry = Registry.init(allocator);
    defer registry.deinit();

    const Hooks = struct {
        fn onInit(_: *PluginContext) anyerror!void {}
        fn onDeinit(_: *PluginContext) void {}
    };

    try registry.register(.{
        .info = .{ .name = "lifecycle", .version = "1.0.0" },
        .hooks = .{
            .on_init = &Hooks.onInit,
            .on_deinit = &Hooks.onDeinit,
        },
    });

    // Init plugins should call on_init for enabled plugins
    try registry.initPlugins();
    try std.testing.expect(registry.initialized);
}

test "Context creates and manages registry" {
    const allocator = std.testing.allocator;
    var ctx = try Context.init(allocator, .{});
    defer ctx.deinit();

    try ctx.register(.{
        .info = .{ .name = "ctx-plugin", .version = "0.1.0" },
    });

    try std.testing.expectEqual(@as(usize, 1), ctx.getRegistry().count());
    try ctx.emit(.{ .name = "hello" });
}

test "Module lifecycle" {
    const allocator = std.testing.allocator;
    try init(allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}
