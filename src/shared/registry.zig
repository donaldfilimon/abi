//! Plugin Registry
//!
//! This module provides a centralized registry for managing loaded plugins,
//! handling dependencies, and providing a unified interface for plugin operations.

const std = @import("std");
const types = @import("types.zig");
const interface = @import("interface.zig");
const loader = @import("loader.zig");

const PluginError = types.PluginError;
const PluginType = types.PluginType;
const PluginState = types.PluginState;
const PluginInfo = types.PluginInfo;
const PluginConfig = types.PluginConfig;
const Plugin = interface.Plugin;
const PluginLoader = loader.PluginLoader;

/// Plugin registry entry
const PluginEntry = struct {
    plugin: *Plugin,
    config: PluginConfig,
    load_order: u32,
    dependencies: std.ArrayListUnmanaged([]u8) = .{},
    dependents: std.ArrayListUnmanaged([]u8) = .{},

    pub fn init(allocator: std.mem.Allocator, plugin: *Plugin, load_order: u32) PluginEntry {
        return .{
            .plugin = plugin,
            .config = PluginConfig.init(allocator),
            .load_order = load_order,
        };
    }

    pub fn deinit(self: *PluginEntry, allocator: std.mem.Allocator) void {
        self.config.deinit();
        for (self.dependencies.items) |dep| {
            allocator.free(dep);
        }
        self.dependencies.deinit(allocator);
        for (self.dependents.items) |dep| {
            allocator.free(dep);
        }
        self.dependents.deinit(allocator);
    }
};

/// Centralized plugin registry
pub const PluginRegistry = struct {
    allocator: std.mem.Allocator,
    loader: PluginLoader,
    plugins: std.StringHashMap(PluginEntry),
    load_order_counter: u32 = 0,
    event_handlers: std.ArrayListUnmanaged(EventHandler) = .{},

    const EventHandler = struct {
        event_type: u32,
        handler_fn: *const fn (event_data: ?*anyopaque) void,
    };

    pub fn init(allocator: std.mem.Allocator) !PluginRegistry {
        return .{
            .allocator = allocator,
            .loader = loader.createLoader(allocator),
            .plugins = std.StringHashMap(PluginEntry).init(allocator),
        };
    }

    pub fn deinit(self: *PluginRegistry) void {
        // Stop and unload all plugins
        var iterator = self.plugins.iterator();
        while (iterator.next()) |entry| {
            self.unloadPluginInternal(entry.key_ptr.*) catch {};
        }

        self.plugins.deinit();
        self.loader.deinit();
        self.event_handlers.deinit(self.allocator);
    }

    /// Add a search path for plugins
    pub fn addPluginPath(self: *PluginRegistry, path: []const u8) !void {
        try self.loader.addPluginPath(path);
    }

    /// Discover plugins in search paths
    pub fn discoverPlugins(self: *PluginRegistry) !std.ArrayList([]u8) {
        return try self.loader.discoverPlugins();
    }

    /// Load a plugin from file
    pub fn loadPlugin(self: *PluginRegistry, plugin_path: []const u8) !void {
        // Load the plugin interface
        const plugin_interface = try self.loader.loadPlugin(plugin_path);

        // Create plugin wrapper
        const plugin = try interface.createPlugin(self.allocator, plugin_interface);
        errdefer interface.destroyPlugin(self.allocator, plugin);

        // Get plugin info
        const info = plugin.getInfo();

        // Check if already registered
        if (self.plugins.contains(info.name)) {
            return PluginError.AlreadyRegistered;
        }

        // Create plugin entry
        const load_order = self.load_order_counter;
        self.load_order_counter += 1;

        var entry = PluginEntry.init(self.allocator, plugin, load_order);
        errdefer entry.deinit(self.allocator);

        // Check dependencies
        try self.validateDependencies(info);

        // Register the plugin
        try self.plugins.put(try self.allocator.dupe(u8, info.name), entry);

        // Update dependency graph
        try self.updateDependencyGraph(info.name, info.dependencies);
    }

    /// Unload a plugin
    pub fn unloadPlugin(self: *PluginRegistry, plugin_name: []const u8) !void {
        try self.unloadPluginInternal(plugin_name);
    }

    /// Register a plugin interface that is built into the process (no dynamic library)
    pub fn registerBuiltinInterface(self: *PluginRegistry, plugin_interface: *const interface.PluginInterface) !void {
        if (!plugin_interface.isValid()) {
            return PluginError.InvalidPlugin;
        }

        // Create plugin wrapper
        const plugin = try interface.createPlugin(self.allocator, plugin_interface);
        errdefer interface.destroyPlugin(self.allocator, plugin);

        // Get plugin info
        const info = plugin.getInfo();

        // Check if already registered
        if (self.plugins.contains(info.name)) {
            return PluginError.AlreadyRegistered;
        }

        // Create plugin entry
        const load_order = self.load_order_counter;
        self.load_order_counter += 1;

        var entry = PluginEntry.init(self.allocator, plugin, load_order);
        errdefer entry.deinit(self.allocator);

        // Check dependencies
        try self.validateDependencies(info);

        // Register the plugin
        try self.plugins.put(try self.allocator.dupe(u8, info.name), entry);

        // Update dependency graph
        try self.updateDependencyGraph(info.name, info.dependencies);
    }

    /// Initialize a plugin with configuration
    pub fn initializePlugin(self: *PluginRegistry, plugin_name: []const u8, config: ?*PluginConfig) !void {
        const entry = self.plugins.getPtr(plugin_name) orelse return PluginError.PluginNotFound;

        // Use provided config or default
        const plugin_config = config orelse &entry.config;

        try entry.plugin.initialize(plugin_config);
    }

    /// Start a plugin
    pub fn startPlugin(self: *PluginRegistry, plugin_name: []const u8) !void {
        const entry = self.plugins.get(plugin_name) orelse return PluginError.PluginNotFound;
        try entry.plugin.start();
    }

    /// Stop a plugin
    pub fn stopPlugin(self: *PluginRegistry, plugin_name: []const u8) !void {
        const entry = self.plugins.get(plugin_name) orelse return PluginError.PluginNotFound;
        try entry.plugin.stop();
    }

    /// Start all plugins in dependency order
    pub fn startAllPlugins(self: *PluginRegistry) !void {
        // Get plugins sorted by load order
        var plugin_list = std.ArrayList(*PluginEntry).init(self.allocator);
        defer plugin_list.deinit();

        var iterator = self.plugins.valueIterator();
        while (iterator.next()) |entry| {
            try plugin_list.append(entry);
        }

        // Sort by load order
        std.sort.insertion(*PluginEntry, plugin_list.items, {}, struct {
            fn lessThan(_: void, a: *PluginEntry, b: *PluginEntry) bool {
                return a.load_order < b.load_order;
            }
        }.lessThan);

        // Start plugins in order
        for (plugin_list.items) |entry| {
            try entry.plugin.start();
        }
    }

    /// Stop all plugins in reverse order
    pub fn stopAllPlugins(self: *PluginRegistry) !void {
        // Get plugins sorted by reverse load order
        var plugin_list = std.ArrayList(*PluginEntry).init(self.allocator);
        defer plugin_list.deinit();

        var iterator = self.plugins.valueIterator();
        while (iterator.next()) |entry| {
            try plugin_list.append(entry);
        }

        // Sort by reverse load order
        std.sort.insertion(*PluginEntry, plugin_list.items, {}, struct {
            fn lessThan(_: void, a: *PluginEntry, b: *PluginEntry) bool {
                return a.load_order > b.load_order;
            }
        }.lessThan);

        // Stop plugins in reverse order
        for (plugin_list.items) |entry| {
            try entry.plugin.stop();
        }
    }

    /// Get plugin by name
    pub fn getPlugin(self: *PluginRegistry, plugin_name: []const u8) ?*Plugin {
        const entry = self.plugins.get(plugin_name);
        return if (entry) |e| e.plugin else null;
    }

    /// Get plugins by type
    pub fn getPluginsByType(self: *PluginRegistry, plugin_type: PluginType) !std.ArrayList(*Plugin) {
        var result = std.ArrayList(*Plugin).init(self.allocator);
        errdefer result.deinit();

        var iterator = self.plugins.valueIterator();
        while (iterator.next()) |entry| {
            const info = entry.plugin.getInfo();
            if (info.plugin_type == plugin_type) {
                try result.append(entry.plugin);
            }
        }

        return result;
    }

    /// Get all plugin names
    pub fn getPluginNames(self: *PluginRegistry) !std.ArrayList([]u8) {
        var names = std.ArrayList([]u8).init(self.allocator);
        errdefer {
            for (names.items) |name| {
                self.allocator.free(name);
            }
            names.deinit();
        }

        var iterator = self.plugins.keyIterator();
        while (iterator.next()) |name| {
            const owned = try self.allocator.dupe(u8, name.*);
            errdefer self.allocator.free(owned);
            try names.append(owned);
        }

        return names;
    }

    /// Get plugin count
    pub fn getPluginCount(self: *PluginRegistry) usize {
        return self.plugins.count();
    }

    /// Get plugin information
    pub fn getPluginInfo(self: *PluginRegistry, plugin_name: []const u8) ?*const PluginInfo {
        const entry = self.plugins.get(plugin_name);
        return if (entry) |e| e.plugin.getInfo() else null;
    }

    /// Configure a plugin
    pub fn configurePlugin(self: *PluginRegistry, plugin_name: []const u8, config: *const PluginConfig) !void {
        const entry = self.plugins.get(plugin_name) orelse return PluginError.PluginNotFound;
        try entry.plugin.configure(config);
    }

    /// Send event to all interested plugins
    pub fn broadcastEvent(self: *PluginRegistry, event_type: u32, event_data: ?*anyopaque) !void {
        var iterator = self.plugins.valueIterator();
        while (iterator.next()) |entry| {
            entry.plugin.onEvent(event_type, event_data) catch |err| {
                // Log error but continue with other plugins
                std.log.warn("Plugin {} failed to handle event {}: {}", .{ entry.plugin.getInfo().name, event_type, err });
            };
        }

        // Also call registered event handlers
        for (self.event_handlers.items) |handler| {
            if (handler.event_type == event_type) {
                handler.handler_fn(event_data);
            }
        }
    }

    /// Register an event handler
    pub fn registerEventHandler(self: *PluginRegistry, event_type: u32, handler_fn: *const fn (event_data: ?*anyopaque) void) !void {
        try self.event_handlers.append(self.allocator, .{
            .event_type = event_type,
            .handler_fn = handler_fn,
        });
    }

    /// Internal plugin unloading
    fn unloadPluginInternal(self: *PluginRegistry, plugin_name: []const u8) !void {
        var entry = self.plugins.fetchRemove(plugin_name) orelse return PluginError.PluginNotFound;
        defer {
            entry.value.deinit(self.allocator);
            self.allocator.free(entry.key);
        }

        // Check if any plugins depend on this one
        if (entry.value.dependents.items.len > 0) {
            return PluginError.ConflictingPlugin;
        }

        // Stop the plugin if running
        if (entry.value.plugin.getState() == .running) {
            try entry.value.plugin.stop();
        }

        // Remove from dependency graph
        for (entry.value.dependencies.items) |dep_name| {
            if (self.plugins.getPtr(dep_name)) |dep_entry| {
                for (dep_entry.dependents.items, 0..) |dependent, i| {
                    if (std.mem.eql(u8, dependent, plugin_name)) {
                        const removed = dep_entry.dependents.swapRemove(i);
                        self.allocator.free(removed);
                        break;
                    }
                }
            }
        }

        // Destroy the plugin
        interface.destroyPlugin(self.allocator, entry.value.plugin);
    }

    /// Validate plugin dependencies
    fn validateDependencies(self: *PluginRegistry, info: *const PluginInfo) !void {
        for (info.dependencies) |dep_name| {
            if (!self.plugins.contains(dep_name)) {
                return PluginError.DependencyMissing;
            }
        }
    }

    /// Update dependency graph
    fn updateDependencyGraph(self: *PluginRegistry, plugin_name: []const u8, dependencies: []const []const u8) !void {
        const entry = self.plugins.getPtr(plugin_name) orelse return;

        for (dependencies) |dep_name| {
            // Add to this plugin's dependencies
            const owned_dep_name = try self.allocator.dupe(u8, dep_name);
            try entry.dependencies.append(self.allocator, owned_dep_name);

            // Add this plugin as a dependent of the dependency
            if (self.plugins.getPtr(dep_name)) |dep_entry| {
                const owned_plugin_name = try self.allocator.dupe(u8, plugin_name);
                try dep_entry.dependents.append(self.allocator, owned_plugin_name);
            }
        }
    }
};

/// Create a plugin registry instance
pub fn createRegistry(allocator: std.mem.Allocator) !PluginRegistry {
    return try PluginRegistry.init(allocator);
}

// =============================================================================
// TESTS
// =============================================================================

test "PluginRegistry initialization" {
    var registry = try createRegistry(std.testing.allocator);
    defer registry.deinit();

    try std.testing.expectEqual(@as(usize, 0), registry.getPluginCount());
}

test "Plugin path management" {
    var registry = try createRegistry(std.testing.allocator);
    defer registry.deinit();

    try registry.addPluginPath("/usr/local/lib/plugins");

    var discovered = try registry.discoverPlugins();
    defer {
        for (discovered.items) |path| {
            std.testing.allocator.free(path);
        }
        discovered.deinit(std.testing.allocator);
    }

    // Should not crash and return a list (may be empty)
    try std.testing.expect(discovered.items.len >= 0);
}
