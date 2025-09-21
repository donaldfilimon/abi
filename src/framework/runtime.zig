//! High-level runtime that bootstraps the plugin registry, applies feature
//! toggles, and manages plugin discovery for ABI applications.

const std = @import("std");
const config = @import("config.zig");
const registry_mod = @import("../shared/registry.zig");
const types = @import("../shared/types.zig");

/// Orchestrates feature toggles, plugin discovery, and lifecycle management.
pub const Framework = struct {
    allocator: std.mem.Allocator,
    toggles: config.FeatureToggles,
    registry: registry_mod.PluginRegistry,
    plugin_paths: std.ArrayListUnmanaged([]u8) = .{},
    discovered_plugins: std.ArrayListUnmanaged([]u8) = .{},
    auto_discover_plugins: bool,
    auto_register_plugins: bool,
    auto_start_plugins: bool,

    pub fn init(allocator: std.mem.Allocator, options: config.FrameworkOptions) !Framework {
        var registry = try registry_mod.createRegistry(allocator);
        errdefer registry.deinit();

        var framework = Framework{
            .allocator = allocator,
            .toggles = config.deriveFeatureToggles(options),
            .registry = registry,
            .plugin_paths = .{},
            .discovered_plugins = .{},
            .auto_discover_plugins = options.auto_discover_plugins,
            .auto_register_plugins = options.auto_register_plugins,
            .auto_start_plugins = options.auto_start_plugins,
        };

        errdefer framework.deinit();

        try framework.setPluginPaths(options.plugin_paths);

        if (framework.auto_discover_plugins) {
            try framework.refreshPlugins();
        }

        return framework;
    }

    pub fn deinit(self: *Framework) void {
        if (self.auto_start_plugins) {
            self.registry.stopAllPlugins() catch {};
        }
        self.registry.deinit();
        self.clearDiscovered();
        self.discovered_plugins.deinit(self.allocator);
        self.clearPluginPaths();
        self.plugin_paths.deinit(self.allocator);
    }

    pub fn pluginRegistry(self: *Framework) *registry_mod.PluginRegistry {
        return &self.registry;
    }

    pub fn features(self: *const Framework) config.FeatureIterator {
        return self.toggles.iterator();
    }

    pub fn featureCount(self: *const Framework) usize {
        return self.toggles.count();
    }

    pub fn isFeatureEnabled(self: *const Framework, feature: config.Feature) bool {
        return self.toggles.isEnabled(feature);
    }

    pub fn setFeature(self: *Framework, feature: config.Feature, enabled: bool) bool {
        const previous = self.toggles.isEnabled(feature);
        if (previous == enabled) return false;
        self.toggles.set(feature, enabled);
        return true;
    }

    pub fn enableFeature(self: *Framework, feature: config.Feature) bool {
        return self.setFeature(feature, true);
    }

    pub fn disableFeature(self: *Framework, feature: config.Feature) bool {
        return self.setFeature(feature, false);
    }

    pub fn addPluginPath(self: *Framework, path: []const u8) !void {
        const owned = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned);
        try self.registry.addPluginPath(owned);
        errdefer self.registry.loader.removePluginPath(owned);

        try self.plugin_paths.append(self.allocator, owned);
    }

    pub fn setPluginPaths(self: *Framework, paths: []const []const u8) !void {
        self.clearPluginPaths();
        for (paths) |path| {
            try self.addPluginPath(path);
        }
    }

    pub fn pluginPathCount(self: *const Framework) usize {
        return self.plugin_paths.items.len;
    }

    pub fn pluginPath(self: *const Framework, index: usize) []const u8 {
        return self.plugin_paths.items[index];
    }

    pub fn refreshPlugins(self: *Framework) !void {
        var discovered = try self.registry.discoverPlugins();
        defer {
            for (discovered.items) |path| {
                self.allocator.free(path);
            }
            discovered.deinit();
        }

        self.clearDiscovered();

        for (discovered.items) |path| {
            const owned = try self.allocator.dupe(u8, path);
            errdefer self.allocator.free(owned);
            try self.discovered_plugins.append(self.allocator, owned);
        }

        if (self.auto_register_plugins) {
            try self.loadDiscoveredPlugins();
        }

        if (self.auto_start_plugins) {
            try self.registry.startAllPlugins();
        }
    }

    pub fn loadDiscoveredPlugins(self: *Framework) !void {
        for (self.discovered_plugins.items) |path| {
            self.registry.loadPlugin(path) catch |err| switch (err) {
                types.PluginError.AlreadyRegistered => continue,
                else => return err,
            };
        }
    }

    pub fn discoveredPluginCount(self: *const Framework) usize {
        return self.discovered_plugins.items.len;
    }

    pub fn discoveredPlugin(self: *const Framework, index: usize) []const u8 {
        return self.discovered_plugins.items[index];
    }

    pub fn writeSummary(self: *const Framework, writer: anytype) !void {
        try writer.print("Features enabled ({d}):\n", .{self.featureCount()});
        var iter = self.features();
        while (iter.next()) |feature| {
            try writer.print("  - {s}: {s}\n", .{ config.featureLabel(feature), config.featureDescription(feature) });
        }

        if (self.plugin_paths.items.len > 0) {
            try writer.print("Plugin search paths ({d}):\n", .{self.plugin_paths.items.len});
            for (self.plugin_paths.items) |path| {
                try writer.print("  - {s}\n", .{path});
            }
        } else {
            try writer.print("Plugin search paths: none configured\n", .{});
        }

        try writer.print("Registered plugins: {d}\n", .{self.registry.getPluginCount()});
        try writer.print("Discovered plugins awaiting load: {d}\n", .{self.discovered_plugins.items.len});
    }

    fn clearDiscovered(self: *Framework) void {
        for (self.discovered_plugins.items) |path| {
            self.allocator.free(path);
        }
        self.discovered_plugins.clearRetainingCapacity();
    }

    fn clearPluginPaths(self: *Framework) void {
        for (self.plugin_paths.items) |path| {
            self.registry.loader.removePluginPath(path);
            self.allocator.free(path);
        }
        self.plugin_paths.clearRetainingCapacity();
    }
};

test "framework initialises with defaults" {
    var framework = try Framework.init(std.testing.allocator, .{});
    defer framework.deinit();

    try std.testing.expect(framework.isFeatureEnabled(.ai));
    try std.testing.expect(framework.isFeatureEnabled(.database));
    try std.testing.expect(framework.isFeatureEnabled(.web));
    try std.testing.expect(framework.isFeatureEnabled(.monitoring));
    try std.testing.expect(framework.isFeatureEnabled(.simd));
    try std.testing.expect(!framework.isFeatureEnabled(.gpu));
    try std.testing.expectEqual(@as(usize, 0), framework.pluginPathCount());
}

test "framework respects custom feature selection" {
    var framework = try Framework.init(std.testing.allocator, .{
        .enabled_features = &.{ .gpu, .connectors },
        .disabled_features = &.{ .connectors },
    });
    defer framework.deinit();

    try std.testing.expect(framework.isFeatureEnabled(.gpu));
    try std.testing.expect(!framework.isFeatureEnabled(.connectors));
    try std.testing.expectEqual(@as(usize, 1), framework.featureCount());
}

test "framework manages plugin search paths" {
    var framework = try Framework.init(std.testing.allocator, .{});
    defer framework.deinit();

    try framework.setPluginPaths(&.{ "./plugins", "./more-plugins" });
    try std.testing.expectEqual(@as(usize, 2), framework.pluginPathCount());
    try std.testing.expectEqualStrings("./plugins", framework.pluginPath(0));
    try std.testing.expectEqualStrings("./more-plugins", framework.pluginPath(1));
    try std.testing.expectEqual(@as(usize, 2), framework.registry.loader.plugin_paths.items.len);
    try std.testing.expectEqualStrings("./plugins", framework.registry.loader.plugin_paths.items[0]);
    try std.testing.expectEqualStrings("./more-plugins", framework.registry.loader.plugin_paths.items[1]);

    try framework.setPluginPaths(&.{ "./fresh-plugins" });
    try std.testing.expectEqual(@as(usize, 1), framework.pluginPathCount());
    try std.testing.expectEqualStrings("./fresh-plugins", framework.pluginPath(0));
    try std.testing.expectEqual(@as(usize, 1), framework.registry.loader.plugin_paths.items.len);
    try std.testing.expectEqualStrings("./fresh-plugins", framework.registry.loader.plugin_paths.items[0]);
}

test "framework summary reports configured state" {
    var framework = try Framework.init(std.testing.allocator, .{
        .enable_gpu = true,
        .plugin_paths = &.{ "./plugins" },
    });
    defer framework.deinit();

    var buffer: [512]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    try framework.writeSummary(stream.writer());

    const written = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, written, "GPU Acceleration") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "plugins") != null);
}
