const std = @import("std");

pub const PluginDescriptor = struct {
    name: []const u8,
    path: []const u8,
    feature: []const u8,
};

pub const PluginRegistry = struct {
    allocator: std.mem.Allocator,
    plugins: std.ArrayListUnmanaged(PluginDescriptor),

    pub fn init(allocator: std.mem.Allocator) PluginRegistry {
        return .{
            .allocator = allocator,
            .plugins = std.ArrayListUnmanaged(PluginDescriptor).empty,
        };
    }

    pub fn deinit(self: *PluginRegistry) void {
        for (self.plugins.items) |plugin| {
            self.allocator.free(plugin.name);
            self.allocator.free(plugin.path);
            self.allocator.free(plugin.feature);
        }
        self.plugins.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn register(
        self: *PluginRegistry,
        name: []const u8,
        path: []const u8,
        feature: []const u8,
    ) !void {
        try self.plugins.append(self.allocator, .{
            .name = try self.allocator.dupe(u8, name),
            .path = try self.allocator.dupe(u8, path),
            .feature = try self.allocator.dupe(u8, feature),
        });
    }

    pub fn findByName(self: *PluginRegistry, name: []const u8) ?PluginDescriptor {
        for (self.plugins.items) |plugin| {
            if (std.mem.eql(u8, plugin.name, name)) return plugin;
        }
        return null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "plugin registry init and deinit" {
    const allocator = std.testing.allocator;
    var registry = PluginRegistry.init(allocator);
    defer registry.deinit();

    try std.testing.expectEqual(@as(usize, 0), registry.plugins.items.len);
}

test "plugin registry register" {
    const allocator = std.testing.allocator;
    var registry = PluginRegistry.init(allocator);
    defer registry.deinit();

    try registry.register("test-plugin", "/path/to/plugin", "feature-x");
    try std.testing.expectEqual(@as(usize, 1), registry.plugins.items.len);
    try std.testing.expectEqualStrings("test-plugin", registry.plugins.items[0].name);
}

test "plugin registry findByName" {
    const allocator = std.testing.allocator;
    var registry = PluginRegistry.init(allocator);
    defer registry.deinit();

    try registry.register("plugin-a", "/path/a", "feature-a");
    try registry.register("plugin-b", "/path/b", "feature-b");

    const found = registry.findByName("plugin-b");
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("plugin-b", found.?.name);
    try std.testing.expectEqualStrings("/path/b", found.?.path);

    const not_found = registry.findByName("nonexistent");
    try std.testing.expect(not_found == null);
}
