const std = @import("std");

pub const PluginDescriptor = struct {
    name: []const u8,
    path: []const u8,
    feature: []const u8,
};

pub const PluginRegistry = struct {
    allocator: std.mem.Allocator,
    plugins: std.ArrayList(PluginDescriptor),

    pub fn init(allocator: std.mem.Allocator) PluginRegistry {
        return .{
            .allocator = allocator,
            .plugins = std.ArrayList(PluginDescriptor).empty,
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

    pub fn register(self: *PluginRegistry, name: []const u8, path: []const u8, feature: []const u8) !void {
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
