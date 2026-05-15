//! Core Registry System
const std = @import("std");
const foundation = @import("foundation/mod.zig");

pub const Registry = struct {
    allocator: std.mem.Allocator,
    modules: std.StringHashMapUnmanaged([]const u8),

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .allocator = allocator,
            .modules = .{},
        };
    }

    pub fn deinit(self: *Registry) void {
        var it = self.modules.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.modules.deinit(self.allocator);
    }

    pub fn register(self: *Registry, name: []const u8, info: []const u8) !void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);

        const owned_info = try self.allocator.dupe(u8, info);
        errdefer self.allocator.free(owned_info);

        const result = try self.modules.getOrPut(self.allocator, owned_name);
        if (result.found_existing) {
            self.allocator.free(owned_name);
            self.allocator.free(result.key_ptr.*);
            self.allocator.free(result.value_ptr.*);
        }
        result.key_ptr.* = owned_name;
        result.value_ptr.* = owned_info;
    }

    pub fn loadPlugins(self: *Registry) !void {
        const plugin_registry = @import("plugin_registry.zig");
        try plugin_registry.registerPlugins(self);
    }

    pub fn getOSController(self: *Registry, gated: bool) !foundation.os.OSController {
        if (!gated) return error.AccessDenied;
        return foundation.os.OSController.init(self.allocator);
    }
};

pub const Config = struct {
    max_concurrent_streams: u32 = 10,
    heartbeat_interval_ms: u32 = 5000,
    default_backend: []const u8 = "stdgpu",
};

test "Registry owns registered entries" {
    const testing = std.testing;

    var registry = Registry.init(testing.allocator);
    defer registry.deinit();

    try registry.register("example", "first");
    try registry.register("example", "second");

    try testing.expectEqualStrings("second", registry.modules.get("example") orelse return error.MissingPlugin);
}
