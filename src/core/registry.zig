//! Core Registry System
const std = @import("std");
const os = @import("../foundation/os.zig");
const sync = @import("../foundation/sync.zig");

pub const Registry = struct {
    allocator: std.mem.Allocator,
    modules: std.StringHashMapUnmanaged([]const u8),
    lock: sync.RwLock = .{},

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .allocator = allocator,
            .modules = .{},
        };
    }

    pub fn deinit(self: *Registry) void {
        self.lock.lockWrite();
        defer self.lock.unlockWrite();
        var it = self.modules.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.modules.deinit(self.allocator);
    }

    pub fn register(self: *Registry, name: []const u8, info: []const u8) !void {
        self.lock.lockWrite();
        defer self.lock.unlockWrite();

        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);

        const owned_info = try self.allocator.dupe(u8, info);
        errdefer self.allocator.free(owned_info);

        const result = try self.modules.getOrPut(self.allocator, owned_name);
        if (result.found_existing) {
            self.allocator.free(owned_name);
            self.allocator.free(result.value_ptr.*);
            result.value_ptr.* = owned_info;
            return;
        }
        result.key_ptr.* = owned_name;
        result.value_ptr.* = owned_info;
    }

    pub fn loadPlugins(self: *Registry) !void {
        const plugin_registry = @import("../plugin_registry.zig");
        try plugin_registry.registerPlugins(self);
    }

    pub fn getOSController(self: *Registry, gated: bool) !os.OSController {
        self.lock.lockRead();
        defer self.lock.unlockRead();
        if (!gated) return error.AccessDenied;
        return os.OSController.init(self.allocator);
    }
};

pub const Config = struct {
    max_concurrent_streams: u32 = 10,
    heartbeat_interval_ms: u32 = 5000,
    default_backend: []const u8 = "stdgpu",
};

test {
    std.testing.refAllDecls(@This());
}

test "Registry owns registered entries" {
    const testing = std.testing;

    var registry = Registry.init(testing.allocator);
    defer registry.deinit();

    try registry.register("example", "first");
    try registry.register("example", "second");

    try testing.expectEqualStrings("second", registry.modules.get("example") orelse return error.MissingPlugin);
}
