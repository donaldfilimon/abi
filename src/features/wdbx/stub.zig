const std = @import("std");

pub const Store = struct {
    pub fn init(a: std.mem.Allocator) Store {
        _ = a;
        return .{};
    }
    pub fn deinit(self: *Store) void {
        _ = self;
    }
    pub fn store(self: *Store, key: []const u8, val: []const u8) !void {
        _ = self;
        _ = key;
        _ = val;
    }
    pub fn get(self: *const Store, key: []const u8) ?[]const u8 {
        _ = self;
        _ = key;
        return null;
    }
    pub fn count(self: *const Store) usize {
        _ = self;
        return 0;
    }
};
