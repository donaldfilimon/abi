const std = @import("std");

pub const Store = struct {
    pub fn init(a: std.mem.Allocator) Store {
        _ = a;
        return .{};
    }
    pub fn store(self: *Store, key: []const u8, val: []const u8) !void {
        _ = self;
        _ = key;
        _ = val;
    }
};
