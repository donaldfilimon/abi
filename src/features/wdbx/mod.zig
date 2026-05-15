const std = @import("std");
pub const Store = struct {
    allocator: std.mem.Allocator,
    pub fn init(a: std.mem.Allocator) Store {
        return .{ .allocator = a };
    }
    pub fn store(self: *Store, key: []const u8, val: []const u8) !void {
        _ = self;
        _ = key;
        _ = val;
    }
};
