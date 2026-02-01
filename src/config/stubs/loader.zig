const std = @import("std");
const stub = @import("../stub.zig");

pub const ConfigLoader = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn load(self: *Self) stub.LoadError!stub.Config {
        _ = self;
        return error.ConfigDisabled;
    }

    pub fn loadWithBase(self: *Self, base: stub.Config) stub.LoadError!stub.Config {
        _ = self;
        _ = base;
        return error.ConfigDisabled;
    }
};
