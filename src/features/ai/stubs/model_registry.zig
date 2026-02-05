const std = @import("std");

pub const ModelInfo = struct {};
pub const ModelRegistry = struct {
    pub fn init(_: std.mem.Allocator) @This() {
        return .{};
    }
};
