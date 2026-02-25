//! Process / CLI utilities stub.

const std = @import("std");

pub fn run(allocator: std.mem.Allocator, argv: []const []const u8) !std.ArrayListUnmanaged(u8) {
    _ = allocator;
    _ = argv;
    return .{};
}
