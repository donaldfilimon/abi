const std = @import("std");
const launcher = @import("../tui/mod.zig");

pub fn run(allocator: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    try launcher.run(allocator, io, args);
}
