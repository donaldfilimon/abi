const std = @import("std");
const dashboard = @import("../gpu_dashboard.zig");

pub fn run(allocator: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    try dashboard.run(allocator, io, args);
}
