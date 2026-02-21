const std = @import("std");
const train_monitor = @import("../train/monitor.zig");

pub fn run(allocator: std.mem.Allocator, _: std.Io, args: []const [:0]const u8) !void {
    try train_monitor.runMonitor(allocator, args);
}
