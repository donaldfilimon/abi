const std = @import("std");

pub fn main() !void {
    // Simple main entry point for the ABI framework
    std.debug.print("ABI Framework v1.0.0-alpha\n", .{});
    std.debug.print("Run 'zig build --help' for available commands\n", .{});
}
