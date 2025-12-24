const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    std.debug.print("ABI Framework CLI\n", .{});
    std.debug.print("Version: {s}\n", .{abi.version()});
    std.debug.print("\nAvailable commands:\n", .{});
    std.debug.print("  --help    Show this help message\n", .{});
    std.debug.print("  --version Show version information\n", .{});
}
