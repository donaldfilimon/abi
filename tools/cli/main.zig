const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    _ = allocator;

    std.debug.print("ABI Framework CLI\n", .{});
    std.debug.print("Version: {s}\n", .{abi.version()});
    std.debug.print("\nAvailable commands:\n", .{});
    std.debug.print("  --help    Show this help message\n", .{});
    std.debug.print("  --version Show version information\n", .{});
}
