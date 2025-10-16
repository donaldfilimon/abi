const std = @import("std");
const abi = @import("../mod.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    _ = allocator;

    var stdout = std.io.getStdOut().writer();
    try stdout.print("ABI CLI placeholder\n", .{});
    try stdout.print("Framework version: {s}\n", .{abi.version()});
}
