const std = @import("std");
const abi = @import("mod.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();
    try abi.init(allocator);
    defer abi.deinit();

    try abi.ensureCategory(.ai);
    try abi.ensureCategory(.web);

    const manager = try abi.featuresManager();
    std.debug.print("ABI Framework {s} ready\n", .{abi.version()});
    std.debug.print("Initialized features: {d}\n", .{manager.initializedCount()});
}
