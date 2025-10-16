const std = @import("std");
const common = @import("cli/common.zig");
const router = @import("cli/router.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var ctx = common.Context{ .allocator = allocator };
    try router.run(&ctx, args);
}
