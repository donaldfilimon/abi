const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), .{});
    defer framework.deinit();

    std.debug.print("ABI Framework bootstrap complete\n", .{});
}
