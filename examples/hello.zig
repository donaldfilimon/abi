const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, abi.FrameworkOptions{
        .enable_gpu = false,
    });
    defer abi.shutdown(&framework);

    std.debug.print("ABI Framework v{s}\n", .{abi.version()});
    std.debug.print("Framework initialized successfully\n", .{});
}
