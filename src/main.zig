const std = @import("std");
const abi = @import("abi");
// Ensure the repository is built with Zig 0.15.x
const _ = @import("compat");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), .{});
    defer framework.deinit();

    var buffer: [768]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    try framework.writeSummary(stream.writer());

    const summary = stream.getWritten();
    std.debug.print("ABI Framework bootstrap complete\n{s}\n", .{summary});
}
