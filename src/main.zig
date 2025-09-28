//! ABI bootstrap executable showcasing allocator initialisation while
//! exercising the new Zig 0.16 streaming writer API. The CLI dispatcher is
//! being modernised separately; for now the binary prints the framework
//! summary using the new output layer.

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var stdout_adapter = std.io.getStdOut().writer().adaptToNewApi(&.{});
    const stdout_writer = stdout_adapter.new_interface;

    var framework = try abi.init(gpa.allocator(), .{});
    defer framework.deinit();

    var buffer: [768]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    try framework.writeSummary(stream.writer());

    const summary = stream.getWritten();
    try stdout_writer.print("ABI Framework bootstrap complete\n{s}\n", .{summary});
}
