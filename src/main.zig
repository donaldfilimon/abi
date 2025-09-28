//! ABI bootstrap executable showcasing allocator initialisation while
//! exercising the new Zig 0.16 streaming writer API. The CLI dispatcher is
//! being modernised separately; for now the binary prints the framework
//! summary using the new output layer.

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const stdout = std.io.getStdOut().writer();

    if (args.len < 2) {
        try stdout.print("Usage: abi [version|run]\n", .{});
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "version")) {
        try stdout.print("ABI Framework Version: {s}\n", .{abi.version});
    } else if (std.mem.eql(u8, command, "run")) {
        try stdout.print("Initializing ABI Framework...\n", .{});
        var framework = try abi.initFramework(allocator, null);
        defer framework.deinit();

        try framework.start();
        try stdout.print("Framework started. Running for 2 seconds...\n", .{});
        std.time.sleep(2 * std.time.ns_per_s);
        framework.stop();
        try stdout.print("Framework stopped.\n", .{});
    } else {
        try stdout.print("Unknown command: {s}\n", .{command});
    }
}
