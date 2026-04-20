const std = @import("std");
const smc = @import("platform/smc.zig");

pub fn main() !void {
    std.debug.print("Testing SMC read availability...\n", .{});

    // Attempting to read SMC sensors
    const reading = smc.read() catch |err| {
        std.debug.print("Caught expected/unexpected error: {s}\n", .{@errorName(err)});
        return;
    };

    std.debug.print("Successfully read SMC: {any}\n", .{reading});
}
