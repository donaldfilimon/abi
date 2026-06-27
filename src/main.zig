//! ABI CLI executable entry — delegates to `cli/dispatch.zig`.

const std = @import("std");
const dispatch_mod = @import("cli/dispatch.zig");
const handlers = @import("cli/handlers/mod.zig");
const env = @import("foundation/env.zig");

pub fn main(init: std.process.Init) !void {
    // Capture the process environment for portable, libc-free env lookups.
    env.install(init.environ_map);
    const allocator = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (args.len >= 2) {
        if (std.mem.eql(u8, args[1], "--tui")) {
            _ = try handlers.renderTui(allocator);
            return;
        }
    }

    const exit_code = dispatch_mod.runCli(init.io, allocator, args) catch |err| switch (err) {
        error.CommandDenied => blk: {
            std.debug.print("error: command denied by safety policy\n", .{});
            break :blk 3;
        },
        else => blk: {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            break :blk 1;
        },
    };

    if (exit_code != 0) std.process.exit(exit_code);
}
