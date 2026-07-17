//! ABI CLI executable entry — delegates to `cli/dispatch.zig`.

const std = @import("std");
const dispatch_mod = @import("cli/dispatch.zig");
const env = @import("abi").foundation.env;

fn runCliMapped(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    return dispatch_mod.runCli(io, allocator, args) catch |err| switch (err) {
        error.CommandDenied => blk: {
            std.debug.print("error: command denied by safety policy\n", .{});
            break :blk 3;
        },
        else => blk: {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            break :blk 1;
        },
    };
}

pub fn main(init: std.process.Init) !void {
    // Capture the process environment for portable, libc-free env lookups.
    env.install(init.environ_map);
    const allocator = std.heap.page_allocator;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    if (args.len >= 2) {
        if (std.mem.eql(u8, args[1], "--tui")) {
            var tui_args = try arena.alloc([]const u8, args.len);
            tui_args[0] = args[0];
            tui_args[1] = "tui";
            @memcpy(tui_args[2..], args[2..]);
            const exit_code = try runCliMapped(init.io, allocator, tui_args);
            if (exit_code != 0) std.process.exit(exit_code);
            return;
        }
    }

    const exit_code = try runCliMapped(init.io, allocator, args);

    if (exit_code != 0) std.process.exit(exit_code);
}
