//! ABI CLI executable entry — delegates to `abi_cli/dispatch.zig`.

const std = @import("std");
const dispatch_mod = @import("abi_cli/dispatch.zig");
const handlers = @import("abi_cli/handlers/mod.zig");

pub fn main(init: std.process.Init) !void {
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
