//! Removed standalone launcher mode.

const std = @import("std");
const context_mod = @import("../../../framework/context.zig");
const utils = @import("../../../utils/mod.zig");

pub fn run(_: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    utils.output.printError("'abi ui launch' was removed. Use 'abi ui' for the shared shell.", .{});
    return error.InvalidArgument;
}

pub fn printHelp() void {
    utils.output.print(
        \\Usage: abi ui
        \\
        \\The standalone launcher mode was removed. Use `abi ui` to open the
        \\shared shell and press `/` for the command palette.
        \\
    , .{});
}

test {
    std.testing.refAllDecls(@This());
}
