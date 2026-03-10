//! Brain dashboard command redirected to the shared dashboard runtime.

const std = @import("std");
const command = @import("../../../command.zig");
const context_mod = @import("../../../framework/context.zig");
const dashboard_cmd = @import("./dashboard.zig");

pub const meta: command.Meta = .{
    .name = "brain",
    .description = "Brain visualization dashboard via shared dashboard runtime",
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try dashboard_cmd.forwardToView(ctx, args, "brain");
}
