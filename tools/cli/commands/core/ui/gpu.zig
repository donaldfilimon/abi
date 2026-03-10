//! GPU dashboard command redirected to the shared dashboard runtime.

const std = @import("std");
const command = @import("../../../command");
const context_mod = @import("../../../framework/context");
const dashboard_cmd = @import("./dashboard");

pub const meta: command.Meta = .{
    .name = "gpu",
    .description = "GPU monitoring dashboard via shared dashboard runtime",
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try dashboard_cmd.forwardToView(ctx, args, "gpu");
}
