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
    const allocator = ctx.allocator;
    var new_args = try std.ArrayList([:0]const u8).initCapacity(allocator, args.len + 2);
    defer new_args.deinit();

    try new_args.append("--view");
    try new_args.append("brain");
    for (args) |arg| {
        try new_args.append(arg);
    }

    try dashboard_cmd.run(ctx, new_args.items);
}
