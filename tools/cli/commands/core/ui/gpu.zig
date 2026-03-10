//! GPU dashboard command redirected to the shared dashboard runtime.

const std = @import("std");
const command = @import("../../../command.zig");
const context_mod = @import("../../../framework/context.zig");
const dashboard_cmd = @import("./dashboard.zig");

pub const meta: command.Meta = .{
    .name = "gpu",
    .description = "GPU monitoring dashboard via shared dashboard runtime",
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var new_args = try std.ArrayList([:0]const u8).initCapacity(allocator, args.len + 2);
    defer new_args.deinit(allocator);

    try new_args.append(allocator, "--view");
    try new_args.append(allocator, "gpu");
    for (args) |arg| {
        try new_args.append(allocator, arg);
    }

    try dashboard_cmd.run(ctx, new_args.items);
}
