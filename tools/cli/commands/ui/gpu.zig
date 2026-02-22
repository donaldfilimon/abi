const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const dashboard = @import("../gpu_dashboard.zig");

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try dashboard.run(ctx, args);
}
