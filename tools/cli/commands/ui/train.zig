const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const train_monitor = @import("../train/monitor.zig");

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try train_monitor.runMonitor(ctx, args);
}
