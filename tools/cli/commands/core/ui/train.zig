const std = @import("std");
const command = @import("../../../command.zig");
const context_mod = @import("../../../framework/context.zig");
const train_monitor = @import("../../ai/train/monitor.zig");

pub const meta: command.Meta = .{
    .name = "train",
    .description = "Training progress monitor dashboard",
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try train_monitor.runMonitor(ctx, args);
}

test {
    std.testing.refAllDecls(@This());
}
