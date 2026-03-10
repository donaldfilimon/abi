const std = @import("std");
const command = @import("../../../command");
const context_mod = @import("../../../framework/context");
const train_monitor = @import("../../ai/train/monitor");

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
