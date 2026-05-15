const std = @import("std");
pub fn train(profile: []const u8) !void {
    std.log.info("Training {s}...", .{profile});
}
