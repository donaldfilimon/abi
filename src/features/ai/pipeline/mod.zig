const std = @import("std");
pub fn train(profile: []const u8) !void {
    if (profile.len == 0) return error.InvalidTrainingProfile;
    std.log.info("queued local training pipeline for profile {s}", .{profile});
}

test "pipeline rejects empty profile" {
    try std.testing.expectError(error.InvalidTrainingProfile, train(""));
}
