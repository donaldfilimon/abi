const std = @import("std");
const discord = @import("discord.zig");
const wdbx = @import("../../../core/database/wdbx.zig");

test "AbbeyDiscordBot with WDBX integration" {
    const allocator = std.testing.allocator;

    var bot = try discord.AbbeyDiscordBot.init(allocator, .{});
    defer bot.deinit();

    // Verify integration handle
    var db_handle = try wdbx.createDatabase(allocator, "test-db");
    defer wdbx.closeDatabase(&db_handle);

    // Minimal vector insertion test
    const vec = [_]f32{ 0.1, 0.2, 0.3 };
    try wdbx.insertVector(&db_handle, 1, &vec, "meta");

    const stats = wdbx.getStats(&db_handle);
    try std.testing.expectEqual(@as(usize, 1), stats.count);
}
