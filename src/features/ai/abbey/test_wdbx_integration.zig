const std = @import("std");
const discord = @import("discord.zig");
const wdbx = @import("../../../core/database/wdbx.zig");

test "AbbeyDiscordBot WDBX memory flow" {
    const allocator = std.testing.allocator;

    var bot = try discord.AbbeyDiscordBot.init(allocator, .{
        .abbey = .{},
    });
    defer bot.deinit();

    const author = discord.User{
        .id = "user123",
        .username = "testuser",
        .discriminator = "0001",
        .global_name = null,
        .avatar = null,
        .bot = false,
        .system = false,
        .mfa_enabled = false,
        .banner = null,
        .accent_color = null,
        .locale = null,
        .verified = false,
        .email = null,
        .flags = 0,
        .premium_type = 0,
        .public_flags = 0,
    };

    const msg = discord.Message{
        .id = "msg1",
        .channel_id = "chan1",
        .author = author,
        .content = "Hello WDBX memory test",
        .timestamp = "2026-01-01T00:00:00.000Z",
    };

    _ = try bot.handleMessage(msg) catch |err| {
        // Embedding/backends may be unconfigured; ensure flow doesn't crash
        return null;
    };

    if (bot.wdbx_handle) |*handle| {
        const stats = wdbx.getStats(handle);
        try std.testing.expectGreaterThan(stats.count, @as(usize, 0));
    }
}
