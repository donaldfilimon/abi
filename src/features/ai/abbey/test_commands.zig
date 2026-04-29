const std = @import("std");
const discord = @import("../../../connectors/discord/mod.zig");
const discord_bot = @import("discord.zig");

test "dispatch slash commands" {
    const allocator = std.testing.allocator;

    var bot = try discord_bot.AbbeyDiscordBot.init(allocator, .{});
    defer bot.deinit();

    // Mock interaction for /abbey-mood
    const mood_interaction = discord.Interaction{
        .id = "1",
        .application_id = "1",
        .version = 1,
        .data = discord.InteractionData{
            .id = "1",
            .name = "abbey-mood",
            .options = &.{},
        },
    };

    const mood_resp = try discord_bot.handleSlashCommand(&bot, mood_interaction);
    try std.testing.expectEqual(@as(u8, 4), mood_resp.response_type);
    try std.testing.expect(std.mem.indexOf(u8, mood_resp.data.content, "Current mood") != null);

    // Mock interaction for /abbey-clear
    const clear_interaction = discord.Interaction{
        .id = "2",
        .application_id = "1",
        .version = 1,
        .data = discord.InteractionData{
            .id = "2",
            .name = "abbey-clear",
            .options = &.{},
        },
    };

    const clear_resp = try discord_bot.handleSlashCommand(&bot, clear_interaction);
    try std.testing.expectEqual(@as(u8, 4), clear_resp.response_type);
    try std.testing.expectEqualStrings("Conversation context cleared! Let's start fresh.", clear_resp.data.content);
}
