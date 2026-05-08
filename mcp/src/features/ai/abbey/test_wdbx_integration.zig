const std = @import("std");
const discord_connector = @import("../../../connectors/discord/mod.zig");
const discord_bot = @import("discord.zig");

fn testAuthor(id: []const u8, username: []const u8) discord_connector.User {
    return .{
        .id = id,
        .username = username,
        .discriminator = "0001",
    };
}

fn testMessage(id: []const u8, channel_id: []const u8, author: discord_connector.User, content: []const u8) discord_connector.Message {
    return .{
        .id = id,
        .channel_id = channel_id,
        .author = author,
        .content = content,
        .timestamp = "2026-01-01T00:00:00.000Z",
    };
}

test "AbbeyDiscordBot records routed messages in assistant memory" {
    const allocator = std.testing.allocator;

    var bot = try discord_bot.AbbeyDiscordBot.init(allocator, .{
        .abbey = .{},
        .respond_to_all = true,
        .memory_top_k = 5,
    });
    defer bot.deinit();

    try std.testing.expect(bot.assistant.router.memory != null);

    const author = testAuthor("user123", "memorytest");
    const msg1 = testMessage("msg1", "chan1", author, "Explain ABI memory routing");
    var resp1 = (try bot.handleMessage(msg1)).?;
    defer resp1.deinit();

    const msg2 = testMessage("msg2", "chan1", author, "Explain ABI memory routing again");
    var resp2 = (try bot.handleMessage(msg2)).?;
    defer resp2.deinit();

    const mem = &bot.assistant.router.memory.?;
    try std.testing.expectEqual(@as(u64, 2), mem.getInteractionCount());
    try std.testing.expect(mem.chain.current_head != null);
}

test "AbbeyDiscordBot ignores messages outside response policy" {
    const allocator = std.testing.allocator;

    var bot = try discord_bot.AbbeyDiscordBot.init(allocator, .{
        .abbey = .{},
        .respond_to_all = false,
    });
    defer bot.deinit();

    const author = testAuthor("user456", "quiettest");
    const msg = testMessage("msg3", "chan2", author, "This should not be handled");
    try std.testing.expect((try bot.handleMessage(msg)) == null);

    const mem = &bot.assistant.router.memory.?;
    try std.testing.expectEqual(@as(u64, 0), mem.getInteractionCount());
}

test "AbbeyDiscordBot skips bot-authored messages" {
    const allocator = std.testing.allocator;

    var bot = try discord_bot.AbbeyDiscordBot.init(allocator, .{
        .abbey = .{},
        .respond_to_all = true,
    });
    defer bot.deinit();

    var author = testAuthor("bot-user", "bot");
    author.bot = true;

    const msg = testMessage("msg4", "chan3", author, "Bot messages are ignored");
    try std.testing.expect((try bot.handleMessage(msg)) == null);

    const mem = &bot.assistant.router.memory.?;
    try std.testing.expectEqual(@as(u64, 0), mem.getInteractionCount());
}
