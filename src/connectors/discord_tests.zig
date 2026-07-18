const std = @import("std");
const connector = @import("connector.zig");
const discord = @import("discord.zig");

const ConnectorError = connector.ConnectorError;

test "discord bot connect fails without token" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "",
        .client_id = "123456789012345678",
    });
    defer bot.deinit();
    try std.testing.expectError(ConnectorError.AuthenticationError, bot.connect());

    var whitespace_token_bot = discord.Bot.init(allocator, .{
        .token = "bad token",
        .client_id = "123456789012345678",
    });
    defer whitespace_token_bot.deinit();
    try std.testing.expectError(ConnectorError.AuthenticationError, whitespace_token_bot.connect());

    var control_token_bot = discord.Bot.init(allocator, .{
        .token = "bad\ttoken",
        .client_id = "123456789012345678",
    });
    defer control_token_bot.deinit();
    try std.testing.expectError(ConnectorError.AuthenticationError, control_token_bot.connect());

    var invalid_client_bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "client-123",
    });
    defer invalid_client_bot.deinit();
    try std.testing.expectError(ConnectorError.InvalidResponse, invalid_client_bot.connect());
}

test "discord bot connect succeeds with token" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "123456789012345678",
    });
    defer bot.deinit();
    try bot.connect();
    try std.testing.expect(bot.connected);
}

test "discord bot send message requires connection" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "123456789012345678",
    });
    defer bot.deinit();
    try std.testing.expectError(
        ConnectorError.ConnectionFailed,
        bot.sendMessage(allocator, "123456789012345679", "hello"),
    );
}

test "discord local send and receive validate payloads" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "123456789012345678",
    });
    defer bot.deinit();
    try bot.connect();

    const ack = try bot.sendMessage(allocator, "123456789012345679", "hello");
    defer allocator.free(ack);
    try std.testing.expect(std.mem.indexOf(u8, ack, "queued-local") != null);
    try std.testing.expect(std.mem.indexOf(u8, ack, "123456789012345679") != null);

    const processed = try bot.handleMessage("123456789012345680", "hello");
    defer allocator.free(processed);
    try std.testing.expect(std.mem.indexOf(u8, processed, "processed message") != null);
    try std.testing.expectError(ConnectorError.InvalidResponse, bot.handleMessage("", "hello"));
    try std.testing.expectError(ConnectorError.InvalidResponse, bot.handleMessage("author-1", "hello"));
    try std.testing.expectError(ConnectorError.InvalidResponse, bot.handleMessage("123456789012345678901234567890123", "hello"));
}

test "discord live transport is explicit opt-in boundary" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "123456789012345678",
        .transport = .live,
    });
    defer bot.deinit();
    try std.testing.expectError(ConnectorError.LiveTransportUnavailable, bot.connect());
}

test "discord validates snowflake ids and message size" {
    try discord.validateDiscordId("123456789012345678");
    try std.testing.expectError(ConnectorError.InvalidResponse, discord.validateDiscordId(""));
    try std.testing.expectError(ConnectorError.InvalidResponse, discord.validateDiscordId("channel-1"));
    try std.testing.expectError(ConnectorError.InvalidResponse, discord.validateDiscordId("123456789012345678901234567890123"));
    try std.testing.expectError(ConnectorError.InvalidResponse, discord.validateMessageContent(""));

    var max_sized: [discord.DISCORD_MAX_MESSAGE_BYTES]u8 = undefined;
    @memset(&max_sized, 'a');
    try discord.validateMessageContent(&max_sized);

    var oversized: [discord.DISCORD_MAX_MESSAGE_BYTES + 1]u8 = undefined;
    @memset(&oversized, 'a');
    try std.testing.expectError(ConnectorError.InvalidResponse, discord.validateMessageContent(&oversized));
}

test "discord live send validates inputs before network dispatch" {
    const allocator = std.testing.allocator;
    const io: std.Io = undefined;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "123456789012345678",
        .transport = .live,
    });
    defer bot.deinit();

    try std.testing.expectError(
        ConnectorError.InvalidResponse,
        bot.sendMessageLive(io, allocator, "channel-1", "hello"),
    );
    try std.testing.expectError(
        ConnectorError.InvalidResponse,
        bot.sendMessageLive(io, allocator, "123456789012345679", ""),
    );
}

test "discord live send validates credentials before network dispatch" {
    const allocator = std.testing.allocator;
    const io: std.Io = undefined;
    var bot = discord.Bot.init(allocator, .{
        .token = "bad token",
        .client_id = "123456789012345678",
        .transport = .live,
    });
    defer bot.deinit();

    try std.testing.expectError(
        ConnectorError.AuthenticationError,
        bot.sendMessageLive(io, allocator, "123456789012345679", "hello"),
    );
}

test {
    std.testing.refAllDecls(@This());
}
