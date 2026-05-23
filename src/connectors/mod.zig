const std = @import("std");

// Re-export shared types at the connectors top level
pub const connector = @import("connector.zig");
pub const http = @import("http.zig");
pub const json = @import("json.zig");
pub const openai = @import("openai.zig");
pub const anthropic = @import("anthropic.zig");
pub const discord = @import("discord.zig");
pub const twilio = @import("twilio.zig");

// Flatten shared types for backward compatibility
pub const ConnectorError = connector.ConnectorError;
pub const TransportMode = connector.TransportMode;
pub const ConnectorConfig = connector.ConnectorConfig;
pub const transportModeName = connector.transportModeName;
pub const Response = connector.Response;

test {
    std.testing.refAllDecls(@This());
}

test "openai client init and deinit" {
    const allocator = std.testing.allocator;
    var client = openai.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.openai.com",
    });
    defer client.deinit();
    try std.testing.expectEqualStrings("https://api.openai.com", client.config.base_url);
}

test "openai chatCompletion returns response" {
    const allocator = std.testing.allocator;
    var client = openai.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.openai.com",
    });
    defer client.deinit();

    var response = try client.chatCompletion(allocator, "gpt-4", "[{\"role\":\"user\",\"content\":\"hello\"}]");
    defer response.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expect(response.body.len > 0);
}

test "openai body builder validates messages json and escapes model" {
    const allocator = std.testing.allocator;
    const body = try json.buildOpenAiBody(allocator, "gpt-\"quoted\"", "[{\"role\":\"user\",\"content\":\"hello\"}]", true);
    defer allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "gpt-\\\"quoted\\\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"stream\":true") != null);
    try std.testing.expectError(ConnectorError.InvalidResponse, json.buildOpenAiBody(allocator, "gpt", "{}", false));
}

test "openai live transport is explicit opt-in boundary" {
    const allocator = std.testing.allocator;
    var client = openai.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.openai.com",
        .transport = .live,
    });
    defer client.deinit();

    try std.testing.expectError(
        ConnectorError.LiveTransportUnavailable,
        client.chatCompletion(allocator, "gpt-4", "[]"),
    );
}

test "anthropic client init and deinit" {
    const allocator = std.testing.allocator;
    var client = anthropic.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.anthropic.com",
    });
    defer client.deinit();
    try std.testing.expectEqualStrings("https://api.anthropic.com", client.config.base_url);
}

test "anthropic message returns response" {
    const allocator = std.testing.allocator;
    var client = anthropic.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.anthropic.com",
    });
    defer client.deinit();

    var response = try client.message(allocator, "claude-3", "hello", 1024);
    defer response.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), response.status);
}

test "anthropic body builder escapes prompt" {
    const allocator = std.testing.allocator;
    const body = try json.buildAnthropicBody(allocator, "claude", "hello \"world\"", 128, false);
    defer allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "hello \\\"world\\\"") != null);
}

test "live url helpers build expected values" {
    const allocator = std.testing.allocator;
    const url = try http.joinUrl(allocator, "https://api.openai.com/", "/v1/chat/completions");
    defer allocator.free(url);
    try std.testing.expectEqualStrings("https://api.openai.com/v1/chat/completions", url);

    const bearer = try http.bearerHeader(allocator, "key");
    defer allocator.free(bearer);
    try std.testing.expectEqualStrings("Bearer key", bearer);
}

test "discord bot connect fails without token" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "",
        .client_id = "test",
    });
    defer bot.deinit();
    try std.testing.expectError(ConnectorError.AuthenticationError, bot.connect());
}

test "discord bot connect succeeds with token" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "test",
    });
    defer bot.deinit();
    try bot.connect();
    try std.testing.expect(bot.connected);
}

test "discord bot send message requires connection" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "test",
    });
    defer bot.deinit();
    try std.testing.expectError(
        ConnectorError.ConnectionFailed,
        bot.sendMessage(allocator, "channel-1", "hello"),
    );
}

test "discord live transport is explicit opt-in boundary" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "test",
        .transport = .live,
    });
    defer bot.deinit();
    try std.testing.expectError(ConnectorError.LiveTransportUnavailable, bot.connect());
}

test "twilio config validation requires credentials" {
    try std.testing.expectError(ConnectorError.AuthenticationError, twilio.validateTwilioConfig(.{
        .account_sid = "",
        .auth_token = "token",
    }));
    try std.testing.expectError(ConnectorError.AuthenticationError, twilio.validateTwilioConfig(.{
        .account_sid = "AC123",
        .auth_token = "",
    }));
    try std.testing.expectError(ConnectorError.Timeout, twilio.validateTwilioConfig(.{
        .account_sid = "AC123",
        .auth_token = "token",
        .timeout_ms = 0,
    }));
}

test "twilio parses ConversationRelay transcript event" {
    const allocator = std.testing.allocator;
    var event = try twilio.parseConversationRelayEvent(allocator,
        \\{"type":"transcript","conversationId":"CA123","customerId":"customer-1","transcript":"I need help","memory":{"profile_id":"profile-1","recall_summary":"prefers SMS"}}
    );
    defer event.deinit(allocator);

    try std.testing.expectEqual(.user_transcript, event.kind);
    try std.testing.expectEqualStrings("CA123", event.conversation_id);
    try std.testing.expectEqualStrings("customer-1", event.customer_id);
    try std.testing.expectEqualStrings("I need help", event.transcript);
    try std.testing.expect(event.memory != null);
    try std.testing.expectEqualStrings("prefers SMS", event.memory.?.recall_summary);
}

test "twilio local response includes memory context" {
    const allocator = std.testing.allocator;
    var client = twilio.Client.init(allocator, twilio.TwilioConfig.local());
    defer client.deinit();

    var response = try client.handleConversationRelayEvent(allocator, .{
        .kind = .user_transcript,
        .conversation_id = "CA123",
        .customer_id = "customer-1",
        .transcript = "I need order help",
        .memory = .{ .recall_summary = "customer has an open shipping case" },
    }, "Abi can help with that order.");
    defer response.deinit(allocator);

    try std.testing.expect(response.escalation == null);
    try std.testing.expect(std.mem.indexOf(u8, response.text, "open shipping case") != null);
}

test "twilio local response builds escalation payload" {
    const allocator = std.testing.allocator;
    var client = twilio.Client.init(allocator, twilio.TwilioConfig.local());
    defer client.deinit();

    var response = try client.handleConversationRelayEvent(allocator, .{
        .kind = .user_transcript,
        .conversation_id = "CA456",
        .customer_id = "customer-2",
        .transcript = "Please connect me to a human representative",
    }, "Abi can help.");
    defer response.deinit(allocator);

    try std.testing.expect(response.escalation != null);
    try std.testing.expectEqualStrings("human_requested", response.escalation.?.reason_code);

    const json_str = try twilio.buildConversationRelayJson(allocator, response);
    defer allocator.free(json_str);
    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"escalation\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_str, "human_requested") != null);
}

test "twilio live transport is explicit boundary" {
    const allocator = std.testing.allocator;
    var client = twilio.Client.init(allocator, .{
        .account_sid = "AC123",
        .auth_token = "token",
        .transport = .live,
    });
    defer client.deinit();

    try std.testing.expectError(ConnectorError.LiveTransportUnavailable, client.handleConversationRelayEvent(allocator, .{
        .kind = .user_transcript,
        .conversation_id = "CA789",
        .customer_id = "customer-3",
        .transcript = "hello",
    }, "hello"));
}
