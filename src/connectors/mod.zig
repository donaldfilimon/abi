const std = @import("std");

// Re-export shared types at the connectors top level
pub const connector = @import("connector.zig");
pub const http = @import("http.zig");
pub const json = @import("json.zig");
pub const openai = @import("openai.zig");
pub const anthropic = @import("anthropic.zig");
pub const discord = @import("discord.zig");
pub const twilio = @import("twilio.zig");
pub const grok = @import("grok.zig");

// Flatten shared types for backward compatibility
pub const ConnectorError = connector.ConnectorError;
pub const TransportMode = connector.TransportMode;
pub const ConnectorConfig = connector.ConnectorConfig;
pub const transportModeName = connector.transportModeName;
pub const Response = connector.Response;

const valid_twilio_sid = "AC" ++ "0123456789abcdef0123456789abcdef";
const valid_twilio_token = "0123456789abcdef0123456789abcdef";

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

test "openai streamChatCompletion returns SSE done marker" {
    const allocator = std.testing.allocator;
    var client = openai.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.openai.com",
    });
    defer client.deinit();

    var response = try client.streamChatCompletion(allocator, "gpt-4", "[{\"role\":\"user\",\"content\":\"hello\"}]");
    defer response.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expect(std.mem.indexOf(u8, response.body, "data: [DONE]") != null);
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

test "anthropic streamMessage returns content events" {
    const allocator = std.testing.allocator;
    var client = anthropic.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.anthropic.com",
    });
    defer client.deinit();

    var response = try client.streamMessage(allocator, "claude-3", "hello", 1024);
    defer response.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expect(std.mem.indexOf(u8, response.body, "event: content_block_delta") != null);
    try std.testing.expect(std.mem.indexOf(u8, response.body, "event: message_stop") != null);
}

test "anthropic live transport is explicit opt-in boundary" {
    const allocator = std.testing.allocator;
    var client = anthropic.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.anthropic.com",
        .transport = .live,
    });
    defer client.deinit();

    try std.testing.expectError(
        ConnectorError.LiveTransportUnavailable,
        client.message(allocator, "claude-3", "hello", 1024),
    );
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

    const url_no_slashes = try http.joinUrl(allocator, "https://api.openai.com", "v1/chat/completions");
    defer allocator.free(url_no_slashes);
    try std.testing.expectEqualStrings("https://api.openai.com/v1/chat/completions", url_no_slashes);

    const url_base_slash = try http.joinUrl(allocator, "https://api.openai.com/", "v1/chat/completions");
    defer allocator.free(url_base_slash);
    try std.testing.expectEqualStrings("https://api.openai.com/v1/chat/completions", url_base_slash);

    const url_path_slash = try http.joinUrl(allocator, "https://api.openai.com", "/v1/chat/completions");
    defer allocator.free(url_path_slash);
    try std.testing.expectEqualStrings("https://api.openai.com/v1/chat/completions", url_path_slash);

    try std.testing.expectError(ConnectorError.ConnectionFailed, http.joinUrl(allocator, "", "/v1"));
    try std.testing.expectError(ConnectorError.ConnectionFailed, http.joinUrl(allocator, "https://api.openai.com", ""));

    const bearer = try http.bearerHeader(allocator, "key");
    defer allocator.free(bearer);
    try std.testing.expectEqualStrings("Bearer key", bearer);

    const bot = try http.botHeader(allocator, "discord-token");
    defer allocator.free(bot);
    try std.testing.expectEqualStrings("Bot discord-token", bot);

    const basic = try http.basicAuthHeader(allocator, "AC123", "token");
    defer allocator.free(basic);
    try std.testing.expectEqualStrings("Basic QUMxMjM6dG9rZW4=", basic);
}

test "shared connector config validation rejects unsafe defaults" {
    try connector.validateConnectorConfig(.{ .api_key = "key", .base_url = "https://example.com" });
    try std.testing.expectError(ConnectorError.AuthenticationError, connector.validateConnectorConfig(.{ .api_key = "", .base_url = "https://example.com" }));
    try std.testing.expectError(ConnectorError.ConnectionFailed, connector.validateConnectorConfig(.{ .api_key = "key", .base_url = "" }));
    try std.testing.expectError(ConnectorError.Timeout, connector.validateConnectorConfig(.{ .api_key = "key", .base_url = "https://example.com", .timeout_ms = 0 }));
}

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

test "twilio config validation requires shaped credentials" {
    try twilio.validateTwilioConfig(.{
        .account_sid = valid_twilio_sid,
        .auth_token = valid_twilio_token,
    });
    try std.testing.expectError(ConnectorError.AuthenticationError, twilio.validateTwilioConfig(.{
        .account_sid = "",
        .auth_token = valid_twilio_token,
    }));
    try std.testing.expectError(ConnectorError.AuthenticationError, twilio.validateTwilioConfig(.{
        .account_sid = "AC123",
        .auth_token = valid_twilio_token,
    }));
    try std.testing.expectError(ConnectorError.AuthenticationError, twilio.validateTwilioConfig(.{
        .account_sid = "ZZ0123456789abcdef0123456789abcdef",
        .auth_token = valid_twilio_token,
    }));
    try std.testing.expectError(ConnectorError.AuthenticationError, twilio.validateTwilioConfig(.{
        .account_sid = "AC0123456789abcdef0123456789abcdeg",
        .auth_token = valid_twilio_token,
    }));
    try std.testing.expectError(ConnectorError.AuthenticationError, twilio.validateTwilioConfig(.{
        .account_sid = valid_twilio_sid,
        .auth_token = "bad-token-with-newline\n00000000",
    }));
    try std.testing.expectError(ConnectorError.AuthenticationError, twilio.validateTwilioConfig(.{
        .account_sid = valid_twilio_sid,
        .auth_token = "0123456789abcdef0123456789abcdeg",
    }));
    try std.testing.expectError(ConnectorError.ConnectionFailed, twilio.validateTwilioConfig(.{
        .account_sid = valid_twilio_sid,
        .auth_token = valid_twilio_token,
        .base_url = "",
    }));
    try std.testing.expectError(ConnectorError.Timeout, twilio.validateTwilioConfig(.{
        .account_sid = valid_twilio_sid,
        .auth_token = valid_twilio_token,
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

test "twilio parses ConversationRelay aliases and intelligence" {
    const allocator = std.testing.allocator;
    var event = try twilio.parseConversationRelayEvent(allocator,
        \\{"event":"prompt","callSid":"CA999","from":"+15551234567","text":"please escalate","intelligence":{"sentiment":"negative","complianceStatus":"review","escalationRecommended":true}}
    );
    defer event.deinit(allocator);

    try std.testing.expectEqual(.user_transcript, event.kind);
    try std.testing.expectEqualStrings("CA999", event.conversation_id);
    try std.testing.expectEqualStrings("+15551234567", event.customer_id);
    try std.testing.expectEqualStrings("please escalate", event.transcript);
    try std.testing.expect(event.intelligence != null);
    try std.testing.expect(event.intelligence.?.escalation_recommended);
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

test "twilio local response handles setup dtmf interrupt disconnect and intelligence escalation" {
    const allocator = std.testing.allocator;
    var client = twilio.Client.init(allocator, twilio.TwilioConfig.local());
    defer client.deinit();

    var setup = try client.handleConversationRelayEvent(allocator, .{ .kind = .setup, .conversation_id = "CA1", .customer_id = "customer" }, "ignored");
    defer setup.deinit(allocator);
    try std.testing.expect(std.mem.indexOf(u8, setup.text, "ABI support") != null);

    var dtmf = try client.handleConversationRelayEvent(allocator, .{ .kind = .dtmf, .conversation_id = "CA1", .customer_id = "customer", .digit = "7" }, "ignored");
    defer dtmf.deinit(allocator);
    try std.testing.expect(std.mem.indexOf(u8, dtmf.text, "7") != null);

    var interrupt = try client.handleConversationRelayEvent(allocator, .{ .kind = .interrupt, .conversation_id = "CA1", .customer_id = "customer" }, "ignored");
    defer interrupt.deinit(allocator);
    try std.testing.expect(std.mem.indexOf(u8, interrupt.text, "interruption") != null);

    var disconnect = try client.handleConversationRelayEvent(allocator, .{ .kind = .disconnect, .conversation_id = "CA1", .customer_id = "customer" }, "ignored");
    defer disconnect.deinit(allocator);
    try std.testing.expect(std.mem.indexOf(u8, disconnect.text, "Thanks") != null);

    var escalation = try client.handleConversationRelayEvent(allocator, .{
        .kind = .user_transcript,
        .conversation_id = "CA1",
        .customer_id = "customer",
        .transcript = "continue",
        .intelligence = .{ .escalation_recommended = true },
    }, "ignored");
    defer escalation.deinit(allocator);
    try std.testing.expect(escalation.escalation != null);
    try std.testing.expectEqualStrings("intelligence_signal", escalation.escalation.?.reason_code);
}

test "twilio ConversationRelay JSON escapes text" {
    const allocator = std.testing.allocator;
    var response = twilio.ConversationRelayResponse{
        .text = try allocator.dupe(u8, "hello \"world\"\n"),
    };
    defer response.deinit(allocator);

    const json_str = try twilio.buildConversationRelayJson(allocator, response);
    defer allocator.free(json_str);
    try std.testing.expect(std.mem.indexOf(u8, json_str, "hello \\\"world\\\"\\n") != null);
}

test "twilio live transport is explicit boundary" {
    const allocator = std.testing.allocator;
    const event: twilio.ConversationRelayEvent = .{
        .kind = .user_transcript,
        .conversation_id = "CA789",
        .customer_id = "customer-3",
        .transcript = "hello",
    };

    var live_client = twilio.Client.init(allocator, .{
        .account_sid = valid_twilio_sid,
        .auth_token = valid_twilio_token,
        .transport = .live,
    });
    defer live_client.deinit();
    try std.testing.expectError(
        ConnectorError.LiveTransportUnavailable,
        live_client.handleConversationRelayEvent(allocator, event, "hello"),
    );

    var local_client = twilio.Client.init(allocator, twilio.TwilioConfig.local());
    defer local_client.deinit();
    var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{});
    defer threaded.deinit();
    try std.testing.expectError(
        ConnectorError.LiveTransportUnavailable,
        local_client.handleConversationRelayEventLive(threaded.io(), allocator, event, "hello"),
    );
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

test "twilio rejects malformed ConversationRelay payloads" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(ConnectorError.InvalidResponse, twilio.parseConversationRelayEvent(allocator, "{"));
    try std.testing.expectError(ConnectorError.InvalidResponse, twilio.parseConversationRelayEvent(allocator, "[]"));
    try std.testing.expectError(ConnectorError.InvalidResponse, twilio.parseConversationRelayEvent(allocator, "{}"));
    try std.testing.expectError(ConnectorError.InvalidResponse, twilio.parseConversationRelayEvent(allocator, "{\"type\":\"unknown\"}"));
    try std.testing.expectError(ConnectorError.InvalidResponse, twilio.parseConversationRelayEvent(allocator, "{\"type\":\"transcript\",\"conversationId\":123,\"text\":\"hello\"}"));
    try std.testing.expectError(ConnectorError.InvalidResponse, twilio.parseConversationRelayEvent(allocator, "{\"type\":\"transcript\",\"customerId\":false,\"text\":\"hello\"}"));
    try std.testing.expectError(ConnectorError.InvalidResponse, twilio.parseConversationRelayEvent(allocator, "{\"type\":\"transcript\",\"memory\":false,\"text\":\"hello\"}"));
    try std.testing.expectError(ConnectorError.InvalidResponse, twilio.parseConversationRelayEvent(allocator, "{\"type\":\"transcript\",\"memory\":{\"recall_summary\":7},\"text\":\"hello\"}"));
    try std.testing.expectError(ConnectorError.InvalidResponse, twilio.parseConversationRelayEvent(allocator, "{\"type\":\"transcript\",\"intelligence\":false,\"text\":\"hello\"}"));
    try std.testing.expectError(ConnectorError.InvalidResponse, twilio.parseConversationRelayEvent(allocator, "{\"type\":\"transcript\",\"intelligence\":{\"escalationRecommended\":\"yes\"},\"text\":\"hello\"}"));
}

test "twilio live path validates config before network dispatch" {
    const allocator = std.testing.allocator;
    const io: std.Io = undefined;
    const event: twilio.ConversationRelayEvent = .{
        .kind = .user_transcript,
        .conversation_id = "CA789",
        .customer_id = "customer-3",
        .transcript = "hello",
    };

    var invalid_sid_client = twilio.Client.init(allocator, .{
        .account_sid = "AC123",
        .auth_token = valid_twilio_token,
        .transport = .live,
    });
    defer invalid_sid_client.deinit();
    try std.testing.expectError(
        ConnectorError.AuthenticationError,
        invalid_sid_client.handleConversationRelayEventLive(io, allocator, event, "hello"),
    );

    var timeout_client = twilio.Client.init(allocator, .{
        .account_sid = valid_twilio_sid,
        .auth_token = valid_twilio_token,
        .timeout_ms = 0,
        .transport = .live,
    });
    defer timeout_client.deinit();
    try std.testing.expectError(
        ConnectorError.Timeout,
        timeout_client.handleConversationRelayEventLive(io, allocator, event, "hello"),
    );
}

test "grok client init and deinit" {
    const allocator = std.testing.allocator;
    var client = grok.Client.init(allocator, .{
        .api_key = "xai-test-key-12345",
        .base_url = "https://api.x.ai",
    });
    defer client.deinit();
    try std.testing.expectEqualStrings("https://api.x.ai", client.config.base_url);
}

test "grok chatCompletion returns response" {
    const allocator = std.testing.allocator;
    var client = grok.Client.init(allocator, .{
        .api_key = "xai-test-key-12345",
        .base_url = "https://api.x.ai",
    });
    defer client.deinit();

    var response = try client.chatCompletion(allocator, "grok-3", "[{\"role\":\"user\",\"content\":\"hello\"}]");
    defer response.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expect(response.body.len > 0);
}

test "grok streamChatCompletion returns SSE done marker" {
    const allocator = std.testing.allocator;
    var client = grok.Client.init(allocator, .{
        .api_key = "xai-test-key-12345",
        .base_url = "https://api.x.ai",
    });
    defer client.deinit();

    var response = try client.streamChatCompletion(allocator, "grok-3", "[{\"role\":\"user\",\"content\":\"hello\"}]");
    defer response.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expect(std.mem.indexOf(u8, response.body, "data: [DONE]") != null);
}

test "grok live transport is explicit opt-in boundary" {
    const allocator = std.testing.allocator;
    var client = grok.Client.init(allocator, .{
        .api_key = "xai-test-key-12345",
        .base_url = "https://api.x.ai",
        .transport = .live,
    });
    defer client.deinit();

    try std.testing.expectError(
        ConnectorError.LiveTransportUnavailable,
        client.chatCompletion(allocator, "grok-3", "[]"),
    );
}

test "grok config validation rejects empty/short/whitespace keys" {
    try std.testing.expectError(ConnectorError.AuthenticationError, grok.validateGrokConfig(.{ .api_key = "", .base_url = "https://api.x.ai" }));
    try std.testing.expectError(ConnectorError.AuthenticationError, grok.validateGrokConfig(.{ .api_key = "short", .base_url = "https://api.x.ai" }));
    try std.testing.expectError(ConnectorError.AuthenticationError, grok.validateGrokConfig(.{ .api_key = "key with space", .base_url = "https://api.x.ai" }));
    try std.testing.expectError(ConnectorError.AuthenticationError, grok.validateGrokConfig(.{ .api_key = "key\twith\ttab", .base_url = "https://api.x.ai" }));
    try grok.validateGrokConfig(.{ .api_key = "xai-valid-key-12345", .base_url = "https://api.x.ai" });
}

test "grok local completion empty input" {
    const allocator = std.testing.allocator;
    var client = grok.Client.init(allocator, grok.grokConfig());
    defer client.deinit();

    var response = try client.chatCompletion(allocator, "grok-3", "[]");
    defer response.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expect(response.body.len > 0);
}
