const std = @import("std");
const connector = @import("connector.zig");
const twilio = @import("twilio.zig");

const ConnectorError = connector.ConnectorError;

const valid_twilio_sid = "AC" ++ "0123456789abcdef0123456789abcdef";
const valid_twilio_token = "0123456789abcdef0123456789abcdef";

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

test {
    std.testing.refAllDecls(@This());
}
