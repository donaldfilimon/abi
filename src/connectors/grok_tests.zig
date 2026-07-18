const std = @import("std");
const connector = @import("connector.zig");
const grok = @import("grok.zig");

const ConnectorError = connector.ConnectorError;

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

test {
    std.testing.refAllDecls(@This());
}
