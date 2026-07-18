const std = @import("std");
const connector = @import("connector.zig");
const openai = @import("openai.zig");
const json = @import("json.zig");

const ConnectorError = connector.ConnectorError;

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

test {
    std.testing.refAllDecls(@This());
}
