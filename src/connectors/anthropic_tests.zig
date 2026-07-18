const std = @import("std");
const connector = @import("connector.zig");
const anthropic = @import("anthropic.zig");
const http = @import("http.zig");
const json = @import("json.zig");

const ConnectorError = connector.ConnectorError;

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

test "anthropic streamMessageLiveIncremental rejects non-live transport" {
    const allocator = std.testing.allocator;
    var client = anthropic.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.anthropic.com",
    });
    defer client.deinit();

    const noop: *const fn (ctx: *anyopaque, chunk: http.StreamChunk) ConnectorError!void = struct {
        fn call(_: *anyopaque, _: http.StreamChunk) ConnectorError!void {}
    }.call;
    var dummy: u8 = 0;
    try std.testing.expectError(
        ConnectorError.LiveTransportUnavailable,
        client.streamMessageLiveIncremental(std.testing.io, allocator, "claude-3", "hello", 128, noop, &dummy),
    );
}

test {
    std.testing.refAllDecls(@This());
}
