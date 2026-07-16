const std = @import("std");
const connector = @import("connector.zig");
const json = @import("json.zig");

const ConnectorError = connector.ConnectorError;

test "openai body builder validates messages json and escapes model" {
    const allocator = std.testing.allocator;
    const body = try json.buildOpenAiBody(allocator, "gpt-\"quoted\"", "[{\"role\":\"user\",\"content\":\"hello\"}]", true);
    defer allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "gpt-\\\"quoted\\\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"stream\":true") != null);
    try std.testing.expectError(ConnectorError.InvalidResponse, json.buildOpenAiBody(allocator, "gpt", "{}", false));
}

test "anthropic body builder escapes prompt" {
    const allocator = std.testing.allocator;
    const body = try json.buildAnthropicBody(allocator, "claude", "hello \"world\"", 128, false);
    defer allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "hello \\\"world\\\"") != null);
}

test {
    std.testing.refAllDecls(@This());
}
