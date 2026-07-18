const std = @import("std");
const connector = @import("connector.zig");
const http = @import("http.zig");

const ConnectorError = connector.ConnectorError;

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

test {
    std.testing.refAllDecls(@This());
}
