const std = @import("std");
const connector = @import("connector.zig");
const json = @import("json.zig");
const http = @import("http.zig");

const ConnectorError = connector.ConnectorError;
const ConnectorConfig = connector.ConnectorConfig;
const Response = connector.Response;
const validateConnectorConfig = connector.validateConnectorConfig;

/// xAI / Grok API base URL.
const GROK_BASE_URL = "https://api.x.ai";

/// Minimum acceptable api_key length (xAI keys are typically longer).
const MIN_KEY_LENGTH = 8;

pub const Client = struct {
    allocator: std.mem.Allocator,
    config: ConnectorConfig,

    pub fn init(allocator: std.mem.Allocator, config: ConnectorConfig) Client {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *Client) void {
        _ = self;
    }

    /// Local (deterministic) chat completion via Grok/xAI.
    /// Returns an OpenAI-compatible JSON response body.
    pub fn chatCompletion(
        self: *Client,
        allocator: std.mem.Allocator,
        model: []const u8,
        messages: []const u8,
    ) ConnectorError!Response {
        try validateGrokConfig(self.config);
        if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
        const body = try json.buildOpenAiBody(allocator, model, messages, false);
        defer allocator.free(body);

        std.log.info("Grok [local] chat completion model={s} base={s}", .{ model, self.config.base_url });

        return .{
            .status = 200,
            .body = try grokLocalResponse(allocator, model, messages, body.len),
            .owned = true,
        };
    }

    /// Live HTTP chat completion to the xAI API.
    /// Requires valid credentials and `.live` transport mode.
    pub fn chatCompletionLive(
        self: *Client,
        io: std.Io,
        allocator: std.mem.Allocator,
        model: []const u8,
        messages: []const u8,
    ) ConnectorError!Response {
        try validateGrokConfig(self.config);
        const body = try json.buildOpenAiBody(allocator, model, messages, false);
        defer allocator.free(body);
        const authorization = try http.bearerHeader(allocator, self.config.api_key);
        defer allocator.free(authorization);
        return http.httpPostJson(io, allocator, self.config, "/v1/chat/completions", body, &.{
            .{ .name = "authorization", .value = authorization },
        });
    }

    /// Local streaming chat completion (deterministic SSE-like response).
    pub fn streamChatCompletion(
        self: *Client,
        allocator: std.mem.Allocator,
        model: []const u8,
        messages: []const u8,
    ) ConnectorError!Response {
        try validateGrokConfig(self.config);
        if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
        const body = try json.buildOpenAiBody(allocator, model, messages, true);
        defer allocator.free(body);

        std.log.info("Grok [local] stream completion model={s} base={s}", .{ model, self.config.base_url });

        return .{
            .status = 200,
            .body = try json.openAiLocalStream(allocator, model, messages, body.len),
            .owned = true,
        };
    }

    /// Live streaming chat completion to the xAI API.
    pub fn streamChatCompletionLive(
        self: *Client,
        io: std.Io,
        allocator: std.mem.Allocator,
        model: []const u8,
        messages: []const u8,
    ) ConnectorError!Response {
        try validateGrokConfig(self.config);
        const body = try json.buildOpenAiBody(allocator, model, messages, true);
        defer allocator.free(body);
        const authorization = try http.bearerHeader(allocator, self.config.api_key);
        defer allocator.free(authorization);
        return http.httpPostJson(io, allocator, self.config, "/v1/chat/completions", body, &.{
            .{ .name = "authorization", .value = authorization },
        });
    }
};

/// Validate Grok-specific config constraints beyond base connector validation.
pub fn validateGrokConfig(config: ConnectorConfig) ConnectorError!void {
    try validateConnectorConfig(config);
    // Reject keys that are too short or contain whitespace/control characters.
    if (config.api_key.len < MIN_KEY_LENGTH) return ConnectorError.AuthenticationError;
    for (config.api_key) |byte| {
        if (byte < 0x21 or byte > 0x7e) return ConnectorError.AuthenticationError;
    }
}

/// Build a deterministic local Grok response mimicking the OpenAI format.
fn grokLocalResponse(allocator: std.mem.Allocator, model: []const u8, messages: []const u8, body_len: usize) ConnectorError![]u8 {
    const text = try std.fmt.allocPrint(
        allocator,
        "Grok [local] completions model={s} messages_bytes={d} request_bytes={d}",
        .{ model, messages.len, body_len },
    );
    defer allocator.free(text);
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":");
    try json.appendJsonString(&out, allocator, text);
    try out.appendSlice(allocator, "}}]}");
    return try out.toOwnedSlice(allocator);
}

/// Convenience factory for a default local-mode config.
pub fn grokConfig() ConnectorConfig {
    return .{
        .api_key = "grok-local-key",
        .base_url = GROK_BASE_URL,
        .transport = .local,
    };
}

test {
    std.testing.refAllDecls(@This());
}

test "grok local mode chatCompletion returns deterministic response" {
    const allocator = std.testing.allocator;
    var client = Client.init(allocator, grokConfig());
    defer client.deinit();

    var response = try client.chatCompletion(allocator, "grok-3", "[{\"role\":\"user\",\"content\":\"hello\"}]");
    defer response.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expect(std.mem.indexOf(u8, response.body, "Grok [local] completions") != null);
    try std.testing.expect(std.mem.indexOf(u8, response.body, "messages_bytes") != null);
}

test "grok local mode streamChatCompletion returns SSE markers" {
    const allocator = std.testing.allocator;
    var client = Client.init(allocator, grokConfig());
    defer client.deinit();

    var response = try client.streamChatCompletion(allocator, "grok-3", "[{\"role\":\"user\",\"content\":\"hello\"}]");
    defer response.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expect(std.mem.indexOf(u8, response.body, "data: [DONE]") != null);
}

test "grok live transport returns LiveTransportUnavailable from local methods" {
    const allocator = std.testing.allocator;
    var client = Client.init(allocator, .{
        .api_key = "grok-valid-key-123",
        .base_url = "https://api.x.ai",
        .transport = .live,
    });
    defer client.deinit();

    try std.testing.expectError(
        ConnectorError.LiveTransportUnavailable,
        client.chatCompletion(allocator, "grok-3", "[]"),
    );
}

test "grok validateGrokConfig rejects empty key" {
    try std.testing.expectError(
        ConnectorError.AuthenticationError,
        validateGrokConfig(.{ .api_key = "", .base_url = "https://api.x.ai" }),
    );
}

test "grok validateGrokConfig rejects short key" {
    try std.testing.expectError(
        ConnectorError.AuthenticationError,
        validateGrokConfig(.{ .api_key = "short", .base_url = "https://api.x.ai" }),
    );
}

test "grok validateGrokConfig rejects whitespace in key" {
    try std.testing.expectError(
        ConnectorError.AuthenticationError,
        validateGrokConfig(.{ .api_key = "key with space", .base_url = "https://api.x.ai" }),
    );
}

test "grok validateGrokConfig rejects control chars in key" {
    try std.testing.expectError(
        ConnectorError.AuthenticationError,
        validateGrokConfig(.{ .api_key = "key\twith\ttabs", .base_url = "https://api.x.ai" }),
    );
}

test "grok validateGrokConfig accepts valid key" {
    try validateGrokConfig(.{ .api_key = "xai-valid-key-12345", .base_url = "https://api.x.ai" });
}

test "grok client init config captures defaults" {
    const allocator = std.testing.allocator;
    var client = Client.init(allocator, grokConfig());
    defer client.deinit();
    try std.testing.expectEqualStrings("https://api.x.ai", client.config.base_url);
    try std.testing.expectEqualStrings("grok-local-key", client.config.api_key);
}

test "grok local completion empty input" {
    const allocator = std.testing.allocator;
    var client = Client.init(allocator, grokConfig());
    defer client.deinit();

    var response = try client.chatCompletion(allocator, "grok-3", "[]");
    defer response.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expect(response.body.len > 0);
}
