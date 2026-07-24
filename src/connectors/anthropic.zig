const std = @import("std");
const connector = @import("connector.zig");
const json = @import("json.zig");
const http = @import("http.zig");

const ConnectorError = connector.ConnectorError;
const ConnectorConfig = connector.ConnectorConfig;
const Response = connector.Response;
const validateConnectorConfig = connector.validateConnectorConfig;

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

    pub fn message(
        self: *Client,
        allocator: std.mem.Allocator,
        model: []const u8,
        prompt: []const u8,
        max_tokens: u32,
    ) ConnectorError!Response {
        try validateConnectorConfig(self.config);
        if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
        const body = try json.buildAnthropicBody(allocator, model, prompt, max_tokens, false);
        defer allocator.free(body);

        std.log.info("Anthropic-compatible local message request for model {s} via {s}", .{ model, self.config.base_url });

        return .{
            .status = 200,
            .body = try json.anthropicLocalResponse(allocator, model, prompt, max_tokens, body.len),
            .owned = true,
        };
    }

    pub fn messageLive(
        self: *Client,
        io: std.Io,
        allocator: std.mem.Allocator,
        model: []const u8,
        prompt: []const u8,
        max_tokens: u32,
    ) ConnectorError!Response {
        try validateConnectorConfig(self.config);
        const body = try json.buildAnthropicBody(allocator, model, prompt, max_tokens, false);
        defer allocator.free(body);
        return http.httpPostJson(io, allocator, self.config, "/v1/messages", body, &.{
            .{ .name = "x-api-key", .value = self.config.api_key },
            .{ .name = "anthropic-version", .value = "2023-06-01" },
        });
    }

    pub fn streamMessage(
        self: *Client,
        allocator: std.mem.Allocator,
        model: []const u8,
        prompt: []const u8,
        max_tokens: u32,
    ) ConnectorError!Response {
        try validateConnectorConfig(self.config);
        if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
        const body = try json.buildAnthropicBody(allocator, model, prompt, max_tokens, true);
        defer allocator.free(body);

        std.log.info("Anthropic-compatible local streaming request for model {s} via {s}", .{ model, self.config.base_url });

        return .{
            .status = 200,
            .body = try json.anthropicLocalStream(allocator, model, prompt, max_tokens, body.len),
            .owned = true,
        };
    }

    /// Non-streaming live POST that still requests `stream:true` in the body
    /// and returns the raw SSE payload as `Response.body` (legacy shape).
    pub fn streamMessageLive(
        self: *Client,
        io: std.Io,
        allocator: std.mem.Allocator,
        model: []const u8,
        prompt: []const u8,
        max_tokens: u32,
    ) ConnectorError!Response {
        try validateConnectorConfig(self.config);
        const body = try json.buildAnthropicBody(allocator, model, prompt, max_tokens, true);
        defer allocator.free(body);
        return http.httpPostJson(io, allocator, self.config, "/v1/messages", body, &.{
            .{ .name = "x-api-key", .value = self.config.api_key },
            .{ .name = "anthropic-version", .value = "2023-06-01" },
        });
    }

    /// Live Anthropic Messages SSE: invokes `on_chunk` for each text delta and
    /// returns the concatenated token text (owned). Same callback shape as the
    /// local-bridge OpenAI SSE path.
    pub fn streamMessageLiveIncremental(
        self: *Client,
        io: std.Io,
        allocator: std.mem.Allocator,
        model: []const u8,
        prompt: []const u8,
        max_tokens: u32,
        on_chunk: http.StreamCallback,
        callback_ctx: *anyopaque,
    ) ConnectorError![]const u8 {
        try validateConnectorConfig(self.config);
        const body = try json.buildAnthropicBody(allocator, model, prompt, max_tokens, true);
        defer allocator.free(body);
        return http.httpPostJsonStreamingIncremental(io, allocator, self.config, "/v1/messages", body, on_chunk, callback_ctx, &.{
            .{ .name = "x-api-key", .value = self.config.api_key },
            .{ .name = "anthropic-version", .value = "2023-06-01" },
        });
    }
};

test "Client.message on the local transport synthesizes a parseable response without any network call" {
    const a = std.testing.allocator;
    var client = Client.init(a, .{
        .api_key = "test-key",
        .base_url = "http://local.invalid",
        .transport = .local,
    });
    defer client.deinit();

    var response = try client.message(a, "fable-5", "hello there", 256);
    defer response.deinit(a);

    try std.testing.expectEqual(@as(u16, 200), response.status);
    try json.validateJsonValue(a, response.body, .object);
    try std.testing.expect(std.mem.indexOf(u8, response.body, "model=fable-5") != null);
    try std.testing.expect(std.mem.indexOf(u8, response.body, "max_tokens=256") != null);
}

test "Client.streamMessage on the local transport synthesizes SSE-shaped output" {
    const a = std.testing.allocator;
    var client = Client.init(a, .{
        .api_key = "test-key",
        .base_url = "http://local.invalid",
        .transport = .local,
    });
    defer client.deinit();

    var response = try client.streamMessage(a, "fable-5", "hi", 128);
    defer response.deinit(a);

    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expect(std.mem.startsWith(u8, response.body, "event: content_block_delta\ndata: "));
}

test "Client.message rejects an empty api key before building any request" {
    const a = std.testing.allocator;
    var client = Client.init(a, .{
        .api_key = "",
        .base_url = "http://local.invalid",
        .transport = .local,
    });
    defer client.deinit();

    try std.testing.expectError(ConnectorError.AuthenticationError, client.message(a, "fable-5", "hi", 128));
}

test "Client.message on the live transport refuses local synthesis (LiveTransportUnavailable)" {
    const a = std.testing.allocator;
    var client = Client.init(a, .{
        .api_key = "test-key",
        .base_url = "https://api.anthropic.com",
        .transport = .live,
    });
    defer client.deinit();

    try std.testing.expectError(ConnectorError.LiveTransportUnavailable, client.message(a, "fable-5", "hi", 128));
}

test {
    std.testing.refAllDecls(@This());
}
