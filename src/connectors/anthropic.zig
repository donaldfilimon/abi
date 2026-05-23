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
};
