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

    pub fn chatCompletion(
        self: *Client,
        allocator: std.mem.Allocator,
        model: []const u8,
        messages: []const u8,
    ) ConnectorError!Response {
        try validateConnectorConfig(self.config);
        if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
        const body = try json.buildOpenAiBody(allocator, model, messages, false);
        defer allocator.free(body);

        std.log.info("OpenAI-compatible local chat request for model {s} via {s}", .{ model, self.config.base_url });

        return .{
            .status = 200,
            .body = try json.openAiLocalResponse(allocator, model, messages, body.len),
            .owned = true,
        };
    }

    pub fn chatCompletionLive(
        self: *Client,
        io: std.Io,
        allocator: std.mem.Allocator,
        model: []const u8,
        messages: []const u8,
    ) ConnectorError!Response {
        try validateConnectorConfig(self.config);
        const body = try json.buildOpenAiBody(allocator, model, messages, false);
        defer allocator.free(body);
        const authorization = try http.bearerHeader(allocator, self.config.api_key);
        defer allocator.free(authorization);
        return http.httpPostJson(io, allocator, self.config, "/v1/chat/completions", body, &.{
            .{ .name = "authorization", .value = authorization },
        });
    }

    pub fn streamChatCompletion(
        self: *Client,
        allocator: std.mem.Allocator,
        model: []const u8,
        messages: []const u8,
    ) ConnectorError!Response {
        try validateConnectorConfig(self.config);
        if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
        const body = try json.buildOpenAiBody(allocator, model, messages, true);
        defer allocator.free(body);

        std.log.info("OpenAI-compatible local streaming request for model {s} via {s}", .{ model, self.config.base_url });

        return .{
            .status = 200,
            .body = try json.openAiLocalStream(allocator, model, messages, body.len),
            .owned = true,
        };
    }

    pub fn streamChatCompletionLive(
        self: *Client,
        io: std.Io,
        allocator: std.mem.Allocator,
        model: []const u8,
        messages: []const u8,
    ) ConnectorError!Response {
        try validateConnectorConfig(self.config);
        const body = try json.buildOpenAiBody(allocator, model, messages, true);
        defer allocator.free(body);
        const authorization = try http.bearerHeader(allocator, self.config.api_key);
        defer allocator.free(authorization);
        return http.httpPostJson(io, allocator, self.config, "/v1/chat/completions", body, &.{
            .{ .name = "authorization", .value = authorization },
        });
    }
};

test {
    std.testing.refAllDecls(@This());
}
