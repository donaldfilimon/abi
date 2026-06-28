const std = @import("std");
const connector = @import("connector.zig");
const json = @import("json.zig");
const http = @import("http.zig");

const ConnectorError = connector.ConnectorError;
const Response = connector.Response;

pub const DISCORD_MAX_MESSAGE_BYTES: usize = 2000;
const DISCORD_MAX_ID_BYTES: usize = 32;

pub const BotConfig = struct {
    token: []const u8,
    client_id: []const u8,
    intents: u32 = 0xFFFF,
    transport: connector.TransportMode = .local,
};

pub const Bot = struct {
    allocator: std.mem.Allocator,
    config: BotConfig,
    connected: bool = false,

    pub fn init(allocator: std.mem.Allocator, config: BotConfig) Bot {
        return .{
            .allocator = allocator,
            .config = config,
            .connected = false,
        };
    }

    pub fn deinit(self: *Bot) void {
        if (self.connected) {
            std.log.info("Discord bot disconnecting", .{});
            self.connected = false;
        }
    }

    pub fn connect(self: *Bot) !void {
        try validateBotConfig(self.config);
        if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
        std.log.info("Discord bot opened local session for client {s}", .{self.config.client_id});
        self.connected = true;
    }

    pub fn sendMessage(
        self: *Bot,
        allocator: std.mem.Allocator,
        channel_id: []const u8,
        content: []const u8,
    ) ![]u8 {
        if (!self.connected) return ConnectorError.ConnectionFailed;

        try validateDiscordId(channel_id);
        try validateMessageContent(content);
        const summary = redactedMessageSummary(content);
        std.log.info("Discord local send to channel {s}: content_bytes={d} content=redacted", .{ channel_id, summary.content_bytes });

        return try json.discordLocalAck(allocator, channel_id, content);
    }

    pub fn sendMessageLive(
        self: *Bot,
        io: std.Io,
        allocator: std.mem.Allocator,
        channel_id: []const u8,
        content: []const u8,
    ) ConnectorError!Response {
        if (self.config.transport != .live) return ConnectorError.LiveTransportUnavailable;
        try validateBotConfig(self.config);
        try validateDiscordId(channel_id);
        try validateMessageContent(content);
        const body = try json.buildDiscordMessageBody(allocator, content);
        defer allocator.free(body);
        const authorization = try http.botHeader(allocator, self.config.token);
        defer allocator.free(authorization);
        const path = try std.fmt.allocPrint(allocator, "/api/v10/channels/{s}/messages", .{channel_id});
        defer allocator.free(path);
        return http.httpPostJson(io, allocator, .{
            .api_key = self.config.token,
            .base_url = "https://discord.com",
            .timeout_ms = 30000,
            .transport = .live,
        }, path, body, &.{
            .{ .name = "authorization", .value = authorization },
        });
    }

    pub fn handleMessage(self: *Bot, author: []const u8, content: []const u8) ![]const u8 {
        if (!self.connected) return ConnectorError.ConnectionFailed;

        try validateDiscordId(author);
        try validateMessageContent(content);
        const summary = redactedMessageSummary(content);
        std.log.info("Discord local receive from {s}: content_bytes={d} content=redacted", .{ author, summary.content_bytes });

        return try std.fmt.allocPrint(
            self.allocator,
            "processed message from {s}",
            .{author},
        );
    }
};

pub fn validateBotConfig(config: BotConfig) ConnectorError!void {
    try validateToken(config.token);
    try validateDiscordId(config.client_id);
}

pub fn validateDiscordId(id: []const u8) ConnectorError!void {
    if (id.len == 0 or id.len > DISCORD_MAX_ID_BYTES) return ConnectorError.InvalidResponse;
    for (id) |byte| {
        if (byte < '0' or byte > '9') return ConnectorError.InvalidResponse;
    }
}

pub fn validateMessageContent(content: []const u8) ConnectorError!void {
    if (content.len == 0 or content.len > DISCORD_MAX_MESSAGE_BYTES) return ConnectorError.InvalidResponse;
}

fn validateToken(token: []const u8) ConnectorError!void {
    if (token.len == 0) return ConnectorError.AuthenticationError;
    for (token) |byte| {
        if (std.ascii.isWhitespace(byte) or byte < 0x21 or byte > 0x7e) return ConnectorError.AuthenticationError;
    }
}

const RedactedMessageSummary = struct {
    content_bytes: usize,
};

fn redactedMessageSummary(content: []const u8) RedactedMessageSummary {
    return .{ .content_bytes = content.len };
}

test "discord redacted log summary omits message text" {
    const secret = "customer says reset my password";
    const summary = redactedMessageSummary(secret);
    try std.testing.expectEqual(secret.len, summary.content_bytes);
}
