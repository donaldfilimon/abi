const std = @import("std");
const connector = @import("connector.zig");
const json = @import("json.zig");
const http = @import("http.zig");

const ConnectorError = connector.ConnectorError;
const Response = connector.Response;

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
        if (self.config.token.len == 0) return ConnectorError.AuthenticationError;
        if (self.config.client_id.len == 0) return ConnectorError.AuthenticationError;
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

        if (channel_id.len == 0 or content.len == 0) return ConnectorError.InvalidResponse;
        std.log.info("Discord local send to channel {s}: {s}", .{ channel_id, content });

        return try json.discordLocalAck(allocator, channel_id, content);
    }

    pub fn sendMessageLive(
        self: *Bot,
        io: std.Io,
        allocator: std.mem.Allocator,
        channel_id: []const u8,
        content: []const u8,
    ) ConnectorError!Response {
        if (self.config.token.len == 0) return ConnectorError.AuthenticationError;
        if (channel_id.len == 0 or content.len == 0) return ConnectorError.InvalidResponse;
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

        if (author.len == 0 or content.len == 0) return ConnectorError.InvalidResponse;
        std.log.info("Discord local receive from {s}: {s}", .{ author, content });

        return try std.fmt.allocPrint(
            self.allocator,
            "processed message from {s}",
            .{author},
        );
    }
};
