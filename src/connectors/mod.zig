const std = @import("std");

pub const ConnectorError = error{
    OutOfMemory,
    ConnectionFailed,
    AuthenticationError,
    RateLimited,
    InvalidResponse,
    Timeout,
    LiveTransportUnavailable,
};

pub const TransportMode = enum {
    local,
    live,
};

pub const ConnectorConfig = struct {
    api_key: []const u8,
    base_url: []const u8,
    timeout_ms: u32 = 30000,
    transport: TransportMode = .local,
};

pub fn transportModeName(mode: TransportMode) []const u8 {
    return switch (mode) {
        .local => "local",
        .live => "live",
    };
}

pub const Response = struct {
    status: u16,
    body: []u8,
    owned: bool = true,

    pub fn deinit(self: *Response, allocator: std.mem.Allocator) void {
        if (self.owned and self.body.len > 0) {
            allocator.free(self.body);
            self.body = "";
        }
    }
};

pub const openai = struct {
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
            const body = try buildOpenAiBody(allocator, model, messages, false);
            defer allocator.free(body);

            std.log.info("OpenAI-compatible local chat request for model {s} via {s}", .{ model, self.config.base_url });

            return .{
                .status = 200,
                .body = try openAiLocalResponse(allocator, model, messages, body.len),
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
            const body = try buildOpenAiBody(allocator, model, messages, false);
            defer allocator.free(body);
            const authorization = try bearerHeader(allocator, self.config.api_key);
            defer allocator.free(authorization);
            return httpPostJson(io, allocator, self.config, "/v1/chat/completions", body, &.{
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
            const body = try buildOpenAiBody(allocator, model, messages, true);
            defer allocator.free(body);

            std.log.info("OpenAI-compatible local streaming request for model {s} via {s}", .{ model, self.config.base_url });

            return .{
                .status = 200,
                .body = try openAiLocalStream(allocator, model, messages, body.len),
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
            const body = try buildOpenAiBody(allocator, model, messages, true);
            defer allocator.free(body);
            const authorization = try bearerHeader(allocator, self.config.api_key);
            defer allocator.free(authorization);
            return httpPostJson(io, allocator, self.config, "/v1/chat/completions", body, &.{
                .{ .name = "authorization", .value = authorization },
            });
        }
    };
};

pub const anthropic = struct {
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
            const body = try buildAnthropicBody(allocator, model, prompt, max_tokens, false);
            defer allocator.free(body);

            std.log.info("Anthropic-compatible local message request for model {s} via {s}", .{ model, self.config.base_url });

            return .{
                .status = 200,
                .body = try anthropicLocalResponse(allocator, model, prompt, max_tokens, body.len),
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
            const body = try buildAnthropicBody(allocator, model, prompt, max_tokens, false);
            defer allocator.free(body);
            return httpPostJson(io, allocator, self.config, "/v1/messages", body, &.{
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
            const body = try buildAnthropicBody(allocator, model, prompt, max_tokens, true);
            defer allocator.free(body);

            std.log.info("Anthropic-compatible local streaming request for model {s} via {s}", .{ model, self.config.base_url });

            return .{
                .status = 200,
                .body = try anthropicLocalStream(allocator, model, prompt, max_tokens, body.len),
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
            const body = try buildAnthropicBody(allocator, model, prompt, max_tokens, true);
            defer allocator.free(body);
            return httpPostJson(io, allocator, self.config, "/v1/messages", body, &.{
                .{ .name = "x-api-key", .value = self.config.api_key },
                .{ .name = "anthropic-version", .value = "2023-06-01" },
            });
        }
    };
};

pub const discord = struct {
    pub const BotConfig = struct {
        token: []const u8,
        client_id: []const u8,
        intents: u32 = 0xFFFF,
        transport: TransportMode = .local,
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

            return try discordLocalAck(allocator, channel_id, content);
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
            const body = try buildDiscordMessageBody(allocator, content);
            defer allocator.free(body);
            const authorization = try botHeader(allocator, self.config.token);
            defer allocator.free(authorization);
            const path = try std.fmt.allocPrint(allocator, "/api/v10/channels/{s}/messages", .{channel_id});
            defer allocator.free(path);
            return httpPostJson(io, allocator, .{
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
};

fn validateConnectorConfig(config: ConnectorConfig) ConnectorError!void {
    if (config.api_key.len == 0) return ConnectorError.AuthenticationError;
    if (config.base_url.len == 0) return ConnectorError.ConnectionFailed;
    if (config.timeout_ms == 0) return ConnectorError.Timeout;
}

fn httpPostJson(
    io: std.Io,
    allocator: std.mem.Allocator,
    config: ConnectorConfig,
    path: []const u8,
    body: []const u8,
    extra_headers: []const std.http.Header,
) ConnectorError!Response {
    if (config.transport != .live) return ConnectorError.LiveTransportUnavailable;

    const url = try joinUrl(allocator, config.base_url, path);
    defer allocator.free(url);

    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    var response_writer = std.Io.Writer.Allocating.init(allocator);
    defer response_writer.deinit();

    const result = client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = body,
        .headers = .{ .content_type = .{ .override = "application/json" } },
        .extra_headers = extra_headers,
        .response_writer = &response_writer.writer,
        .redirect_behavior = .unhandled,
        .keep_alive = false,
    }) catch |err| return mapHttpError(err);

    try mapHttpStatus(result.status);
    return .{
        .status = @intCast(@intFromEnum(result.status)),
        .body = try response_writer.toOwnedSlice(),
        .owned = true,
    };
}

fn mapHttpError(err: anyerror) ConnectorError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.Timeout => error.Timeout,
        error.AuthenticationError => error.AuthenticationError,
        else => error.ConnectionFailed,
    };
}

fn mapHttpStatus(status: std.http.Status) ConnectorError!void {
    const code: u16 = @intCast(@intFromEnum(status));
    return switch (code) {
        200...299 => {},
        401, 403 => ConnectorError.AuthenticationError,
        408, 504 => ConnectorError.Timeout,
        429 => ConnectorError.RateLimited,
        else => ConnectorError.InvalidResponse,
    };
}

fn joinUrl(allocator: std.mem.Allocator, base_url: []const u8, path: []const u8) ConnectorError![]u8 {
    if (base_url.len == 0 or path.len == 0) return ConnectorError.ConnectionFailed;
    const base_has_slash = base_url[base_url.len - 1] == '/';
    const path_has_slash = path[0] == '/';
    if (base_has_slash and path_has_slash) return try std.fmt.allocPrint(allocator, "{s}{s}", .{ base_url, path[1..] });
    if (!base_has_slash and !path_has_slash) return try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_url, path });
    return try std.fmt.allocPrint(allocator, "{s}{s}", .{ base_url, path });
}

fn bearerHeader(allocator: std.mem.Allocator, api_key: []const u8) ConnectorError![]u8 {
    return try std.fmt.allocPrint(allocator, "Bearer {s}", .{api_key});
}

fn botHeader(allocator: std.mem.Allocator, token: []const u8) ConnectorError![]u8 {
    return try std.fmt.allocPrint(allocator, "Bot {s}", .{token});
}

fn buildOpenAiBody(allocator: std.mem.Allocator, model: []const u8, messages: []const u8, stream: bool) ConnectorError![]u8 {
    try validateJsonValue(allocator, messages, .array);
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"model\":");
    try appendJsonString(&out, allocator, model);
    try out.appendSlice(allocator, ",\"messages\":");
    try out.appendSlice(allocator, messages);
    if (stream) try out.appendSlice(allocator, ",\"stream\":true");
    try out.append(allocator, '}');
    return try out.toOwnedSlice(allocator);
}

fn buildAnthropicBody(allocator: std.mem.Allocator, model: []const u8, prompt: []const u8, max_tokens: u32, stream: bool) ConnectorError![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"model\":");
    try appendJsonString(&out, allocator, model);
    try out.print(allocator, ",\"max_tokens\":{d}", .{max_tokens});
    if (stream) try out.appendSlice(allocator, ",\"stream\":true");
    try out.appendSlice(allocator, ",\"messages\":[{\"role\":\"user\",\"content\":");
    try appendJsonString(&out, allocator, prompt);
    try out.appendSlice(allocator, "}]}");
    return try out.toOwnedSlice(allocator);
}

fn buildDiscordMessageBody(allocator: std.mem.Allocator, content: []const u8) ConnectorError![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"content\":");
    try appendJsonString(&out, allocator, content);
    try out.append(allocator, '}');
    return try out.toOwnedSlice(allocator);
}

const ExpectedJsonKind = enum { array, object, any };

fn validateJsonValue(allocator: std.mem.Allocator, input: []const u8, expected: ExpectedJsonKind) ConnectorError!void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, input, .{}) catch return ConnectorError.InvalidResponse;
    defer parsed.deinit();
    switch (expected) {
        .any => {},
        .array => switch (parsed.value) {
            .array => {},
            else => return ConnectorError.InvalidResponse,
        },
        .object => switch (parsed.value) {
            .object => {},
            else => return ConnectorError.InvalidResponse,
        },
    }
}

fn openAiLocalResponse(allocator: std.mem.Allocator, model: []const u8, messages: []const u8, body_len: usize) ![]u8 {
    const text = try std.fmt.allocPrint(
        allocator,
        "OpenAI-compatible local response model={s} messages_bytes={d} request_bytes={d}",
        .{ model, messages.len, body_len },
    );
    defer allocator.free(text);
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":");
    try appendJsonString(&out, allocator, text);
    try out.appendSlice(allocator, "}}]}");
    return try out.toOwnedSlice(allocator);
}

fn openAiLocalStream(allocator: std.mem.Allocator, model: []const u8, messages: []const u8, body_len: usize) ![]u8 {
    const content = try std.fmt.allocPrint(allocator, "OpenAI-compatible local stream model={s} messages_bytes={d} request_bytes={d}", .{ model, messages.len, body_len });
    defer allocator.free(content);
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "data: ");
    try appendOpenAiDelta(&out, allocator, content);
    try out.appendSlice(allocator, "\n\ndata: [DONE]\n\n");
    return try out.toOwnedSlice(allocator);
}

fn appendOpenAiDelta(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, content: []const u8) !void {
    try out.appendSlice(allocator, "{\"choices\":[{\"delta\":{\"content\":");
    try appendJsonString(out, allocator, content);
    try out.appendSlice(allocator, "}}]}");
}

fn anthropicLocalResponse(allocator: std.mem.Allocator, model: []const u8, prompt: []const u8, max_tokens: u32, body_len: usize) ![]u8 {
    const text = try std.fmt.allocPrint(
        allocator,
        "Anthropic-compatible local response model={s} prompt_bytes={d} max_tokens={d} request_bytes={d}",
        .{ model, prompt.len, max_tokens, body_len },
    );
    defer allocator.free(text);
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"content\":[{\"type\":\"text\",\"text\":");
    try appendJsonString(&out, allocator, text);
    try out.appendSlice(allocator, "}]}");
    return try out.toOwnedSlice(allocator);
}

fn anthropicLocalStream(allocator: std.mem.Allocator, model: []const u8, prompt: []const u8, max_tokens: u32, body_len: usize) ![]u8 {
    const content = try std.fmt.allocPrint(allocator, "Anthropic-compatible local stream model={s} prompt_bytes={d} max_tokens={d} request_bytes={d}", .{ model, prompt.len, max_tokens, body_len });
    defer allocator.free(content);
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "event: content_block_delta\ndata: ");
    try appendAnthropicDelta(&out, allocator, content);
    try out.appendSlice(allocator, "\n\nevent: message_stop\n\n");
    return try out.toOwnedSlice(allocator);
}

fn appendAnthropicDelta(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, content: []const u8) !void {
    try out.appendSlice(allocator, "{\"delta\":{\"text\":");
    try appendJsonString(out, allocator, content);
    try out.appendSlice(allocator, "}}");
}

fn discordLocalAck(allocator: std.mem.Allocator, channel_id: []const u8, content: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"status\":\"queued-local\",\"channel_id\":");
    try appendJsonString(&out, allocator, channel_id);
    try out.print(allocator, ",\"content_bytes\":{d}}}", .{content.len});
    return try out.toOwnedSlice(allocator);
}

fn appendJsonString(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try out.append(allocator, '"');
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0x00...0x07 => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            0x08 => try out.appendSlice(allocator, "\\b"),
            0x0c => try out.appendSlice(allocator, "\\f"),
            0x0b => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            0x0e...0x1f => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            else => try out.append(allocator, byte),
        }
    }
    try out.append(allocator, '"');
}

test {
    std.testing.refAllDecls(@This());
}

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

test "openai body builder validates messages json and escapes model" {
    const allocator = std.testing.allocator;
    const body = try buildOpenAiBody(allocator, "gpt-\"quoted\"", "[{\"role\":\"user\",\"content\":\"hello\"}]", true);
    defer allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "gpt-\\\"quoted\\\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"stream\":true") != null);
    try std.testing.expectError(ConnectorError.InvalidResponse, buildOpenAiBody(allocator, "gpt", "{}", false));
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

test "anthropic body builder escapes prompt" {
    const allocator = std.testing.allocator;
    const body = try buildAnthropicBody(allocator, "claude", "hello \"world\"", 128, false);
    defer allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "hello \\\"world\\\"") != null);
}

test "live url helpers build expected values" {
    const allocator = std.testing.allocator;
    const url = try joinUrl(allocator, "https://api.openai.com/", "/v1/chat/completions");
    defer allocator.free(url);
    try std.testing.expectEqualStrings("https://api.openai.com/v1/chat/completions", url);

    const bearer = try bearerHeader(allocator, "key");
    defer allocator.free(bearer);
    try std.testing.expectEqualStrings("Bearer key", bearer);
}

test "discord bot connect fails without token" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "",
        .client_id = "test",
    });
    defer bot.deinit();
    try std.testing.expectError(ConnectorError.AuthenticationError, bot.connect());
}

test "discord bot connect succeeds with token" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "test",
    });
    defer bot.deinit();
    try bot.connect();
    try std.testing.expect(bot.connected);
}

test "discord bot send message requires connection" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "test",
    });
    defer bot.deinit();
    try std.testing.expectError(
        ConnectorError.ConnectionFailed,
        bot.sendMessage(allocator, "channel-1", "hello"),
    );
}

test "discord live transport is explicit opt-in boundary" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "test",
        .transport = .live,
    });
    defer bot.deinit();
    try std.testing.expectError(ConnectorError.LiveTransportUnavailable, bot.connect());
}
