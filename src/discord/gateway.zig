//! Discord Gateway and WebSocket integration
//!
//! This module handles real-time Discord communication via WebSocket gateway:
//! - WebSocket connection management
//! - Event handling and routing
//! - Message processing pipeline
//! - Rate limiting and reconnection logic

const std = @import("std");
const core = @import("../core/mod.zig");
const types = @import("types.zig");
const agent = @import("../agent.zig");
const database = @import("../mlai/wdbx/db.zig");
const discord_api = @import("api.zig");

/// Re-export commonly used types
pub const Allocator = core.Allocator;

pub const MessageHandler = *const fn (bot: *DiscordBot, message: IncomingMessage) anyerror!void;

pub const IncomingMessage = struct {
    id: []const u8,
    content: []const u8,
    author_id: []const u8,
    author_username: []const u8,
    channel_id: []const u8,
    guild_id: ?[]const u8,
    timestamp: []const u8,
    is_bot: bool,

    pub fn deinit(self: IncomingMessage, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.content);
        allocator.free(self.author_id);
        allocator.free(self.author_username);
        allocator.free(self.channel_id);
        if (self.guild_id) |guild_id| allocator.free(guild_id);
        allocator.free(self.timestamp);
    }
};

pub const DiscordBot = struct {
    allocator: std.mem.Allocator,
    token: []const u8,
    rate_limit: RateLimiter,
    ws: ?std.websocket.Client = null,
    api_client: *discord_api.DiscordClient,
    ai_agent: *agent.Agent,
    learning_db: *database.Database,
    message_handler: ?MessageHandler = null,
    bot_user_id: ?[]const u8 = null,
    should_stop: bool = false,
    heartbeat_interval: ?u32 = null,
    sequence: ?u64 = null,

    pub fn init(allocator: std.mem.Allocator, token: []const u8, ai_agent: *agent.Agent, learning_db: *database.Database) !*DiscordBot {
        const config = discord_api.DiscordConfig{ .token = token };
        const api_client = try discord_api.DiscordClient.init(allocator, config);

        const bot = try allocator.create(DiscordBot);
        bot.* = .{
            .allocator = allocator,
            .token = token,
            .rate_limit = RateLimiter.init(50),
            .ws = null,
            .api_client = api_client,
            .ai_agent = ai_agent,
            .learning_db = learning_db,
        };

        return bot;
    }

    pub fn deinit(self: *DiscordBot) void {
        if (self.ws) |*w| w.deinit();
        self.api_client.deinit();
        if (self.bot_user_id) |id| self.allocator.free(id);
        self.allocator.destroy(self);
    }

    pub fn setMessageHandler(self: *DiscordBot, handler: MessageHandler) void {
        self.message_handler = handler;
    }

    pub fn connect(self: *DiscordBot) !void {
        const gateway_url = try fetchGatewayUrl(self.allocator, self.token);
        defer self.allocator.free(gateway_url);

        std.debug.print("🔗 Connecting to Discord Gateway: {s}\n", .{gateway_url});

        self.ws = try std.websocket.Client.connect(self.allocator, gateway_url, .{});
        var ws = &self.ws.?;

        var frame_buf: [4096]u8 = undefined;

        while (!self.should_stop) {
            const frame = ws.readFrame(&frame_buf) catch |err| {
                std.debug.print("❌ WebSocket read error: {}\n", .{err});
                break;
            };

            switch (frame.opcode) {
                .text => {
                    try handleTextFrame(self, frame.data);
                },
                .close => {
                    std.debug.print("🔌 Discord connection closed\n", .{});
                    break;
                },
                else => {},
            }
        }
    }

    pub fn disconnect(self: *DiscordBot) void {
        self.should_stop = true;
        if (self.ws) |*ws| {
            ws.close(.{}) catch {};
        }
    }

    pub fn sendMessage(self: *DiscordBot, channel_id: []const u8, content: []const u8) !void {
        if (!self.rate_limit.allow()) {
            std.debug.print("⚠️ Rate limited, skipping message\n", .{});
            return;
        }

        try self.api_client.sendText(channel_id, content);
    }
};

fn handleTextFrame(self: *DiscordBot, payload: []const u8) !void {
    var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, payload, .{}) catch |err| {
        std.debug.print("❌ JSON parse error: {}\n", .{err});
        return;
    };
    defer parsed.deinit();

    const root = &parsed.value;
    const op_value = root.object.get("op") orelse return;
    const op: types.OpCode = @enumFromInt(@as(u8, @intCast(op_value.integer)));

    // Update sequence number if present
    if (root.object.get("s")) |s| {
        if (s != .null) {
            self.sequence = @as(u64, @intCast(s.integer));
        }
    }

    switch (op) {
        .hello => try handleHello(self, root),
        .dispatch => try handleDispatch(self, root),
        .heartbeat_ack => {
            std.debug.print("💓 Heartbeat acknowledged\n", .{});
        },
        .invalid_session => {
            std.debug.print("❌ Invalid session, reconnecting...\n", .{});
            try identify(self);
        },
        .reconnect => {
            std.debug.print("🔄 Discord requested reconnect\n", .{});
            // In a full implementation, we'd reconnect here
        },
        else => {
            std.debug.print("🔍 Unhandled opcode: {}\n", .{op});
        },
    }
}

fn handleHello(self: *DiscordBot, root: *const std.json.Value) !void {
    if (root.object.get("d")) |data| {
        const iv = data.object.get("heartbeat_interval") orelse return;
        const interval = @as(u32, @intCast(iv.integer));
        self.heartbeat_interval = interval;

        std.debug.print("💓 Starting heartbeat with interval: {}ms\n", .{interval});

        // Start heartbeat in a separate thread (simplified for this example)
        try startHeartbeat(self, interval);
        try identify(self);
    }
}

fn handleDispatch(self: *DiscordBot, root: *const std.json.Value) !void {
    const event_type = root.object.get("t") orelse return;
    const event_name = event_type.string;

    if (std.mem.eql(u8, event_name, "READY")) {
        try handleReady(self, root);
    } else if (std.mem.eql(u8, event_name, "MESSAGE_CREATE")) {
        try handleMessageCreate(self, root);
    }
}

fn handleReady(self: *DiscordBot, root: *const std.json.Value) !void {
    const data = root.object.get("d") orelse return;
    const user = data.object.get("user") orelse return;
    const user_id = user.object.get("id") orelse return;

    self.bot_user_id = try self.allocator.dupe(u8, user_id.string);

    const username = user.object.get("username") orelse return;
    std.debug.print("🤖 Bot ready! Logged in as: {s} (ID: {s})\n", .{ username.string, user_id.string });
}

fn handleMessageCreate(self: *DiscordBot, root: *const std.json.Value) !void {
    const data = root.object.get("d") orelse return;

    // Parse message data
    const message_id = data.object.get("id") orelse return;
    const content = data.object.get("content") orelse return;
    const author = data.object.get("author") orelse return;
    const channel_id = data.object.get("channel_id") orelse return;

    const author_id = author.object.get("id") orelse return;
    const author_username = author.object.get("username") orelse return;
    const is_bot = if (author.object.get("bot")) |bot| bot.bool else false;
    const timestamp = data.object.get("timestamp") orelse return;

    // Skip our own messages and other bots
    if (is_bot) return;
    if (self.bot_user_id) |bot_id| {
        if (std.mem.eql(u8, author_id.string, bot_id)) return;
    }

    // Create incoming message
    const incoming_message = IncomingMessage{
        .id = try self.allocator.dupe(u8, message_id.string),
        .content = try self.allocator.dupe(u8, content.string),
        .author_id = try self.allocator.dupe(u8, author_id.string),
        .author_username = try self.allocator.dupe(u8, author_username.string),
        .channel_id = try self.allocator.dupe(u8, channel_id.string),
        .guild_id = if (data.object.get("guild_id")) |gid| try self.allocator.dupe(u8, gid.string) else null,
        .timestamp = try self.allocator.dupe(u8, timestamp.string),
        .is_bot = is_bot,
    };
    defer incoming_message.deinit(self.allocator);

    std.debug.print("📨 Message from {s}: {s}\n", .{ incoming_message.author_username, incoming_message.content });

    // Call message handler if set
    if (self.message_handler) |handler| {
        handler(self, incoming_message) catch |err| {
            std.debug.print("❌ Message handler error: {}\n", .{err});
        };
    }
}

fn startHeartbeat(self: *DiscordBot, _: u32) !void {
    // Simplified heartbeat - in production, this should run in a separate thread
    var payload_map = std.json.ObjectMap.init(self.allocator);
    defer payload_map.deinit();
    try payload_map.put("op", std.json.Value{ .integer = @intFromEnum(types.OpCode.heartbeat) });
    try payload_map.put("d", if (self.sequence) |seq| std.json.Value{ .integer = @intCast(seq) } else std.json.Value{ .null = {} });

    const heartbeat_payload = std.json.Value{ .object = payload_map };

    var buf = std.ArrayList(u8).init(self.allocator);
    defer buf.deinit();

    try std.json.stringify(heartbeat_payload, .{}, buf.writer());

    if (self.ws) |*ws| {
        try ws.writeFrame(.{ .fin = true, .opcode = .text, .data = buf.items });
        std.debug.print("💓 Heartbeat sent\n", .{});
    }
}

fn identify(self: *DiscordBot) !void {
    var ws = &self.ws.?;

    const intents = 1 << 9 | 1 << 15; // GUILD_MESSAGES | MESSAGE_CONTENT

    var properties = std.json.ObjectMap.init(self.allocator);
    defer properties.deinit();
    try properties.put("os", std.json.Value{ .string = "windows" });
    try properties.put("browser", std.json.Value{ .string = "abi-bot" });
    try properties.put("device", std.json.Value{ .string = "abi-bot" });

    var identify_data = std.json.ObjectMap.init(self.allocator);
    defer identify_data.deinit();
    try identify_data.put("token", std.json.Value{ .string = self.token });
    try identify_data.put("intents", std.json.Value{ .integer = intents });
    try identify_data.put("properties", std.json.Value{ .object = properties });

    var payload = std.json.ObjectMap.init(self.allocator);
    defer payload.deinit();
    try payload.put("op", std.json.Value{ .integer = @intFromEnum(types.OpCode.identify) });
    try payload.put("d", std.json.Value{ .object = identify_data });

    var buf = std.ArrayList(u8).init(self.allocator);
    defer buf.deinit();
    try std.json.stringify(std.json.Value{ .object = payload }, .{}, buf.writer());

    try ws.writeFrame(.{ .fin = true, .opcode = .text, .data = buf.items });
    std.debug.print("🆔 Identify payload sent\n", .{});
}

fn fetchGatewayUrl(allocator: std.mem.Allocator, token: []const u8) ![]const u8 {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const auth_value = try std.fmt.allocPrint(allocator, "Bot {s}", .{token});
    defer allocator.free(auth_value);

    const headers = [_]std.http.Header{.{ .name = "Authorization", .value = auth_value }};
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();

    const result = try client.fetch(.{ .location = .{ .url = "https://discord.com/api/v10/gateway" }, .extra_headers = &headers, .response_storage = .dynamic(&buf) });

    if (result.status != .ok) return error.BadResponse;

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf.items, .{});
    defer parsed.deinit();
    const url_val = parsed.value.object.get("url") orelse return error.BadResponse;

    // Add WebSocket protocol parameters
    const gateway_url = try std.fmt.allocPrint(allocator, "{s}/?v=10&encoding=json", .{url_val.string});
    return gateway_url;
}

const RateLimiter = struct {
    capacity: u32,
    tokens: u32,
    last: i128,

    pub fn init(capacity: u32) RateLimiter {
        return .{ .capacity = capacity, .tokens = capacity, .last = std.time.microTimestamp() };
    }

    pub fn allow(self: *RateLimiter) bool {
        const now = std.time.microTimestamp();
        const elapsed = now - self.last;
        const replenish = @min(self.capacity - self.tokens, @as(u32, @intCast(elapsed / 1_000_000)) * self.capacity);
        self.tokens += replenish;
        self.last = now;
        if (self.tokens == 0) return false;
        self.tokens -= 1;
        return true;
    }
};
