//! Discord gateway bot loop (WDBX Rust `src/features/discord.rs`).
//!
//! A real inbound→outbound chat surface: a WebSocket gateway connection in, REST
//! reply out, with prefix-scoped, bot-safe, token-safe command routing. This is
//! the capability ABI's `src/connectors/discord.zig` does *not* have today
//! (that connector validates credentials + message shape with an explicit live
//! transport boundary but has no gateway loop). See
//! `docs/spec/wdbx-rust-capability-extract.mdx` §7.
//!
//! Design points ported faithfully:
//!   - Protocol constants: gateway URL, API base, default intents bitmask.
//!   - `run` loop: Hello (op 10) → Identify (op 2) → heartbeat on op 1 /
//!     late Hello → sequence tracking on dispatch → MESSAGE_CREATE handling
//!     with bot-authored-message ignore.
//!   - Command routing is pure and testable (`parseDiscordCommand` /
//!     `routeDiscordMessage`); replies truncate to 1900 chars.
//!   - Token hygiene: replies never embed the token; `token_configured` drives
//!     the status reply without exposing the secret.
//!
//! Honesty note: the live `WebSocketTransport` performs a real WebSocket
//! handshake + masked-frame I/O over a `std.Io.net.Stream`, but TLS termination
//! is **not** linked (Discord requires wss). It is reference-grade and must run
//! behind a TLS-terminating proxy to reach the real gateway. The offline
//! `FakeTransport` drives the full loop logic in tests with no network.

const std = @import("std");
const connector = @import("connector.zig");
const http = @import("http.zig");
const json = @import("json.zig");
const discord = @import("discord.zig");
const ws_client = @import("discord_ws_client.zig");
const routing = @import("discord_routing.zig");

pub const ConnectorError = connector.ConnectorError;

// Re-export extracted symbols for backward compatibility.
pub const WebSocketClient = ws_client.WebSocketClient;
pub const DiscordCommand = routing.DiscordCommand;
pub const MAX_MESSAGE_CONTENT_BYTES = routing.MAX_MESSAGE_CONTENT_BYTES;
pub const parseDiscordCommand = routing.parseDiscordCommand;
pub const routeDiscordMessage = routing.routeDiscordMessage;
pub const truncate = routing.truncate;
pub const promptSummary = routing.promptSummary;
pub const governanceSummary = routing.governanceSummary;

/// Discord gateway WebSocket endpoint (v10, JSON encoding).
pub const GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json";
/// Discord REST API base.
pub const API_BASE = "https://discord.com/api/v10";
/// Default gateway intents: guild messages (1<<9) | DMs (1<<12) | message content (1<<15).
pub const DEFAULT_INTENTS: u32 = (1 << 9) | (1 << 12) | (1 << 15);

/// Gateway run configuration.
pub const GatewayConfig = struct {
    token: []const u8,
    token_configured: bool,
    intents: u32 = DEFAULT_INTENTS,
    prefix: []const u8 = "!",
};

/// Outcome of a gateway run.
pub const GatewayStats = struct {
    message_events: usize = 0,
    replies_sent: usize = 0,
    heartbeats_sent: usize = 0,
    identified: bool = false,
};

/// The bot loop driver. Holds configuration; `run` performs the turn loop over
/// an injected transport (so it is testable without a live socket).
pub const Gateway = struct {
    config: GatewayConfig,

    pub fn init(config: GatewayConfig) Gateway {
        return .{ .config = config };
    }

    /// Run the gateway loop over `transport`. `max_message_events` bounds the
    /// number of gateway messages processed (used by tests; `null` = unbounded).
    /// The transport must implement:
    ///   `readMessage(allocator) !?[]u8`     (owned JSON message or null)
    ///   `sendText(allocator, text) !void`   (send a JSON text frame)
    ///   `sendReply(allocator, channel_id, content) !void` (REST reply)
    ///   `close() void`
    pub fn run(
        self: *const Gateway,
        allocator: std.mem.Allocator,
        transport: anytype,
        max_message_events: ?usize,
    ) !GatewayStats {
        var stats = GatewayStats{};
        var seq: ?i64 = null;

        // 1. Hello (op 10) -> read heartbeat interval, then Identify.
        const hello = (try transport.readMessage(allocator)) orelse return error.GatewayClosedEarly;
        defer allocator.free(hello);
        _ = parseHeartbeatInterval(allocator, hello) orelse return error.MissingHeartbeatInterval;

        const identify = try buildIdentify(allocator, self.config);
        defer allocator.free(identify);
        try transport.sendText(allocator, identify);
        stats.identified = true;

        while (true) {
            if (max_message_events) |m| if (stats.message_events >= m) break;

            const msg = (try transport.readMessage(allocator)) orelse break;
            defer allocator.free(msg);
            stats.message_events += 1;

            const op = parseOp(allocator, msg) orelse continue;
            switch (op) {
                1 => {
                    // Heartbeat request -> echo op 1 with the last seen sequence.
                    const hb = try buildHeartbeat(allocator, seq);
                    defer allocator.free(hb);
                    try transport.sendText(allocator, hb);
                    stats.heartbeats_sent += 1;
                },
                10 => {
                    // Late/reconnect Hello -> re-Identify.
                    const identify2 = try buildIdentify(allocator, self.config);
                    defer allocator.free(identify2);
                    try transport.sendText(allocator, identify2);
                    stats.identified = true;
                },
                11 => {
                    // Heartbeat ack -> no-op.
                },
                0 => {
                    // Dispatch: track sequence, handle MESSAGE_CREATE.
                    seq = parseSeq(allocator, msg) orelse seq;
                    if (try extractMessageCreate(allocator, msg)) |mc| {
                        defer allocator.free(mc.channel_id);
                        defer allocator.free(mc.content);
                        if (mc.author_bot) continue; // ignore bot-authored messages

                        const reply = (try routeDiscordMessage(
                            allocator,
                            mc.content,
                            self.config.prefix,
                            self.config.token_configured,
                        )) orelse continue;
                        defer allocator.free(reply);

                        const truncated = try truncate(allocator, reply, MAX_MESSAGE_CONTENT_BYTES);
                        defer allocator.free(truncated);

                        try transport.sendReply(allocator, mc.channel_id, truncated);
                        stats.replies_sent += 1;
                    }
                },
                else => {},
            }
        }

        transport.close();
        return stats;
    }
};

// ---------------------------------------------------------------------------
// Gateway message (de)serialization helpers
// ---------------------------------------------------------------------------

fn parseOp(allocator: std.mem.Allocator, msg: []const u8) ?i64 {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, msg, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const v = parsed.value.object.get("op") orelse return null;
    return if (v == .integer) v.integer else null;
}

fn parseSeq(allocator: std.mem.Allocator, msg: []const u8) ?i64 {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, msg, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const v = parsed.value.object.get("s") orelse return null;
    return if (v == .integer) v.integer else null;
}

fn parseHeartbeatInterval(allocator: std.mem.Allocator, msg: []const u8) ?i64 {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, msg, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const d = parsed.value.object.get("d") orelse return null;
    if (d != .object) return null;
    const v = d.object.get("heartbeat_interval") orelse return null;
    return if (v == .integer) v.integer else if (v == .float) @intFromFloat(v.float) else null;
}

const MessageCreate = struct {
    channel_id: []u8,
    content: []u8,
    author_bot: bool,
};

/// Extract a MESSAGE_CREATE dispatch's channel id, content, and author.bot flag.
/// Returns owned copies, or `null` when the message is not a MESSAGE_CREATE or
/// is missing the required fields.
fn extractMessageCreate(allocator: std.mem.Allocator, msg: []const u8) !?MessageCreate {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, msg, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return null;

    const op_v = parsed.value.object.get("op") orelse return null;
    const op = if (op_v == .integer) op_v.integer else -1;
    const t_v = parsed.value.object.get("t") orelse return null;
    const t = if (t_v == .string) t_v.string else null;
    if (op != 0 or t == null or !std.mem.eql(u8, t.?, "MESSAGE_CREATE")) return null;

    const d = parsed.value.object.get("d") orelse return null;
    if (d != .object) return null;

    const channel_v = d.object.get("channel_id") orelse return null;
    const channel_id = if (channel_v == .string) channel_v.string else null;
    const content_v = d.object.get("content") orelse return null;
    const content = if (content_v == .string) content_v.string else null;
    if (channel_id == null or content == null) return null;

    const author_v = d.object.get("author") orelse null;
    const author = if (author_v) |a| (if (a == .object) a else null) else null;
    const bot_v = if (author) |a| a.object.get("bot") else null;
    const author_bot = if (bot_v) |b| (if (b == .bool) b.bool else false) else false;

    return .{
        .channel_id = try allocator.dupe(u8, channel_id.?),
        .content = try allocator.dupe(u8, content.?),
        .author_bot = author_bot,
    };
}

fn buildIdentify(allocator: std.mem.Allocator, config: GatewayConfig) ![]u8 {
    try discord.validateToken(config.token);
    // Route through the canonical JSON-string escaper (json.zig) rather than
    // interpolating the token raw: validateToken accepts printable-non-
    // whitespace bytes, which still includes `"` and `\` — unescaped, either
    // would corrupt this payload.
    return json.buildDiscordIdentifyBody(allocator, config.token, config.intents);
}

fn buildHeartbeat(allocator: std.mem.Allocator, seq: ?i64) ![]u8 {
    if (seq) |s| {
        return std.fmt.allocPrint(allocator, "{{\"op\":1,\"d\":{d}}}", .{s});
    }
    return std.fmt.allocPrint(allocator, "{{\"op\":1,\"d\":null}}", .{});
}

// ---------------------------------------------------------------------------
// Offline transport (used by tests) — drives the loop with scripted messages.
// ---------------------------------------------------------------------------

pub const FakeTransport = struct {
    allocator: std.mem.Allocator,
    incoming: std.ArrayListUnmanaged([]u8) = .empty,
    outgoing: std.ArrayListUnmanaged([]u8) = .empty,
    replies: std.ArrayListUnmanaged(ReplyRecord) = .empty,
    closed: bool = false,
    next_read_index: usize = 0,

    pub const ReplyRecord = struct {
        channel_id: []u8,
        content: []u8,
    };

    pub fn init(allocator: std.mem.Allocator) FakeTransport {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *FakeTransport) void {
        for (self.incoming.items) |m| self.allocator.free(m);
        for (self.outgoing.items) |m| self.allocator.free(m);
        for (self.replies.items) |r| {
            self.allocator.free(r.channel_id);
            self.allocator.free(r.content);
        }
        self.incoming.deinit(self.allocator);
        self.outgoing.deinit(self.allocator);
        self.replies.deinit(self.allocator);
    }

    pub fn enqueue(self: *FakeTransport, msg: []const u8) !void {
        try self.incoming.append(self.allocator, try self.allocator.dupe(u8, msg));
    }

    pub fn readMessage(self: *FakeTransport, allocator: std.mem.Allocator) !?[]u8 {
        _ = allocator;
        if (self.next_read_index >= self.incoming.items.len) return null;
        // `orderedRemove` transfers ownership of the slice out of the list (the
        // next message shifts into this slot), so only the caller frees it.
        return self.incoming.orderedRemove(self.next_read_index);
    }

    pub fn sendText(self: *FakeTransport, allocator: std.mem.Allocator, text: []const u8) !void {
        try self.outgoing.append(allocator, try allocator.dupe(u8, text));
    }

    pub fn sendReply(self: *FakeTransport, allocator: std.mem.Allocator, channel_id: []const u8, content: []const u8) !void {
        try self.replies.append(allocator, .{
            .channel_id = try allocator.dupe(u8, channel_id),
            .content = try allocator.dupe(u8, content),
        });
    }

    pub fn close(self: *FakeTransport) void {
        self.closed = true;
    }
};

// ---------------------------------------------------------------------------
// Live WebSocket transport (reference-grade; TLS not linked)
// ---------------------------------------------------------------------------

/// Live transport wrapping a `WebSocketClient`; replies go out over REST.
pub const WebSocketTransport = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    ws: *WebSocketClient,
    config: GatewayConfig,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, ws: *WebSocketClient, config: GatewayConfig) WebSocketTransport {
        return .{ .allocator = allocator, .io = io, .ws = ws, .config = config };
    }

    pub fn readMessage(self: *WebSocketTransport, allocator: std.mem.Allocator) !?[]u8 {
        return self.ws.readMessage(allocator);
    }

    pub fn sendText(self: *WebSocketTransport, allocator: std.mem.Allocator, text: []const u8) !void {
        try self.ws.sendText(allocator, text);
    }

    pub fn sendReply(self: *WebSocketTransport, allocator: std.mem.Allocator, channel_id: []const u8, content: []const u8) !void {
        // `channel_id` originates from parsed (attacker-influenceable) gateway
        // JSON and is interpolated straight into the REST path below; reject
        // anything that isn't a plain snowflake ID before it reaches the URL,
        // mirroring the same gate `discord.zig`'s own send paths apply.
        try discord.validateDiscordId(channel_id);
        try discord.validateMessageContent(content);
        const body = try json.buildDiscordMessageBody(allocator, content);
        defer allocator.free(body);
        const authorization = try http.botHeader(allocator, self.config.token);
        defer allocator.free(authorization);
        const path = try std.fmt.allocPrint(allocator, "/api/v10/channels/{s}/messages", .{channel_id});
        defer allocator.free(path);
        _ = http.httpPostJson(
            self.io,
            allocator,
            .{ .api_key = self.config.token, .base_url = API_BASE, .timeout_ms = 30000, .transport = .live },
            path,
            body,
            &[_]std.http.Header{.{ .name = "authorization", .value = authorization }},
        ) catch |err| {
            std.log.warn("discord gateway reply failed: {s}", .{@errorName(err)});
        };
    }

    pub fn close(self: *WebSocketTransport) void {
        self.ws.close();
    }
};

/// Connect to the gateway and run the bot loop over the live transport.
/// Reference-grade: requires a TLS-terminating proxy in front of the gateway
/// because TLS is not linked in this client.
pub fn connectAndRun(
    io: std.Io,
    allocator: std.mem.Allocator,
    config: GatewayConfig,
    max_message_events: ?usize,
) !GatewayStats {
    var ws = try WebSocketClient.connect(io, allocator, "gateway.discord.gg", 443);
    defer ws.deinit(allocator);
    var transport = WebSocketTransport.init(allocator, io, &ws, config);
    const gw = Gateway.init(config);
    return gw.run(allocator, &transport, max_message_events);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test {
    std.testing.refAllDecls(@This());
}

test "gateway loop runs hello/identify/heartbeat/message_create offline" {
    const allocator = std.testing.allocator;
    var transport = FakeTransport.init(allocator);
    defer transport.deinit();

    // Hello with heartbeat interval.
    try transport.enqueue(
        \\{"op":10,"d":{"heartbeat_interval":45000},"s":null,"t":null}
    );
    // A heartbeat request from the server.
    try transport.enqueue(
        \\{"op":1,"d":null,"s":null,"t":null}
    );
    // MESSAGE_CREATE from a human user.
    try transport.enqueue(
        \\{"op":0,"s":101,"t":"MESSAGE_CREATE","d":{"channel_id":"123","author":{"bot":false},"content":"!governance"}}
    );
    // MESSAGE_CREATE from a bot -> must be ignored.
    try transport.enqueue(
        \\{"op":0,"s":102,"t":"MESSAGE_CREATE","d":{"channel_id":"999","author":{"bot":true},"content":"!help"}}
    );

    const gw = Gateway.init(.{ .token = "SECRET_TOKEN", .token_configured = true });
    const stats = try gw.run(allocator, &transport, null);

    // Identified once (Hello), one heartbeat reply, one real reply (bot msg ignored).
    try std.testing.expect(stats.identified);
    try std.testing.expectEqual(@as(usize, 1), stats.heartbeats_sent);
    try std.testing.expectEqual(@as(usize, 1), stats.replies_sent);
    // Outgoing: Identify + Heartbeat.
    try std.testing.expectEqual(@as(usize, 2), transport.outgoing.items.len);
    // The reply went to the human's channel, not the bot's.
    try std.testing.expectEqualStrings("123", transport.replies.items[0].channel_id);
    // Reply never contains the token.
    try std.testing.expect(std.mem.indexOf(u8, transport.replies.items[0].content, "SECRET_TOKEN") == null);
    try std.testing.expect(transport.closed);
}

test "gateway loop honors max_message_events bound" {
    const allocator = std.testing.allocator;
    var transport = FakeTransport.init(allocator);
    defer transport.deinit();

    try transport.enqueue(
        \\{"op":10,"d":{"heartbeat_interval":45000},"s":null,"t":null}
    );
    try transport.enqueue(
        \\{"op":0,"s":1,"t":"MESSAGE_CREATE","d":{"channel_id":"1","author":{"bot":false},"content":"!help"}}
    );
    try transport.enqueue(
        \\{"op":0,"s":2,"t":"MESSAGE_CREATE","d":{"channel_id":"2","author":{"bot":false},"content":"!status"}}
    );

    const gw = Gateway.init(.{ .token = "T", .token_configured = false });
    // Only the first 2 messages (Hello + !help) are processed before the bound.
    const stats = try gw.run(allocator, &transport, 2);
    try std.testing.expectEqual(@as(usize, 2), stats.message_events);
}

test "gateway identify escapes a token containing JSON-breaking characters" {
    const allocator = std.testing.allocator;
    var transport = FakeTransport.init(allocator);
    defer transport.deinit();

    try transport.enqueue(
        \\{"op":10,"d":{"heartbeat_interval":45000},"s":null,"t":null}
    );

    // A token containing a quote and a backslash must not corrupt the
    // Identify frame — it must come through escaped, not interpolated raw.
    const gw = Gateway.init(.{ .token = "weird\"token\\value", .token_configured = true });
    const stats = try gw.run(allocator, &transport, 1);
    try std.testing.expect(stats.identified);

    const identify = transport.outgoing.items[0];
    // The whole frame must still parse as valid JSON (proves the token was
    // escaped rather than breaking the surrounding structure)...
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, identify, .{});
    defer parsed.deinit();
    // ...and the token must round-trip to its original, unescaped value.
    const token = parsed.value.object.get("d").?.object.get("token").?.string;
    try std.testing.expectEqualStrings("weird\"token\\value", token);
}

test "gateway run rejects an empty token before sending anything" {
    const allocator = std.testing.allocator;
    var transport = FakeTransport.init(allocator);
    defer transport.deinit();

    try transport.enqueue(
        \\{"op":10,"d":{"heartbeat_interval":45000},"s":null,"t":null}
    );

    const gw = Gateway.init(.{ .token = "", .token_configured = false });
    try std.testing.expectError(error.AuthenticationError, gw.run(allocator, &transport, null));
    try std.testing.expectEqual(@as(usize, 0), transport.outgoing.items.len);
}
