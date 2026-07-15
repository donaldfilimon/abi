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

pub const ConnectorError = connector.ConnectorError;

/// Fill `buf` with pseudo-random bytes (reference-grade xorshift; not a CSPRNG
/// — the live path only needs a non-deterministic-looking WebSocket key and
/// frame mask, which this satisfies).
var rng_state: u64 = 0x9E3779B97F4A7C15;
fn fillRandom(buf: []u8) void {
    var s = rng_state;
    for (buf) |*b| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        b.* = @intCast(s & 0xFF);
    }
    rng_state = s;
}

/// Discord gateway WebSocket endpoint (v10, JSON encoding).
pub const GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json";
/// Discord REST API base.
pub const API_BASE = "https://discord.com/api/v10";
/// Default gateway intents: guild messages (1<<9) | DMs (1<<12) | message content (1<<15).
pub const DEFAULT_INTENTS: u32 = (1 << 9) | (1 << 12) | (1 << 15);
/// Replies are truncated to this many bytes (Discord message limit is 2000).
pub const MAX_MESSAGE_CONTENT_BYTES: usize = 1900;

/// Parsed Discord command from a message body.
pub const DiscordCommand = enum {
    help,
    status,
    prompt,
    governance,
    unknown,
    not_for_abbey,
};

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
// Command routing (pure, testable)
// ---------------------------------------------------------------------------

/// Parse a command from a message body given a command prefix. An empty prefix
/// captures nothing (so the bot never replies to every message); a message
/// without the prefix is `not_for_abbey` (no reply).
pub fn parseDiscordCommand(content: []const u8, prefix: []const u8) DiscordCommand {
    if (prefix.len == 0) return .not_for_abbey;
    if (!std.mem.startsWith(u8, content, prefix)) return .not_for_abbey;
    const rest = content[prefix.len..];
    var it = std.mem.tokenizeScalar(u8, rest, ' ');
    const cmd = it.next() orelse return .unknown;
    if (std.ascii.eqlIgnoreCase(cmd, "help")) return .help;
    if (std.ascii.eqlIgnoreCase(cmd, "status")) return .status;
    if (std.ascii.eqlIgnoreCase(cmd, "prompt")) return .prompt;
    if (std.ascii.eqlIgnoreCase(cmd, "governance")) return .governance;
    return .unknown;
}

fn helpText() []const u8 {
    return "Available commands: !help, !status, !prompt, !governance";
}

/// Deterministic local prompt summary (reference routing reply).
pub fn promptSummary() []const u8 {
    return "prompt summary: local persona routing (Abbey/Aviva/Abi) via keyword sentiment; completions recorded to WDBX.";
}

/// Deterministic local governance summary (reference routing reply).
pub fn governanceSummary() []const u8 {
    return "governance: six-principle constitutional audit (truthfulness, safety, helpfulness, fairness, privacy, transparency) with a weighted E-score and a hard safety-class veto.";
}

fn statusText(token_configured: bool) []const u8 {
    if (token_configured) return "status: connected (token configured)";
    return "status: offline (no token configured)";
}

/// Route a parsed command to a reply string (owned). Returns `null` when no
/// reply should be sent (e.g. a message that is not for Abbey).
pub fn routeDiscordMessage(
    allocator: std.mem.Allocator,
    content: []const u8,
    prefix: []const u8,
    token_configured: bool,
) !?[]u8 {
    const cmd = parseDiscordCommand(content, prefix);
    return switch (cmd) {
        .not_for_abbey => null,
        .unknown => try allocator.dupe(u8, "Unknown command. Type `!help` for available commands."),
        .help => try allocator.dupe(u8, helpText()),
        .status => try allocator.dupe(u8, statusText(token_configured)),
        .prompt => try allocator.dupe(u8, promptSummary()),
        .governance => try allocator.dupe(u8, governanceSummary()),
    };
}

/// Truncate `text` to at most `max` bytes without splitting a UTF-8 sequence.
pub fn truncate(allocator: std.mem.Allocator, text: []const u8, max: usize) ![]u8 {
    if (text.len <= max) return try allocator.dupe(u8, text);
    var end = max;
    while (end > 0 and (text[end] & 0xC0) == 0x80) end -= 1; // back up over UTF-8 continuation
    return try allocator.dupe(u8, text[0..end]);
}

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
    return std.fmt.allocPrint(
        allocator,
        "{{\"op\":2,\"d\":{{\"token\":\"{s}\",\"intents\":{d},\"properties\":{{\"os\":\"abi\",\"browser\":\"abi\",\"device\":\"abi\"}}}}}}",
        .{ config.token, config.intents },
    );
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

/// Minimal WebSocket client over a connected `std.Io.net.Stream`. Performs the
/// HTTP upgrade handshake and masks client→server frames. Server→client frames
/// (text / close / ping / pong) are parsed. Honesty note: TLS termination is
/// not linked, so this must sit behind a TLS-terminating proxy to reach a real
/// wss:// gateway.
pub const WebSocketClient = struct {
    stream: std.Io.net.Stream,
    io: std.Io,
    allocator: std.mem.Allocator,
    read_buf: std.ArrayListUnmanaged(u8) = .empty,
    write_buf: []u8,
    closed: bool = false,

    pub fn connect(io: std.Io, allocator: std.mem.Allocator, host: []const u8, port: u16) !WebSocketClient {
        const addr = try std.Io.net.IpAddress.parseIp4(host, port);
        const stream = try addr.connect(io, .{ .mode = .stream });
        var client = WebSocketClient{
            .stream = stream,
            .io = io,
            .allocator = allocator,
            .write_buf = try allocator.alloc(u8, 8192),
        };
        try client.handshake();
        return client;
    }

    pub fn deinit(self: *WebSocketClient, allocator: std.mem.Allocator) void {
        if (!self.closed) {
            self.stream.close(self.io);
            self.closed = true;
        }
        self.read_buf.deinit(allocator);
        allocator.free(self.write_buf);
    }

    fn writeAll(self: *WebSocketClient, data: []const u8) !void {
        var w = self.stream.writer(self.io, self.write_buf);
        try w.interface.writeAll(data);
    }

    fn readSome(self: *WebSocketClient) !usize {
        var buf: [4096]u8 = undefined;
        var iovec: [1][]u8 = .{buf[0..]};
        const n = try self.stream.read(self.io, &iovec);
        if (n == 0) return error.SocketClosed;
        try self.read_buf.appendSlice(self.allocator, buf[0..n]);
        return n;
    }

    fn handshake(self: *WebSocketClient) !void {
        var key_bytes: [16]u8 = undefined;
        fillRandom(&key_bytes);
        var key_b64: [28]u8 = undefined;
        const key_len = std.base64.standard.Encoder.calcSize(key_bytes.len);
        _ = key_len;
        const encoded = std.base64.standard.Encoder.encode(&key_b64, &key_bytes);
        const request = try std.fmt.allocPrint(
            self.allocator,
            "GET /?v=10&encoding=json HTTP/1.1\r\n" ++
                "Host: gateway.discord.gg\r\n" ++
                "Upgrade: websocket\r\n" ++
                "Connection: Upgrade\r\n" ++
                "Sec-WebSocket-Key: {s}\r\n" ++
                "Sec-WebSocket-Version: 13\r\n\r\n",
            .{encoded},
        );
        defer self.allocator.free(request);
        try self.writeAll(request);

        // Read until the end of the response headers.
        while (true) {
            if (std.mem.indexOf(u8, self.read_buf.items, "\r\n\r\n") != null) break;
            _ = try self.readSome();
        }
        const headers = self.read_buf.items;
        if (std.mem.indexOf(u8, headers, "101") == null) {
            return error.WebSocketHandshakeFailed;
        }
        self.read_buf.clearRetainingCapacity();
    }

    fn writeFrame(self: *WebSocketClient, opcode: u8, payload: []const u8) !void {
        var header: [10]u8 = undefined;
        var hlen: usize = 0;
        header[0] = 0x80 | opcode; // FIN + opcode
        const len = payload.len;
        var mask: [4]u8 = undefined;
        fillRandom(&mask);
        if (len < 126) {
            header[1] = 0x80 | @as(u8, @intCast(len));
            hlen = 2;
        } else if (len <= 0xFFFF) {
            header[1] = 0x80 | 126;
            std.mem.writeInt(u16, header[2..4], @intCast(len), .big);
            hlen = 4;
        } else {
            header[1] = 0x80 | 127;
            std.mem.writeInt(u64, header[2..10], @intCast(len), .big);
            hlen = 10;
        }
        try self.writeAll(header[0..hlen]);
        try self.writeAll(&mask);
        const masked = try self.allocator.alloc(u8, len);
        defer self.allocator.free(masked);
        for (payload, 0..) |b, i| masked[i] = b ^ mask[i % 4];
        try self.writeAll(masked);
    }

    fn readFrame(self: *WebSocketClient, allocator: std.mem.Allocator) !?[]u8 {
        while (true) {
            if (self.read_buf.items.len >= 2) {
                const b0 = self.read_buf.items[0];
                const b1 = self.read_buf.items[1];
                const opcode = b0 & 0x0F;
                const masked = (b1 & 0x80) != 0;
                var len: usize = b1 & 0x7F;
                var off: usize = 2;
                if (len == 126) {
                    if (self.read_buf.items.len < 4) {
                        _ = try self.readSome();
                        continue;
                    }
                    len = std.mem.readInt(u16, self.read_buf.items[2..4], .big);
                    off = 4;
                } else if (len == 127) {
                    if (self.read_buf.items.len < 10) {
                        _ = try self.readSome();
                        continue;
                    }
                    len = @intCast(std.mem.readInt(u64, self.read_buf.items[2..10], .big));
                    off = 10;
                }
                var mask_key: ?[4]u8 = null;
                if (masked) {
                    if (self.read_buf.items.len < off + 4) {
                        _ = try self.readSome();
                        continue;
                    }
                    mask_key = self.read_buf.items[off..][0..4].*;
                    off += 4;
                }
                if (self.read_buf.items.len < off + len) {
                    _ = try self.readSome();
                    continue;
                }
                const payload = self.read_buf.items[off .. off + len];
                var out = try allocator.alloc(u8, len);
                if (mask_key) |mk| {
                    for (payload, 0..) |p, i| out[i] = p ^ mk[i % 4];
                } else {
                    @memcpy(out, payload);
                }
                self.read_buf.replaceRange(allocator, 0, off + len, &[_]u8{}) catch {};
                switch (opcode) {
                    0x8 => { // close
                        allocator.free(out);
                        return null;
                    },
                    0x9 => { // ping -> respond pong
                        allocator.free(out);
                        try self.writeFrame(0xA, &[_]u8{});
                        continue;
                    },
                    0xA => { // pong
                        allocator.free(out);
                        continue;
                    },
                    0x1 => return out, // text
                    else => {
                        allocator.free(out);
                        continue;
                    },
                }
            }
            _ = try self.readSome();
        }
    }

    /// Read the next gateway JSON message, or null when the socket closes.
    pub fn readMessage(self: *WebSocketClient, allocator: std.mem.Allocator) !?[]u8 {
        if (self.closed) return null;
        return try self.readFrame(allocator);
    }

    /// Send a JSON text frame (the gateway protocol payload).
    pub fn sendText(self: *WebSocketClient, _: std.mem.Allocator, text: []const u8) !void {
        if (self.closed) return error.SocketClosed;
        try self.writeFrame(0x1, text);
    }

    /// Send a close frame and tear down the socket.
    pub fn close(self: *WebSocketClient) void {
        if (self.closed) return;
        self.writeFrame(0x8, &[_]u8{}) catch {};
        self.stream.close(self.io);
        self.closed = true;
    }
};

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

test "parseDiscordCommand honors prefix and empty-prefix safety" {
    try std.testing.expect(parseDiscordCommand("!help", "!") == .help);
    try std.testing.expect(parseDiscordCommand("!status", "!") == .status);
    try std.testing.expect(parseDiscordCommand("!prompt", "!") == .prompt);
    try std.testing.expect(parseDiscordCommand("!governance", "!") == .governance);
    try std.testing.expect(parseDiscordCommand("!nonsense", "!") == .unknown);
    // No prefix -> never captures (avoids replying to every message).
    try std.testing.expect(parseDiscordCommand("help", "") == .not_for_abbey);
    // Prefix absent -> not for abbey.
    try std.testing.expect(parseDiscordCommand("help me", "!") == .not_for_abbey);
}

test "routeDiscordMessage produces reply strings and null for non-abbey" {
    const allocator = std.testing.allocator;
    const none = try routeDiscordMessage(allocator, "hello there", "!", true);
    try std.testing.expect(none == null);
    const status = (try routeDiscordMessage(allocator, "!status", "!", true)) orelse return error.UnexpectedNull;
    defer allocator.free(status);
    try std.testing.expect(std.mem.indexOf(u8, status, "connected") != null);
    const status_off = (try routeDiscordMessage(allocator, "!status", "!", false)) orelse return error.UnexpectedNull;
    defer allocator.free(status_off);
    try std.testing.expect(std.mem.indexOf(u8, status_off, "offline") != null);
    const gov = (try routeDiscordMessage(allocator, "!governance", "!", true)) orelse return error.UnexpectedNull;
    defer allocator.free(gov);
    try std.testing.expect(std.mem.indexOf(u8, gov, "constitutional") != null);
}

test "truncate respects byte and utf8 boundaries" {
    const allocator = std.testing.allocator;
    const short = try truncate(allocator, "hello", 1900);
    defer allocator.free(short);
    try std.testing.expectEqualStrings("hello", short);

    // A multi-byte char at the boundary must not be split.
    const t = "ééééé"; // 5 bytes * 2 = 10 bytes total; truncate to 5 -> 2 chars (4 bytes).
    const cut = try truncate(allocator, t, 5);
    defer allocator.free(cut);
    try std.testing.expectEqual(@as(usize, 4), cut.len);
    try std.testing.expectEqualStrings("éé", cut);
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
