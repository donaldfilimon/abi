//! Discord Gateway WebSocket Client
//!
//! Manages the WebSocket connection to the Discord Gateway for receiving
//! real-time events (messages, interactions, presence updates, etc.).
//!
//! Usage:
//! ```zig
//! var handler = GatewayEventHandler{
//!     .ctx = my_context,
//!     .on_message_create = handleMessage,
//! };
//! var client = GatewayClient.init(allocator, bot_token, intents, handler);
//! defer client.deinit();
//! try client.connect();
//! ```

const std = @import("std");
const gateway_types = @import("gateway_types.zig");
const json_utils = @import("../../foundation/mod.zig").utils.json;

pub const GatewayState = gateway_types.GatewayState;
pub const GatewayEventHandler = gateway_types.GatewayEventHandler;
pub const GatewayOpcode = gateway_types.GatewayOpcode;
pub const HelloPayload = gateway_types.HelloPayload;

pub const GatewayError = error{
    InvalidPayload,
    ConnectionFailed,
    AlreadyConnected,
    NotConnected,
};

fn appendJsonString(allocator: std.mem.Allocator, json: *std.ArrayListUnmanaged(u8), value: []const u8) !void {
    try json.append(allocator, '"');
    try json_utils.appendJsonEscaped(allocator, json, value);
    try json.append(allocator, '"');
}

/// Build an offline IDENTIFY payload. Transport over a WebSocket is deferred to
/// a future networking wave; callers can use this for deterministic validation.
pub fn buildIdentifyPayload(
    allocator: std.mem.Allocator,
    token: []const u8,
    intents: u32,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.print(allocator, "{{\"op\":{d},\"d\":{{\"token\":", .{@intFromEnum(GatewayOpcode.IDENTIFY)});
    try appendJsonString(allocator, &json, token);
    try json.print(
        allocator,
        ",\"intents\":{d},\"properties\":{{\"os\":\"zig\",\"browser\":\"abi\",\"device\":\"abi\"}}}}}}",
        .{intents},
    );

    return try json.toOwnedSlice(allocator);
}

/// Build an offline HEARTBEAT payload using the last Gateway sequence number.
pub fn buildHeartbeatPayload(allocator: std.mem.Allocator, sequence_number: ?u64) ![]u8 {
    if (sequence_number) |seq| {
        return try std.fmt.allocPrint(
            allocator,
            "{{\"op\":{d},\"d\":{d}}}",
            .{ @intFromEnum(GatewayOpcode.HEARTBEAT), seq },
        );
    }
    return try std.fmt.allocPrint(
        allocator,
        "{{\"op\":{d},\"d\":null}}",
        .{@intFromEnum(GatewayOpcode.HEARTBEAT)},
    );
}

/// Build an offline RESUME payload for a stored session.
pub fn buildResumePayload(
    allocator: std.mem.Allocator,
    token: []const u8,
    session_id: []const u8,
    sequence_number: u64,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.print(allocator, "{{\"op\":{d},\"d\":{{\"token\":", .{@intFromEnum(GatewayOpcode.RESUME)});
    try appendJsonString(allocator, &json, token);
    try json.appendSlice(allocator, ",\"session_id\":");
    try appendJsonString(allocator, &json, session_id);
    try json.print(allocator, ",\"seq\":{d}}}}}", .{sequence_number});

    return try json.toOwnedSlice(allocator);
}

/// Build an offline PRESENCE_UPDATE payload. Activity arrays are deferred until
/// live Gateway transport and richer presence state are implemented.
pub fn buildPresencePayload(
    allocator: std.mem.Allocator,
    since: ?u64,
    status: []const u8,
    afk: bool,
) ![]u8 {
    const since_json = if (since) |value|
        try std.fmt.allocPrint(allocator, "{d}", .{value})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(since_json);

    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.print(
        allocator,
        "{{\"op\":{d},\"d\":{{\"since\":{s},\"activities\":[],\"status\":",
        .{ @intFromEnum(GatewayOpcode.PRESENCE_UPDATE), since_json },
    );
    try appendJsonString(allocator, &json, status);
    try json.print(allocator, ",\"afk\":{s}}}}}", .{if (afk) "true" else "false"});

    return try json.toOwnedSlice(allocator);
}

pub const GatewayClient = struct {
    allocator: std.mem.Allocator,
    bot_token: []const u8,
    intents: u32,
    handler: GatewayEventHandler,
    sequence_number: ?u64 = null,
    session_id: ?[]const u8 = null,
    resume_gateway_url: ?[]const u8 = null,
    heartbeat_interval_ms: u64 = 0,
    running: std.atomic.Value(bool),
    state: GatewayState,

    /// Initialize a new gateway client. Does not connect.
    /// `bot_token` and `handler` are borrowed (caller retains ownership).
    pub fn init(
        allocator: std.mem.Allocator,
        bot_token: []const u8,
        intents: u32,
        handler: GatewayEventHandler,
    ) GatewayClient {
        return .{
            .allocator = allocator,
            .bot_token = bot_token,
            .intents = intents,
            .handler = handler,
            .running = std.atomic.Value(bool).init(false),
            .state = .disconnected,
        };
    }

    /// Begin offline gateway setup. Transitions through connecting -> identifying -> connected
    /// without opening a socket; real TCP/TLS/WebSocket transport is deferred.
    pub fn connect(self: *GatewayClient) !void {
        if (self.state == .connected) return GatewayError.AlreadyConnected;
        self.state = .connecting;
        self.running.store(true, .release);
        self.state = .identifying;
        self.state = .connected;
    }

    /// Disconnect from the gateway. Resets running flag and state.
    pub fn disconnect(self: *GatewayClient) void {
        self.running.store(false, .release);
        self.state = .disconnected;
    }

    /// Process a raw JSON gateway payload. Parses the opcode and dispatches
    /// to the appropriate handler or internal state update.
    pub fn processPayload(self: *GatewayClient, payload: []const u8) !void {
        // Parse the top-level gateway envelope using the existing GatewayPayload type
        // which has: op: u8, d: ?std.json.Value, s: ?u64, t: ?[]const u8
        const types = @import("types.zig");
        const parsed = std.json.parseFromSlice(
            types.GatewayPayload,
            self.allocator,
            payload,
            .{},
        ) catch return GatewayError.InvalidPayload;
        defer parsed.deinit();

        const envelope = parsed.value;
        const op: u8 = envelope.op;

        // Convert raw opcode integer to typed enum
        const opcode: GatewayOpcode = @enumFromInt(op);

        switch (opcode) {
            .DISPATCH => {
                // Update sequence number
                if (envelope.s) |s| {
                    self.sequence_number = s;
                }
                // Dispatch based on event name
                if (envelope.t) |event_name| {
                    // Pass the raw payload to event handlers. The handler
                    // can re-parse it to extract the `d` field if needed.
                    // This avoids an extra serialize round-trip.
                    const event_data: []const u8 = payload;

                    if (std.mem.eql(u8, event_name, "MESSAGE_CREATE")) {
                        if (self.handler.on_message_create) |cb| {
                            cb(self.handler.ctx, event_data);
                        }
                    } else if (std.mem.eql(u8, event_name, "INTERACTION_CREATE")) {
                        if (self.handler.on_interaction_create) |cb| {
                            cb(self.handler.ctx, event_data);
                        }
                    } else if (std.mem.eql(u8, event_name, "READY")) {
                        // Extract session_id and resume_gateway_url from d
                        if (envelope.d) |d_val| {
                            if (d_val == .object) {
                                if (d_val.object.get("session_id")) |sid| {
                                    if (sid == .string) {
                                        // Free old session_id if any
                                        if (self.session_id) |old| self.allocator.free(old);
                                        self.session_id = self.allocator.dupe(u8, sid.string) catch null;
                                    }
                                }
                                if (d_val.object.get("resume_gateway_url")) |url| {
                                    if (url == .string) {
                                        if (self.resume_gateway_url) |old| self.allocator.free(old);
                                        self.resume_gateway_url = self.allocator.dupe(u8, url.string) catch null;
                                    }
                                }
                            }
                        }
                        if (self.handler.on_ready) |cb| {
                            cb(self.handler.ctx, event_data);
                        }
                    } else if (std.mem.eql(u8, event_name, "GUILD_CREATE")) {
                        if (self.handler.on_guild_create) |cb| {
                            cb(self.handler.ctx, event_data);
                        }
                    } else if (std.mem.eql(u8, event_name, "RESUMED")) {
                        if (self.handler.on_resumed) |cb| {
                            cb(self.handler.ctx);
                        }
                    }
                }
            },
            .HEARTBEAT => {
                // Server is requesting an immediate heartbeat. In a real implementation
                // we would send a heartbeat payload back immediately.
            },
            .HELLO => {
                // Extract heartbeat_interval from d
                if (envelope.d) |d_val| {
                    if (d_val == .object) {
                        if (d_val.object.get("heartbeat_interval")) |interval| {
                            if (interval == .integer and interval.integer >= 0) {
                                self.heartbeat_interval_ms = @intCast(interval.integer);
                            }
                        }
                    }
                }
            },
            .HEARTBEAT_ACK => {
                // Heartbeat acknowledged. In a real implementation we would track
                // latency and detect zombie connections.
            },
            .RECONNECT => {
                self.state = .resuming;
            },
            .INVALID_SESSION => {
                self.state = .disconnected;
            },
            else => {
                // Unknown or unhandled opcode — ignored.
            },
        }
    }

    /// Returns the current gateway connection state.
    pub fn getState(self: *const GatewayClient) GatewayState {
        return self.state;
    }

    /// Returns the last received sequence number, or null if none received.
    pub fn getSequenceNumber(self: *const GatewayClient) ?u64 {
        return self.sequence_number;
    }

    /// Free any heap-allocated state (session_id, resume_gateway_url).
    pub fn deinit(self: *GatewayClient) void {
        if (self.session_id) |sid| self.allocator.free(sid);
        if (self.resume_gateway_url) |url| self.allocator.free(url);
        self.session_id = null;
        self.resume_gateway_url = null;
        self.state = .disconnected;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "gateway client init and deinit" {
    const allocator = std.testing.allocator;
    const handler = GatewayEventHandler{};
    var client = GatewayClient.init(allocator, "test-token", 513, handler);
    defer client.deinit();

    try std.testing.expectEqual(GatewayState.disconnected, client.getState());
    try std.testing.expectEqual(@as(?u64, null), client.getSequenceNumber());
    try std.testing.expectEqualStrings("test-token", client.bot_token);
    try std.testing.expectEqual(@as(u32, 513), client.intents);
}

test "gateway client connect changes state" {
    const allocator = std.testing.allocator;
    const handler = GatewayEventHandler{};
    var client = GatewayClient.init(allocator, "test-token", 513, handler);
    defer client.deinit();

    try std.testing.expectEqual(GatewayState.disconnected, client.getState());
    try client.connect();
    try std.testing.expectEqual(GatewayState.connected, client.getState());
    try std.testing.expect(client.running.load(.acquire));
}

test "gateway client disconnect" {
    const allocator = std.testing.allocator;
    const handler = GatewayEventHandler{};
    var client = GatewayClient.init(allocator, "test-token", 513, handler);
    defer client.deinit();

    try client.connect();
    client.disconnect();
    try std.testing.expectEqual(GatewayState.disconnected, client.getState());
    try std.testing.expect(!client.running.load(.acquire));
}

test "gateway client process HELLO payload" {
    const allocator = std.testing.allocator;
    const handler = GatewayEventHandler{};
    var client = GatewayClient.init(allocator, "test-token", 513, handler);
    defer client.deinit();

    const hello_payload =
        \\{"op":10,"d":{"heartbeat_interval":41250},"s":null,"t":null}
    ;
    try client.processPayload(hello_payload);
    try std.testing.expectEqual(@as(u64, 41250), client.heartbeat_interval_ms);
}

test "gateway client process DISPATCH MESSAGE_CREATE" {
    const allocator = std.testing.allocator;

    var received = false;
    const handler = GatewayEventHandler{
        .ctx = @ptrCast(&received),
        .on_message_create = struct {
            fn callback(ctx: ?*anyopaque, _: []const u8) void {
                const ptr: *bool = @ptrCast(@alignCast(ctx.?));
                ptr.* = true;
            }
        }.callback,
    };
    var client = GatewayClient.init(allocator, "test-token", 513, handler);
    defer client.deinit();

    const dispatch_payload =
        \\{"op":0,"d":{"content":"hello"},"s":42,"t":"MESSAGE_CREATE"}
    ;
    try client.processPayload(dispatch_payload);
    try std.testing.expect(received);
    try std.testing.expectEqual(@as(?u64, 42), client.getSequenceNumber());
}

test "gateway client process RECONNECT" {
    const allocator = std.testing.allocator;
    const handler = GatewayEventHandler{};
    var client = GatewayClient.init(allocator, "test-token", 513, handler);
    defer client.deinit();

    try client.connect();
    const reconnect_payload =
        \\{"op":7,"d":null,"s":null,"t":null}
    ;
    try client.processPayload(reconnect_payload);
    try std.testing.expectEqual(GatewayState.resuming, client.getState());
}

test "gateway client process INVALID_SESSION" {
    const allocator = std.testing.allocator;
    const handler = GatewayEventHandler{};
    var client = GatewayClient.init(allocator, "test-token", 513, handler);
    defer client.deinit();

    try client.connect();
    const invalid_session_payload =
        \\{"op":9,"d":false,"s":null,"t":null}
    ;
    try client.processPayload(invalid_session_payload);
    try std.testing.expectEqual(GatewayState.disconnected, client.getState());
}

test "gateway client process READY extracts session_id" {
    const allocator = std.testing.allocator;
    var ready_called = false;
    const handler = GatewayEventHandler{
        .ctx = @ptrCast(&ready_called),
        .on_ready = struct {
            fn callback(ctx: ?*anyopaque, _: []const u8) void {
                const ptr: *bool = @ptrCast(@alignCast(ctx.?));
                ptr.* = true;
            }
        }.callback,
    };
    var client = GatewayClient.init(allocator, "test-token", 513, handler);
    defer client.deinit();

    const ready_payload =
        \\{"op":0,"d":{"session_id":"abc123","resume_gateway_url":"wss://resume.example.com","v":10},"s":1,"t":"READY"}
    ;
    try client.processPayload(ready_payload);
    try std.testing.expectEqualStrings("abc123", client.session_id.?);
    try std.testing.expectEqualStrings("wss://resume.example.com", client.resume_gateway_url.?);
    try std.testing.expectEqual(@as(?u64, 1), client.getSequenceNumber());
    try std.testing.expect(ready_called);
}

test "gateway payload builders produce offline control frames" {
    const allocator = std.testing.allocator;

    const identify = try buildIdentifyPayload(allocator, "token\"x", 513);
    defer allocator.free(identify);
    try std.testing.expect(std.mem.indexOf(u8, identify, "\"op\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, identify, "token\\\"x") != null);
    try std.testing.expect(std.mem.indexOf(u8, identify, "\"intents\":513") != null);

    const heartbeat = try buildHeartbeatPayload(allocator, 42);
    defer allocator.free(heartbeat);
    try std.testing.expectEqualStrings("{\"op\":1,\"d\":42}", heartbeat);

    const heartbeat_null = try buildHeartbeatPayload(allocator, null);
    defer allocator.free(heartbeat_null);
    try std.testing.expectEqualStrings("{\"op\":1,\"d\":null}", heartbeat_null);

    const resume_payload = try buildResumePayload(allocator, "token", "session", 99);
    defer allocator.free(resume_payload);
    try std.testing.expect(std.mem.indexOf(u8, resume_payload, "\"op\":6") != null);
    try std.testing.expect(std.mem.indexOf(u8, resume_payload, "\"session_id\":\"session\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, resume_payload, "\"seq\":99") != null);

    const presence = try buildPresencePayload(allocator, null, "idle", true);
    defer allocator.free(presence);
    try std.testing.expect(std.mem.indexOf(u8, presence, "\"op\":3") != null);
    try std.testing.expect(std.mem.indexOf(u8, presence, "\"since\":null") != null);
    try std.testing.expect(std.mem.indexOf(u8, presence, "\"status\":\"idle\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, presence, "\"afk\":true") != null);
}

test "gateway client dispatches interaction guild and resumed events" {
    const allocator = std.testing.allocator;

    const Counters = struct {
        interaction: u8 = 0,
        guild: u8 = 0,
        resumed: u8 = 0,
    };

    var counters = Counters{};
    const handler = GatewayEventHandler{
        .ctx = @ptrCast(&counters),
        .on_interaction_create = struct {
            fn callback(ctx: ?*anyopaque, _: []const u8) void {
                const ptr: *Counters = @ptrCast(@alignCast(ctx.?));
                ptr.interaction += 1;
            }
        }.callback,
        .on_guild_create = struct {
            fn callback(ctx: ?*anyopaque, _: []const u8) void {
                const ptr: *Counters = @ptrCast(@alignCast(ctx.?));
                ptr.guild += 1;
            }
        }.callback,
        .on_resumed = struct {
            fn callback(ctx: ?*anyopaque) void {
                const ptr: *Counters = @ptrCast(@alignCast(ctx.?));
                ptr.resumed += 1;
            }
        }.callback,
    };
    var client = GatewayClient.init(allocator, "test-token", 513, handler);
    defer client.deinit();

    try client.processPayload(
        \\{"op":0,"d":{"id":"interaction"},"s":2,"t":"INTERACTION_CREATE"}
    );
    try client.processPayload(
        \\{"op":0,"d":{"id":"guild"},"s":3,"t":"GUILD_CREATE"}
    );
    try client.processPayload(
        \\{"op":0,"d":{},"s":4,"t":"RESUMED"}
    );

    try std.testing.expectEqual(@as(u8, 1), counters.interaction);
    try std.testing.expectEqual(@as(u8, 1), counters.guild);
    try std.testing.expectEqual(@as(u8, 1), counters.resumed);
    try std.testing.expectEqual(@as(?u64, 4), client.getSequenceNumber());
}

test "gateway client process invalid JSON" {
    const allocator = std.testing.allocator;
    const handler = GatewayEventHandler{};
    var client = GatewayClient.init(allocator, "test-token", 513, handler);
    defer client.deinit();

    const result = client.processPayload("not json at all");
    try std.testing.expectError(GatewayError.InvalidPayload, result);
}

test {
    std.testing.refAllDecls(@This());
}
