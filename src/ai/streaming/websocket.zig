//! WebSocket Protocol Handler
//!
//! Implements WebSocket protocol (RFC 6455) for bidirectional streaming:
//! - Handshake with Sec-WebSocket-Accept computation
//! - Frame encoding/decoding (text, binary, ping/pong, close)
//! - Message fragmentation handling
//! - Masking for client frames

const std = @import("std");

/// WebSocket opcode types
pub const Opcode = enum(u4) {
    continuation = 0x0,
    text = 0x1,
    binary = 0x2,
    close = 0x8,
    ping = 0x9,
    pong = 0xA,
    _,
};

/// WebSocket close codes
pub const CloseCode = enum(u16) {
    normal = 1000,
    going_away = 1001,
    protocol_error = 1002,
    unsupported_data = 1003,
    no_status = 1005,
    abnormal = 1006,
    invalid_payload = 1007,
    policy_violation = 1008,
    message_too_big = 1009,
    mandatory_extension = 1010,
    internal_error = 1011,
    _,
};

/// WebSocket frame
pub const Frame = struct {
    fin: bool,
    opcode: Opcode,
    masked: bool,
    mask: ?[4]u8,
    payload: []const u8,

    pub fn isControl(self: Frame) bool {
        return @intFromEnum(self.opcode) >= 0x8;
    }
};

/// WebSocket connection state
pub const ConnectionState = enum {
    connecting,
    open,
    closing,
    closed,
};

/// WebSocket handler configuration
pub const WebSocketConfig = struct {
    /// Maximum message size (default 16MB)
    max_message_size: usize = 16 * 1024 * 1024,
    /// Ping interval in milliseconds (0 = disabled)
    ping_interval_ms: u64 = 30000,
    /// Enable compression (permessage-deflate)
    enable_compression: bool = false,
    /// Message callback
    on_message: ?*const fn (*WebSocketHandler, []const u8) void = null,
    /// Close callback
    on_close: ?*const fn (*WebSocketHandler, CloseCode, []const u8) void = null,
    /// Error callback
    on_error: ?*const fn (*WebSocketHandler, anyerror) void = null,
};

/// WebSocket handler for streaming inference
pub const WebSocketHandler = struct {
    allocator: std.mem.Allocator,
    config: WebSocketConfig,
    state: ConnectionState,
    fragment_buffer: std.ArrayListUnmanaged(u8),
    fragment_opcode: ?Opcode,

    const Self = @This();

    /// Initialize WebSocket handler
    pub fn init(allocator: std.mem.Allocator, config: WebSocketConfig) !Self {
        return .{
            .allocator = allocator,
            .config = config,
            .state = .connecting,
            .fragment_buffer = .empty,
            .fragment_opcode = null,
        };
    }

    /// Deinitialize
    pub fn deinit(self: *Self) void {
        self.fragment_buffer.deinit(self.allocator);
        self.* = undefined;
    }

    /// Process received data
    pub fn processData(self: *Self, data: []const u8) !void {
        var offset: usize = 0;

        while (offset < data.len) {
            const frame = try self.parseFrame(data[offset..]);
            offset += frame.header_size + frame.payload_len;

            try self.handleFrame(frame.frame);
        }
    }

    /// Parse a WebSocket frame from data
    fn parseFrame(self: *Self, data: []const u8) !struct { frame: Frame, header_size: usize, payload_len: usize } {
        if (data.len < 2) return error.IncompleteFrame;

        const byte0 = data[0];
        const byte1 = data[1];

        const fin = (byte0 & 0x80) != 0;
        const opcode: Opcode = @enumFromInt(@as(u4, @truncate(byte0 & 0x0F)));
        const masked = (byte1 & 0x80) != 0;
        var payload_len: u64 = byte1 & 0x7F;
        var header_size: usize = 2;

        // Extended payload length
        if (payload_len == 126) {
            if (data.len < 4) return error.IncompleteFrame;
            payload_len = std.mem.readInt(u16, data[2..4], .big);
            header_size = 4;
        } else if (payload_len == 127) {
            if (data.len < 10) return error.IncompleteFrame;
            payload_len = std.mem.readInt(u64, data[2..10], .big);
            header_size = 10;
        }

        // Check message size limit
        if (payload_len > self.config.max_message_size) {
            return error.MessageTooLarge;
        }

        // Masking key
        var mask: ?[4]u8 = null;
        if (masked) {
            if (data.len < header_size + 4) return error.IncompleteFrame;
            mask = data[header_size..][0..4].*;
            header_size += 4;
        }

        // Payload
        const payload_usize: usize = @intCast(payload_len);
        if (data.len < header_size + payload_usize) return error.IncompleteFrame;

        const payload = try self.allocator.alloc(u8, payload_usize);
        errdefer self.allocator.free(payload);
        @memcpy(payload, data[header_size..][0..payload_usize]);

        // Unmask if needed
        if (mask) |m| {
            for (payload, 0..) |*b, i| {
                b.* ^= m[i % 4];
            }
        }

        return .{
            .frame = .{
                .fin = fin,
                .opcode = opcode,
                .masked = masked,
                .mask = mask,
                .payload = payload,
            },
            .header_size = header_size,
            .payload_len = payload_usize,
        };
    }

    /// Handle a parsed frame
    fn handleFrame(self: *Self, frame: Frame) !void {
        switch (frame.opcode) {
            .continuation => {
                if (self.fragment_opcode == null) return error.UnexpectedContinuation;
                try self.fragment_buffer.appendSlice(self.allocator, frame.payload);

                if (frame.fin) {
                    // Complete message
                    if (self.config.on_message) |callback| {
                        callback(self, self.fragment_buffer.items);
                    }
                    self.fragment_buffer.clearRetainingCapacity();
                    self.fragment_opcode = null;
                }
            },
            .text, .binary => {
                if (!frame.fin) {
                    // Start of fragmented message
                    self.fragment_opcode = frame.opcode;
                    try self.fragment_buffer.appendSlice(self.allocator, frame.payload);
                } else {
                    // Complete message
                    if (self.config.on_message) |callback| {
                        callback(self, frame.payload);
                    }
                }
            },
            .close => {
                self.state = .closing;
                var code = CloseCode.no_status;
                var reason: []const u8 = "";

                if (frame.payload.len >= 2) {
                    code = @enumFromInt(std.mem.readInt(u16, frame.payload[0..2], .big));
                    if (frame.payload.len > 2) {
                        reason = frame.payload[2..];
                    }
                }

                if (self.config.on_close) |callback| {
                    callback(self, code, reason);
                }
            },
            .ping => {
                // Auto-respond with pong
                _ = try self.encodeFrame(.pong, frame.payload, true);
            },
            .pong => {
                // Heartbeat response received
            },
            _ => {
                // Unknown opcode - close connection
                self.state = .closing;
            },
        }
    }

    /// Encode a WebSocket frame
    pub fn encodeFrame(
        self: *Self,
        opcode: Opcode,
        payload: []const u8,
        fin: bool,
    ) ![]u8 {
        var frame_data = std.ArrayListUnmanaged(u8).empty;
        errdefer frame_data.deinit(self.allocator);

        // First byte: FIN + opcode
        const byte0: u8 = (if (fin) @as(u8, 0x80) else @as(u8, 0)) | @intFromEnum(opcode);
        try frame_data.append(self.allocator, byte0);

        // Second byte: mask bit + payload length
        // Server frames are not masked
        if (payload.len < 126) {
            try frame_data.append(self.allocator, @intCast(payload.len));
        } else if (payload.len <= 65535) {
            try frame_data.append(self.allocator, 126);
            try frame_data.appendSlice(self.allocator, &std.mem.toBytes(std.mem.nativeToBig(u16, @intCast(payload.len))));
        } else {
            try frame_data.append(self.allocator, 127);
            try frame_data.appendSlice(self.allocator, &std.mem.toBytes(std.mem.nativeToBig(u64, payload.len)));
        }

        // Payload (unmasked for server)
        try frame_data.appendSlice(self.allocator, payload);

        return frame_data.toOwnedSlice(self.allocator);
    }

    /// Send a text message
    pub fn sendText(self: *Self, text: []const u8) ![]u8 {
        return self.encodeFrame(.text, text, true);
    }

    /// Send a binary message
    pub fn sendBinary(self: *Self, data: []const u8) ![]u8 {
        return self.encodeFrame(.binary, data, true);
    }

    /// Send a close frame
    pub fn sendClose(self: *Self, code: CloseCode, reason: []const u8) ![]u8 {
        var payload = std.ArrayListUnmanaged(u8).empty;
        defer payload.deinit(self.allocator);

        try payload.appendSlice(self.allocator, &std.mem.toBytes(std.mem.nativeToBig(u16, @intFromEnum(code))));
        if (reason.len > 0) {
            try payload.appendSlice(self.allocator, reason);
        }

        self.state = .closing;
        return self.encodeFrame(.close, payload.items, true);
    }

    /// Send a ping frame
    pub fn sendPing(self: *Self, data: []const u8) ![]u8 {
        return self.encodeFrame(.ping, data, true);
    }

    /// Get connection state
    pub fn getState(self: *const Self) ConnectionState {
        return self.state;
    }

    /// Check if connection is open
    pub fn isOpen(self: *const Self) bool {
        return self.state == .open;
    }
};

/// Compute Sec-WebSocket-Accept key from client key
pub fn computeAcceptKey(allocator: std.mem.Allocator, client_key: []const u8) ![]u8 {
    const magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

    // Concatenate key + magic
    var concat = std.ArrayListUnmanaged(u8).empty;
    defer concat.deinit(allocator);
    try concat.appendSlice(allocator, client_key);
    try concat.appendSlice(allocator, magic);

    // SHA-1 hash
    var hash: [20]u8 = undefined;
    std.crypto.hash.Sha1.hash(concat.items, &hash, .{});

    // Base64 encode (SHA1 is 20 bytes, base64 encoded is ceil(20/3)*4 = 28 bytes)
    const encoder = std.base64.standard;
    const encoded_len = comptime std.base64.standard.Encoder.calcSize(20);
    const result = try allocator.alloc(u8, encoded_len);
    _ = encoder.Encoder.encode(result, &hash);

    return result;
}

/// Create a streaming message for WebSocket
pub fn createStreamingMessage(
    allocator: std.mem.Allocator,
    event_type: []const u8,
    data: []const u8,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{\"type\":\"");
    try json.appendSlice(allocator, event_type);
    try json.appendSlice(allocator, "\",\"data\":");

    // Escape data as JSON string
    try json.append(allocator, '"');
    for (data) |c| {
        switch (c) {
            '"' => try json.appendSlice(allocator, "\\\""),
            '\\' => try json.appendSlice(allocator, "\\\\"),
            '\n' => try json.appendSlice(allocator, "\\n"),
            '\r' => try json.appendSlice(allocator, "\\r"),
            '\t' => try json.appendSlice(allocator, "\\t"),
            else => try json.append(allocator, c),
        }
    }
    try json.append(allocator, '"');
    try json.append(allocator, '}');

    return json.toOwnedSlice(allocator);
}

// Tests
test "websocket accept key computation" {
    const allocator = std.testing.allocator;

    // Test vector from RFC 6455
    const client_key = "dGhlIHNhbXBsZSBub25jZQ==";
    const expected = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=";

    const result = try computeAcceptKey(allocator, client_key);
    defer allocator.free(result);

    try std.testing.expectEqualStrings(expected, result);
}

test "websocket frame encoding" {
    const allocator = std.testing.allocator;

    var handler = try WebSocketHandler.init(allocator, .{});
    defer handler.deinit();

    const frame = try handler.encodeFrame(.text, "Hello", true);
    defer allocator.free(frame);

    // Check frame structure
    try std.testing.expect(frame.len >= 2);
    try std.testing.expectEqual(@as(u8, 0x81), frame[0]); // FIN + text opcode
    try std.testing.expectEqual(@as(u8, 5), frame[1]); // Payload length
}

test "websocket create streaming message" {
    const allocator = std.testing.allocator;

    const msg = try createStreamingMessage(allocator, "token", "hello");
    defer allocator.free(msg);

    try std.testing.expect(std.mem.indexOf(u8, msg, "\"type\":\"token\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, msg, "\"data\":\"hello\"") != null);
}

test "websocket close codes" {
    try std.testing.expectEqual(@as(u16, 1000), @intFromEnum(CloseCode.normal));
    try std.testing.expectEqual(@as(u16, 1001), @intFromEnum(CloseCode.going_away));
}
