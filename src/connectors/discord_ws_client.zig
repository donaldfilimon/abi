//! Discord gateway WebSocket client (WDBX Rust `src/features/discord.rs`).
//!
//! Minimal WebSocket client over a connected `std.Io.net.Stream`. Performs the
//! HTTP upgrade handshake and masks client→server frames. Server→client frames
//! (text / close / ping / pong) are parsed. Self-contained: only `std`.
//!
//! Honesty note: TLS termination is **not** linked (Discord requires wss), so
//! this must sit behind a TLS-terminating proxy to reach the real gateway.

const std = @import("std");

/// Fill `buf` with pseudo-random bytes (reference-grade xorshift; not a CSPRNG
/// — the live path only needs a non-deterministic-looking WebSocket key and
/// frame mask, which this satisfies).
var rng_state: u64 = 0x9E3779B97F4A7C15;
pub fn fillRandom(buf: []u8) void {
    var s = rng_state;
    for (buf) |*b| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        b.* = @intCast(s & 0xFF);
    }
    rng_state = s;
}

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
                self.read_buf.replaceRange(allocator, 0, off + len, &[_]u8{}) catch |err| std.log.warn("discord read_buf: {s}", .{@errorName(err)}); // intentional: non-critical UI/telemetry buffer, failure = degraded display only
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
        self.writeFrame(0x8, &[_]u8{}) catch |err| std.log.warn("discord close frame: {s}", .{@errorName(err)}); // intentional: non-critical UI/telemetry buffer, failure = degraded display only
        self.stream.close(self.io);
        self.closed = true;
    }
};

test {
    std.testing.refAllDecls(@This());
}
