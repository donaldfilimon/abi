//! WebSocket frame encoding/decoding benchmarks.

const std = @import("std");

pub const WebSocketFrame = struct {
    fin: bool,
    opcode: u4,
    masked: bool,
    payload: []const u8,
};

pub fn encodeWebSocketFrame(allocator: std.mem.Allocator, frame: WebSocketFrame) ![]u8 {
    var buffer = std.ArrayListUnmanaged(u8).empty;
    errdefer buffer.deinit(allocator);

    const first_byte: u8 = (@as(u8, if (frame.fin) 0x80 else 0)) | frame.opcode;
    try buffer.append(allocator, first_byte);

    const mask_bit: u8 = if (frame.masked) 0x80 else 0;
    if (frame.payload.len < 126) {
        try buffer.append(allocator, mask_bit | @as(u8, @intCast(frame.payload.len)));
    } else if (frame.payload.len <= 65535) {
        try buffer.append(allocator, mask_bit | 126);
        try buffer.append(allocator, @intCast((frame.payload.len >> 8) & 0xFF));
        try buffer.append(allocator, @intCast(frame.payload.len & 0xFF));
    } else {
        try buffer.append(allocator, mask_bit | 127);
        const len = frame.payload.len;
        inline for (0..8) |i| {
            try buffer.append(allocator, @intCast((len >> @intCast((7 - i) * 8)) & 0xFF));
        }
    }

    if (frame.masked) {
        try buffer.appendSlice(allocator, &[_]u8{ 0x12, 0x34, 0x56, 0x78 });
        for (frame.payload, 0..) |b, i| {
            const mask_key = [_]u8{ 0x12, 0x34, 0x56, 0x78 };
            try buffer.append(allocator, b ^ mask_key[i % 4]);
        }
    } else {
        try buffer.appendSlice(allocator, frame.payload);
    }

    return buffer.toOwnedSlice(allocator);
}

pub fn decodeWebSocketFrame(data: []const u8) ?WebSocketFrame {
    if (data.len < 2) return null;

    const fin = (data[0] & 0x80) != 0;
    const opcode: u4 = @intCast(data[0] & 0x0F);
    const masked = (data[1] & 0x80) != 0;

    var payload_len: usize = data[1] & 0x7F;
    var offset: usize = 2;

    if (payload_len == 126) {
        if (data.len < 4) return null;
        payload_len = (@as(usize, data[2]) << 8) | data[3];
        offset = 4;
    } else if (payload_len == 127) {
        if (data.len < 10) return null;
        payload_len = 0;
        inline for (0..8) |i| {
            payload_len = (payload_len << 8) | data[2 + i];
        }
        offset = 10;
    }

    if (masked) offset += 4;
    if (data.len < offset + payload_len) return null;

    return .{ .fin = fin, .opcode = opcode, .masked = masked, .payload = data[offset .. offset + payload_len] };
}

pub fn benchWebSocketEncode(allocator: std.mem.Allocator, payload: []const u8) !void {
    const frame = WebSocketFrame{ .fin = true, .opcode = 1, .masked = true, .payload = payload };
    const encoded = try encodeWebSocketFrame(allocator, frame);
    defer allocator.free(encoded);
    std.mem.doNotOptimizeAway(encoded.ptr);
}

pub fn benchWebSocketDecode(data: []const u8) void {
    const frame = decodeWebSocketFrame(data);
    std.mem.doNotOptimizeAway(&frame);
}

test "websocket frame roundtrip" {
    const allocator = std.testing.allocator;
    const frame = WebSocketFrame{ .fin = true, .opcode = 1, .masked = false, .payload = "Hello" };
    const encoded = try encodeWebSocketFrame(allocator, frame);
    defer allocator.free(encoded);
    const decoded = decodeWebSocketFrame(encoded);
    try std.testing.expect(decoded != null);
    try std.testing.expectEqualStrings("Hello", decoded.?.payload);
}
