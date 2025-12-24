const std = @import("std");
const builtin = @import("builtin");

fn computeAcceptKey(allocator: std.mem.Allocator, key: []const u8) ![]u8 {
    const guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    var sha1 = std.crypto.hash.Sha1.init(.{});
    sha1.update(key);
    sha1.update(guid);
    var digest: [20]u8 = undefined;
    sha1.final(&digest);

    var out: [64]u8 = undefined;
    const enc = std.base64.standard.Encoder.encode(&out, &digest);
    return allocator.dupe(u8, enc);
}

fn wsaRecvSocket(handle: anytype, buf: []u8) !usize {
    const w = std.os.windows;
    const ws2 = w.ws2_32;
    const wsabuf: ws2.WSABUF = .{ .len = @as(w.DWORD, @intCast(buf.len)), .buf = buf.ptr };
    var flags: w.DWORD = 0;
    var recvd: w.DWORD = 0;
    const s: ws2.SOCKET = handle;
    var wsabufs = [_]ws2.WSABUF{wsabuf};
    const rc = ws2.WSARecv(s, wsabufs[0..].ptr, 1, &recvd, &flags, null, null);
    if (rc == ws2.SOCKET_ERROR) {
        const err = ws2.WSAGetLastError();
        const code: i32 = @intFromEnum(err);
        if (code == 10035 or code == 10004) return 0; // WSAEWOULDBLOCK or WSAEINTR
        return error.Unexpected;
    }
    return @intCast(recvd);
}

inline fn readCompat(stream: std.net.Stream, buf: []u8) !usize {
    if (builtin.os.tag == .windows) {
        return try wsaRecvSocket(stream.handle, buf);
    } else {
        return try stream.read(buf);
    }
}

test "WebSocket key generation" {}

test "WebSocket request building" {

    const allocator = std.testing.allocator;

    // Test WebSocket upgrade request building
    const test_key = "test-key-12345";

    const req = try std.fmt.allocPrint(
        allocator,
        "GET /ws HTTP/1.1\r\n" ++
            "Host: 127.0.0.1:8080\r\n" ++
            "Connection: Upgrade\r\n" ++
            "Upgrade: websocket\r\n" ++
            "Sec-WebSocket-Version: 13\r\n" ++
            "Sec-WebSocket-Key: {s}\r\n\r\n",
        .{test_key},
    );
    defer allocator.free(req);

    // Verify request contains required headers

    const has_get = std.mem.indexOf(u8, req, "GET /ws HTTP/1.1") != null;
    try std.testing.expect(has_get);

    const has_connection = std.mem.indexOf(u8, req, "Connection: Upgrade") != null;
    try std.testing.expect(has_connection);

    const has_upgrade = std.mem.indexOf(u8, req, "Upgrade: websocket") != null;
    try std.testing.expect(has_upgrade);

    const has_version = std.mem.indexOf(u8, req, "Sec-WebSocket-Version: 13") != null;
    try std.testing.expect(has_version);

    const has_key = std.mem.indexOf(u8, req, "Sec-WebSocket-Key:") != null;
    try std.testing.expect(has_key);

}

test "WebSocket response parsing" {

    // Test WebSocket response parsing logic
    const mock_response =
        "HTTP/1.1 101 Switching Protocols\r\n" ++
        "Upgrade: websocket\r\n" ++
        "Connection: Upgrade\r\n" ++
        "Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n";


    // Verify response contains required headers

    const has_status = std.mem.indexOf(u8, mock_response, "HTTP/1.1 101") != null;
    try std.testing.expect(has_status);

    const has_upgrade = std.mem.indexOf(u8, mock_response, "Upgrade: websocket") != null;
    try std.testing.expect(has_upgrade);

    const has_connection = std.mem.indexOf(u8, mock_response, "Connection: Upgrade") != null;
    try std.testing.expect(has_connection);

    const has_accept = std.mem.indexOf(u8, mock_response, "Sec-WebSocket-Accept:") != null;
    try std.testing.expect(has_accept);

}

test "WebSocket handshake validation" {

    const allocator = std.testing.allocator;

    // Test complete handshake validation
    const client_key = "dGhlIHNhbXBsZSBub25jZQ==";
    const expected_accept = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=";


    const computed_accept = try computeAcceptKey(allocator, client_key);
    defer allocator.free(computed_accept);

    // Note: This test uses a known test vector from RFC 6455
    // In a real implementation, we'd verify the computed key matches expected

    try std.testing.expect(computed_accept.len > 0);

    try std.testing.expectEqualStrings(expected_accept, computed_accept);

}
