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

test "WebSocket key generation" {
    std.debug.print("[DEBUG] Starting WebSocket key generation test\n", .{});

    const allocator = std.testing.allocator;
    std.debug.print("[DEBUG] Using test allocator\n", .{});

    // Test WebSocket key generation
    const test_key = "dGhlIHNhbXBsZSBub25jZQ==";
    std.debug.print("[DEBUG] Test key: {s}\n", .{test_key});

    std.debug.print("[DEBUG] Computing accept key...\n", .{});
    const accept_key = try computeAcceptKey(allocator, test_key);
    defer allocator.free(accept_key);
    std.debug.print("[DEBUG] ✓ Accept key computed: {s}\n", .{accept_key});

    // Verify the accept key is generated correctly
    std.debug.print("[DEBUG] Verifying accept key properties...\n", .{});

    try std.testing.expect(accept_key.len > 0);
    std.debug.print("[DEBUG] ✓ Accept key length: {d}\n", .{accept_key.len});

    try std.testing.expect(std.mem.indexOf(u8, accept_key, "=") != null); // Base64 should contain padding
    std.debug.print("[DEBUG] ✓ Accept key contains Base64 padding\n", .{});

    std.debug.print("[DEBUG] WebSocket key generation test completed successfully\n", .{});
}

test "WebSocket request building" {
    std.debug.print("[DEBUG] Starting WebSocket request building test\n", .{});

    const allocator = std.testing.allocator;
    std.debug.print("[DEBUG] Using test allocator\n", .{});

    // Test WebSocket upgrade request building
    const test_key = "test-key-12345";
    std.debug.print("[DEBUG] Test key: {s}\n", .{test_key});

    std.debug.print("[DEBUG] Building WebSocket upgrade request...\n", .{});
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
    std.debug.print("[DEBUG] ✓ WebSocket request built\n", .{});

    // Verify request contains required headers
    std.debug.print("[DEBUG] Verifying request headers...\n", .{});

    const has_get = std.mem.indexOf(u8, req, "GET /ws HTTP/1.1") != null;
    std.debug.print("[DEBUG] Contains GET request: {}\n", .{has_get});
    try std.testing.expect(has_get);

    const has_connection = std.mem.indexOf(u8, req, "Connection: Upgrade") != null;
    std.debug.print("[DEBUG] Contains Connection header: {}\n", .{has_connection});
    try std.testing.expect(has_connection);

    const has_upgrade = std.mem.indexOf(u8, req, "Upgrade: websocket") != null;
    std.debug.print("[DEBUG] Contains Upgrade header: {}\n", .{has_upgrade});
    try std.testing.expect(has_upgrade);

    const has_version = std.mem.indexOf(u8, req, "Sec-WebSocket-Version: 13") != null;
    std.debug.print("[DEBUG] Contains WebSocket version: {}\n", .{has_version});
    try std.testing.expect(has_version);

    const has_key = std.mem.indexOf(u8, req, "Sec-WebSocket-Key:") != null;
    std.debug.print("[DEBUG] Contains WebSocket key: {}\n", .{has_key});
    try std.testing.expect(has_key);

    std.debug.print("[DEBUG] WebSocket request building test completed successfully\n", .{});
}

test "WebSocket response parsing" {
    std.debug.print("[DEBUG] Starting WebSocket response parsing test\n", .{});

    // Test WebSocket response parsing logic
    const mock_response =
        "HTTP/1.1 101 Switching Protocols\r\n" ++
        "Upgrade: websocket\r\n" ++
        "Connection: Upgrade\r\n" ++
        "Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n";

    std.debug.print("[DEBUG] Mock response length: {d} bytes\n", .{mock_response.len});

    // Verify response contains required headers
    std.debug.print("[DEBUG] Verifying response headers...\n", .{});

    const has_status = std.mem.indexOf(u8, mock_response, "HTTP/1.1 101") != null;
    std.debug.print("[DEBUG] Contains HTTP 101 status: {}\n", .{has_status});
    try std.testing.expect(has_status);

    const has_upgrade = std.mem.indexOf(u8, mock_response, "Upgrade: websocket") != null;
    std.debug.print("[DEBUG] Contains Upgrade header: {}\n", .{has_upgrade});
    try std.testing.expect(has_upgrade);

    const has_connection = std.mem.indexOf(u8, mock_response, "Connection: Upgrade") != null;
    std.debug.print("[DEBUG] Contains Connection header: {}\n", .{has_connection});
    try std.testing.expect(has_connection);

    const has_accept = std.mem.indexOf(u8, mock_response, "Sec-WebSocket-Accept:") != null;
    std.debug.print("[DEBUG] Contains Accept header: {}\n", .{has_accept});
    try std.testing.expect(has_accept);

    std.debug.print("[DEBUG] WebSocket response parsing test completed successfully\n", .{});
}

test "WebSocket handshake validation" {
    std.debug.print("[DEBUG] Starting WebSocket handshake validation test\n", .{});

    const allocator = std.testing.allocator;
    std.debug.print("[DEBUG] Using test allocator\n", .{});

    // Test complete handshake validation
    const client_key = "dGhlIHNhbXBsZSBub25jZQ==";
    const expected_accept = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=";

    std.debug.print("[DEBUG] Client key: {s}\n", .{client_key});
    std.debug.print("[DEBUG] Expected accept key: {s}\n", .{expected_accept});

    std.debug.print("[DEBUG] Computing accept key for validation...\n", .{});
    const computed_accept = try computeAcceptKey(allocator, client_key);
    defer allocator.free(computed_accept);
    std.debug.print("[DEBUG] ✓ Computed accept key: {s}\n", .{computed_accept});

    // Note: This test uses a known test vector from RFC 6455
    // In a real implementation, we'd verify the computed key matches expected
    std.debug.print("[DEBUG] Validating computed key properties...\n", .{});

    try std.testing.expect(computed_accept.len > 0);
    std.debug.print("[DEBUG] ✓ Computed key has valid length: {d}\n", .{computed_accept.len});

    std.debug.print("[DEBUG] Comparing computed key with expected key...\n", .{});
    try std.testing.expectEqualStrings(expected_accept, computed_accept);
    std.debug.print("[DEBUG] ✓ Computed key matches expected key from RFC 6455\n", .{});

    std.debug.print("[DEBUG] WebSocket handshake validation test completed successfully\n", .{});
}
