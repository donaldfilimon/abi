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

test "WebSocket upgrade handshake to /ws" {
    const allocator = std.testing.allocator;

    // Connect raw TCP to the running server (assumes server on default port)
    const addr = try std.net.Address.parseIp("127.0.0.1", 8080);
    var stream = try std.net.tcpConnectToAddress(addr);
    defer stream.close();

    // Prepare Sec-WebSocket-Key
    var key_src: [16]u8 = undefined;
    std.crypto.random.bytes(&key_src);
    var key_buf: [32]u8 = undefined;
    const key_b64 = std.base64.standard.Encoder.encode(&key_buf, &key_src);

    // Build upgrade request
    const req = try std.fmt.allocPrint(
        allocator,
        "GET /ws HTTP/1.1\r\n" ++
            "Host: 127.0.0.1:8080\r\n" ++
            "Connection: Upgrade\r\n" ++
            "Upgrade: websocket\r\n" ++
            "Sec-WebSocket-Version: 13\r\n" ++
            "Sec-WebSocket-Key: {s}\r\n\r\n",
        .{key_b64},
    );
    defer allocator.free(req);

    try stream.writeAll(req);

    // Read response
    var buf: [2048]u8 = undefined;
    const n = try readCompat(stream, &buf);
    try std.testing.expect(n > 0);

    const resp = buf[0..n];
    try std.testing.expect(std.mem.indexOf(u8, resp, "HTTP/1.1 101") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "Upgrade: websocket") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "Connection: Upgrade") != null);

    // Verify Sec-WebSocket-Accept header exists
    try std.testing.expect(std.mem.indexOf(u8, resp, "Sec-WebSocket-Accept:") != null);
}
