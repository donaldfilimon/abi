//! Unit tests for the TCP Echo component.

const std = @import("std");
const builtin = @import("builtin");

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
        // Treat non-fatal cases as 0 bytes
        if (code == 10035 or code == 10004) return 0; // WSAEWOULDBLOCK or WSAEINTR
        return error.Unexpected;
    }
    return @intCast(recvd);
}

fn serverRead(conn: std.net.Server.Connection, buf: []u8) !usize {
    if (builtin.os.tag == .windows) {
        return try wsaRecvSocket(conn.stream.handle, buf);
    } else {
        return try conn.stream.read(buf);
    }
}

fn clientRead(client: std.net.Stream, buf: []u8) !usize {
    if (builtin.os.tag == .windows) {
        return try wsaRecvSocket(client.handle, buf);
    } else {
        return try client.read(buf);
    }
}

fn runEchoServer(address: std.net.Address, ready_signal: *std.atomic.Value(bool)) !void {
    var server = try address.listen(.{ .reuse_address = true });
    defer server.deinit();

    // Signal that the server is ready
    ready_signal.store(true, .seq_cst);

    // Accept a single client and echo a single message, then exit
    const conn = try server.accept();
    defer conn.stream.close();

    var buffer: [1024]u8 = undefined;
    const n = try serverRead(conn, &buffer);
    if (n > 0) try conn.stream.writeAll(buffer[0..n]);
}

test "TCP echo server/client end-to-end" {
    // Fixed high port to avoid binding complexities
    const port: u16 = 19099;
    const server_addr = try std.net.Address.parseIp("127.0.0.1", port);

    var ready_signal = std.atomic.Value(bool).init(false);

    var server_thread = try std.Thread.spawn(.{}, runEchoServer, .{ server_addr, &ready_signal });
    defer server_thread.join();

    // Wait for server to become ready
    var wait_ms: u32 = 0;
    while (!ready_signal.load(.seq_cst) and wait_ms < 3000) {
        std.Thread.sleep(50 * std.time.ns_per_ms);
        wait_ms += 50;
    }
    try std.testing.expect(ready_signal.load(.seq_cst));

    // Connect client
    var client = try std.net.tcpConnectToAddress(try std.net.Address.parseIp("127.0.0.1", port));
    defer client.close();

    const msg = "hello-echo";
    try client.writeAll(msg);

    var buf: [64]u8 = undefined;
    const m = try clientRead(client, &buf);
    try std.testing.expect(m == msg.len);
    try std.testing.expectEqualSlices(u8, msg, buf[0..m]);
}
