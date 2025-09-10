const std = @import("std");
const web = @import("web_server");

test "web server socket responds to /health" {
    const alloc = std.testing.allocator;
    var srv = try web.WebServer.init(alloc, .{ .port = 30807, .host = "127.0.0.1" });
    defer srv.deinit();

    var th = try std.Thread.spawn(.{}, web.WebServer.startOnce, .{srv});
    defer th.join();

    // Give the server a moment to start listening
    std.Thread.sleep(10 * std.time.ns_per_ms);

    const addr = try std.net.Address.parseIp("127.0.0.1", 30807);
    var conn = try std.net.tcpConnectToAddress(addr);
    defer conn.close();

    const req = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";

    const builtin = @import("builtin");
    if (builtin.os.tag == .windows) {
        const windows = std.os.windows;

        // Send request using socket-specific send
        const sent: c_int = windows.ws2_32.send(conn.handle, @ptrCast(req.ptr), @intCast(req.len), 0);
        try std.testing.expect(sent != windows.ws2_32.SOCKET_ERROR);

        // Receive response using socket-specific recv
        var buf: [1024]u8 = undefined;
        const max_len: c_int = @intCast(@min(buf.len, @as(usize, @intCast(std.math.maxInt(c_int)))));
        const n_recv: c_int = windows.ws2_32.recv(conn.handle, @ptrCast(&buf[0]), max_len, 0);
        try std.testing.expect(n_recv != windows.ws2_32.SOCKET_ERROR);

        const body = buf[0..@intCast(n_recv)];
        try std.testing.expect(std.mem.indexOf(u8, body, "200") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "healthy") != null);
    } else {
        _ = try conn.writeAll(req);

        var buf: [1024]u8 = undefined;
        const n = try conn.read(&buf);
        const body = buf[0..n];
        try std.testing.expect(std.mem.indexOf(u8, body, "200") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "healthy") != null);
    }
}
