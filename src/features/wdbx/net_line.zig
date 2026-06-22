//! Newline-framed socket read helper (Transport Layer).
//!
//! Shared by the cluster_rpc and remote_compute transports. A single
//! `Stream.read` returns only whatever bytes the kernel had ready, so a request
//! or reply split across TCP segments would be truncated by a naive one-shot
//! read. `readLine` accumulates into the caller's buffer until a newline is seen
//! (or the buffer fills / the peer closes), then returns the frame without its
//! trailing CR/LF. Correct on loopback (one read) and on a routable interface
//! (multiple segments).

const std = @import("std");

const Stream = std.Io.net.Stream;

/// Write `bytes` to `conn` and flush. The Writer flushes automatically when its
/// stack buffer fills, so a modest buffer handles messages larger than itself.
pub fn writeLine(io: std.Io, conn: Stream, bytes: []const u8) !void {
    var wb: [4096]u8 = undefined;
    var sw = conn.writer(io, &wb);
    try sw.interface.writeAll(bytes);
    try sw.interface.flush();
}

/// Connect to a 127.0.0.1 peer on `port` and send `msg`, returning the open
/// stream (read the reply with `readLine`). Null if the peer is unreachable
/// (connection refused / bad address) — callers treat that as a down peer.
pub fn dial(io: std.Io, port: u16, msg: []const u8) !?Stream {
    var address = std.Io.net.IpAddress.parseIp4("127.0.0.1", port) catch return null;
    const conn = address.connect(io, .{ .mode = .stream }) catch return null;
    errdefer conn.close(io);
    try writeLine(io, conn, msg);
    return conn;
}

/// Read one newline-terminated frame into `buf`, reassembling across reads.
/// Returns the bytes up to (not including) the first '\n', trimmed of a trailing
/// '\r'. If the peer closes before a newline, returns whatever was received.
pub fn readLine(io: std.Io, conn: Stream, buf: []u8) ![]const u8 {
    var total: usize = 0;
    while (total < buf.len) {
        var rv: [1][]u8 = .{buf[total..]};
        const n = try conn.read(io, &rv);
        if (n == 0) break; // EOF / peer closed
        total += n;
        if (std.mem.indexOfScalar(u8, buf[0..total], '\n') != null) break;
    }
    const end = std.mem.indexOfScalar(u8, buf[0..total], '\n') orelse {
        // No newline and the buffer is full: the frame exceeds `buf` and was
        // truncated. Signal it rather than handing back a silently-cut line.
        if (total == buf.len) return error.LineTooLong;
        // Otherwise the peer closed before sending a newline (incomplete frame);
        // return what arrived so the caller's parser rejects it.
        return std.mem.trimEnd(u8, buf[0..total], "\r");
    };
    return std.mem.trimEnd(u8, buf[0..end], "\r");
}

// --- Tests ---
//
// `readLine` is the load-bearing framing primitive for both RPC transports
// (cluster_rpc, remote_compute) yet was only exercised transitively by their
// loopback tests. These drive it directly over a real 127.0.0.1 socket pair
// (the same `std.testing.io` + loopback pattern those modules use) to pin the
// three frame-boundary outcomes: clean frame, incomplete frame on early close,
// and over-length frame.

test "net_line: readLine returns a single frame and strips trailing CRLF" {
    const io = std.testing.io;
    var server = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 39401);
    var listener = try server.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);

    const client = (try dial(io, 39401, "hello world\r\n")).?;
    defer client.close(io);
    const conn = try listener.accept(io);
    defer conn.close(io);

    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("hello world", try readLine(io, conn, &buf));
}

test "net_line: readLine returns the partial frame when the peer closes without a newline" {
    const io = std.testing.io;
    var server = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 39402);
    var listener = try server.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);

    // Write an unterminated frame, then close so the server reads EOF mid-frame.
    const client = (try dial(io, 39402, "partial")).?;
    client.close(io);
    const conn = try listener.accept(io);
    defer conn.close(io);

    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("partial", try readLine(io, conn, &buf));
}

test "net_line: readLine reports LineTooLong when a frame fills the buffer with no newline" {
    const io = std.testing.io;
    var server = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 39403);
    var listener = try server.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);

    const client = (try dial(io, 39403, "aaaaaaaaaaaaaaaa")).?; // 16 bytes, no '\n'
    defer client.close(io);
    const conn = try listener.accept(io);
    defer conn.close(io);

    var buf: [8]u8 = undefined; // smaller than the payload, so it fills before any newline
    try std.testing.expectError(error.LineTooLong, readLine(io, conn, &buf));
}

test {
    std.testing.refAllDecls(@This());
}
