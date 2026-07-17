const std = @import("std");

pub const MAX_REQUEST_SIZE: usize = 64 * 1024;

pub fn readHttpResponse(io: std.Io, conn: std.Io.net.Stream, buf: []u8) ![]const u8 {
    var total: usize = 0;
    while (total < buf.len) {
        var rv: [1][]u8 = .{buf[total..]};
        const n = conn.read(io, &rv) catch break;
        if (n == 0) break;
        total += n;
    }
    return buf[0..total];
}

pub fn writeHttpAll(io: std.Io, conn: std.Io.net.Stream, bytes: []const u8) !void {
    var buffer: [4096]u8 = undefined;
    var stream_writer = conn.writer(io, &buffer);
    const writer = &stream_writer.interface;
    try writer.writeAll(bytes);
    try writer.flush();
}

pub fn writeUnauthorized(io: std.Io, conn: std.Io.net.Stream, error_msg: []const u8) !void {
    var buffer: [256]u8 = undefined;
    const resp = try std.fmt.bufPrint(
        &buffer,
        "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nWWW-Authenticate: Bearer\r\nConnection: close\r\n\r\n{{\"error\":\"{s}\"}}",
        .{ error_msg.len, error_msg },
    );
    try writeHttpAll(io, conn, resp);
}

pub fn findHttpBody(raw: []const u8) ?[]const u8 {
    const needle = "\r\n\r\n";
    const idx = std.mem.indexOf(u8, raw, needle) orelse return null;
    const body_start = idx + needle.len;
    if (body_start >= raw.len) return null;
    return raw[body_start..];
}

pub const HttpReadResult = union(enum) {
    request: []const u8,
    empty,
    too_large,
};

pub fn readHttpRequest(io: std.Io, conn: std.Io.net.Stream, buf: []u8) HttpReadResult {
    var total: usize = 0;
    var header_end: ?usize = null;
    var want_total: ?usize = null;

    while (true) {
        if (header_end == null) {
            if (std.mem.indexOf(u8, buf[0..total], "\r\n\r\n")) |idx| {
                const end = idx + 4;
                header_end = end;
                const declared = parseContentLength(buf[0..end]) orelse 0;
                want_total = requestTargetWithinBuffer(end, declared, buf.len) orelse return .too_large;
            }
        }

        if (want_total) |want| {
            if (total >= want) break;
        }

        if (total >= buf.len) {
            return .too_large;
        }

        var rv: [1][]u8 = .{buf[total..]};
        const n = conn.read(io, &rv) catch break;
        if (n == 0) break;
        total += n;
    }

    if (total == 0) return .empty;
    return .{ .request = buf[0..total] };
}

pub fn parseContentLength(header_block: []const u8) ?usize {
    var lines = std.mem.splitSequence(u8, header_block, "\r\n");
    _ = lines.next();
    while (lines.next()) |line| {
        if (line.len == 0) break;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const key = std.mem.trim(u8, line[0..colon], " \t");
        if (!std.ascii.eqlIgnoreCase(key, "Content-Length")) continue;
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        return std.fmt.parseInt(usize, value, 10) catch null;
    }
    return null;
}

pub fn requestTargetWithinBuffer(header_end: usize, declared_body_len: usize, capacity: usize) ?usize {
    if (header_end > capacity) return null;
    const remaining = capacity - header_end;
    if (declared_body_len > remaining) return null;
    return header_end + declared_body_len;
}

pub fn headerValue(raw: []const u8, name: []const u8) ?[]const u8 {
    const header_block = if (std.mem.indexOf(u8, raw, "\r\n\r\n")) |idx| raw[0..idx] else raw;
    var lines = std.mem.splitSequence(u8, header_block, "\r\n");
    _ = lines.next();
    while (lines.next()) |line| {
        if (line.len == 0) break;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const key = std.mem.trim(u8, line[0..colon], " \t");
        if (!std.ascii.eqlIgnoreCase(key, name)) continue;
        return std.mem.trim(u8, line[colon + 1 ..], " \t");
    }
    return null;
}

pub fn reasonPhrase(status: u16) []const u8 {
    return switch (status) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        else => "OK",
    };
}

extern fn getsockname(sockfd: std.posix.fd_t, addr: *std.posix.sockaddr, addrlen: *std.posix.socklen_t) c_int;

pub fn bindLoopback(io: std.Io) !struct { server: std.Io.net.Server, port: u16 } {
    const address = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 0);
    var srv = try address.listen(io, .{ .mode = .stream, .reuse_address = true });
    errdefer srv.deinit(io);

    var addr: std.posix.sockaddr = undefined;
    var addrlen: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);
    if (getsockname(srv.socket.handle, &addr, &addrlen) != 0) return error.GetSockNameFailed;
    const addr_in: *const std.posix.sockaddr.in = @ptrCast(@alignCast(&addr));
    const port = std.mem.toNative(u16, addr_in.port, .big);
    if (port == 0) return error.PortIsZero;

    return .{ .server = srv, .port = port };
}

pub fn hasBearerToken(raw: []const u8, token: []const u8) bool {
    const value = headerValue(raw, "Authorization") orelse return false;
    const prefix = "Bearer ";
    if (!std.mem.startsWith(u8, value, prefix)) return false;
    return std.mem.eql(u8, value[prefix.len..], token);
}

test "Content-Length header parser" {
    try std.testing.expectEqual(
        @as(?usize, 42),
        parseContentLength("POST / HTTP/1.1\r\nContent-Length: 42\r\n\r\n"),
    );
    try std.testing.expectEqual(
        @as(?usize, 7),
        parseContentLength("POST / HTTP/1.1\r\nHost: x\r\ncontent-length:   7  \r\n\r\n"),
    );
    try std.testing.expectEqual(
        @as(?usize, null),
        parseContentLength("POST / HTTP/1.1\r\nHost: x\r\n\r\n"),
    );
    try std.testing.expectEqual(
        @as(?usize, null),
        parseContentLength("POST / HTTP/1.1\r\nContent-Length: abc\r\n\r\n"),
    );
}

test "request target rejects oversized Content-Length without overflow" {
    try std.testing.expectEqual(@as(?usize, 48), requestTargetWithinBuffer(40, 8, 64));
    try std.testing.expectEqual(@as(?usize, null), requestTargetWithinBuffer(40, 25, 64));
    try std.testing.expectEqual(@as(?usize, null), requestTargetWithinBuffer(40, std.math.maxInt(usize), 64));
    try std.testing.expectEqual(@as(?usize, null), requestTargetWithinBuffer(65, 0, 64));
}

test "Authorization bearer parser" {
    const raw =
        "POST / HTTP/1.1\r\n" ++
        "Host: 127.0.0.1\r\n" ++
        "authorization:   Bearer local-token  \r\n" ++
        "Content-Length: 2\r\n\r\n{}";

    try std.testing.expect(hasBearerToken(raw, "local-token"));
    try std.testing.expect(!hasBearerToken(raw, "wrong-token"));
    try std.testing.expect(!hasBearerToken("POST / HTTP/1.1\r\n\r\n{}", "local-token"));
    try std.testing.expect(!hasBearerToken("POST / HTTP/1.1\r\nAuthorization: Basic nope\r\n\r\n{}", "local-token"));
}

test {
    std.testing.refAllDecls(@This());
}

test "MAX_REQUEST_SIZE constant" {
    try std.testing.expectEqual(@as(usize, 64 * 1024), MAX_REQUEST_SIZE);
}
