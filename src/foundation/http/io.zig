const std = @import("std");
const headers = @import("headers.zig");

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
    /// Headers declared a Content-Length but the connection ended before that
    /// many body bytes arrived. Callers must not dispatch the partial payload.
    incomplete,
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
                const declared = headers.parseContentLength(buf[0..end]) orelse 0;
                want_total = headers.requestTargetWithinBuffer(end, declared, buf.len) orelse return .too_large;
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
    if (want_total) |want| {
        if (total < want) return .incomplete;
    }
    if (header_end == null) return .incomplete;
    return .{ .request = buf[0..total] };
}

test "HttpReadResult incomplete is distinct from empty and too_large" {
    const tag = HttpReadResult.incomplete;
    try std.testing.expect(tag == .incomplete);
    try std.testing.expect(tag != .empty);
    try std.testing.expect(tag != .too_large);
}

test "MAX_REQUEST_SIZE constant" {
    try std.testing.expectEqual(@as(usize, 64 * 1024), MAX_REQUEST_SIZE);
}

test {
    std.testing.refAllDecls(@This());
}
