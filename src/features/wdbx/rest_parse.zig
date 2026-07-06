const std = @import("std");
const env = @import("../../foundation/env.zig");

pub const REST_TOKEN_ENV = "ABI_WDBX_REST_TOKEN";

pub const Response = struct {
    status: u16,
    body: []u8, // owned by the caller

    pub fn deinit(self: *Response, allocator: std.mem.Allocator) void {
        allocator.free(self.body);
    }
};

pub fn json(allocator: std.mem.Allocator, status: u16, comptime fmt: []const u8, args: anytype) !Response {
    return .{ .status = status, .body = try std.fmt.allocPrint(allocator, fmt, args) };
}

pub const VectorParseError = error{ NotArray, Empty, NonNumber, OutOfMemory };

pub fn parseVectorField(allocator: std.mem.Allocator, vec_node: std.json.Value) VectorParseError![]f32 {
    const arr = switch (vec_node) {
        .array => |a| a,
        else => return error.NotArray,
    };
    if (arr.items.len == 0) return error.Empty;
    const out = try allocator.alloc(f32, arr.items.len);
    errdefer allocator.free(out);
    for (arr.items, 0..) |item, i| {
        out[i] = switch (item) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => return error.NonNumber,
        };
    }
    return out;
}

pub fn vectorParseErrorResponse(allocator: std.mem.Allocator, err: VectorParseError) !Response {
    return switch (err) {
        error.NotArray => json(allocator, 400, "{{\"error\":\"vector must be an array\"}}", .{}),
        error.Empty => json(allocator, 400, "{{\"error\":\"vector must be non-empty\"}}", .{}),
        error.NonNumber => json(allocator, 400, "{{\"error\":\"vector elements must be numbers\"}}", .{}),
        error.OutOfMemory => json(allocator, 500, "{{\"error\":\"oom\"}}", .{}),
    };
}

pub fn escapeJsonString(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0x08 => try out.appendSlice(allocator, "\\b"),
            0x0c => try out.appendSlice(allocator, "\\f"),
            0x00...0x07, 0x0b, 0x0e...0x1f => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            else => try out.append(allocator, byte),
        }
    }
    return out.toOwnedSlice(allocator);
}

pub fn strField(v: std.json.Value) ?[]const u8 {
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

pub fn reasonPhrase(status: u16) []const u8 {
    return switch (status) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        else => "OK",
    };
}

pub fn findBody(raw: []const u8) []const u8 {
    if (std.mem.indexOf(u8, raw, "\r\n\r\n")) |i| return raw[i + 4 ..];
    if (std.mem.indexOf(u8, raw, "\n\n")) |i| return raw[i + 2 ..];
    return "";
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
                const target = end + declared;
                want_total = if (target > buf.len) buf.len + 1 else target;
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

pub fn hasBearerToken(raw: []const u8, token: []const u8) bool {
    const value = headerValue(raw, "Authorization") orelse return false;
    const prefix = "Bearer ";
    if (!std.mem.startsWith(u8, value, prefix)) return false;
    return std.mem.eql(u8, value[prefix.len..], token);
}

pub fn loadBearerToken() ?[]const u8 {
    const raw = env.get(REST_TOKEN_ENV) orelse return null;
    const token = std.mem.trim(u8, raw, " \t\r\n");
    if (token.len == 0) return null;
    return token;
}

test "rest: Content-Length header parser" {
    try std.testing.expectEqual(
        @as(?usize, 42),
        parseContentLength("POST /insert HTTP/1.1\r\nContent-Length: 42\r\n\r\n"),
    );
    try std.testing.expectEqual(
        @as(?usize, 7),
        parseContentLength("POST /insert HTTP/1.1\r\nHost: x\r\ncontent-length:   7  \r\n\r\n"),
    );
    try std.testing.expectEqual(
        @as(?usize, null),
        parseContentLength("POST /insert HTTP/1.1\r\nHost: x\r\n\r\n"),
    );
    try std.testing.expectEqual(
        @as(?usize, null),
        parseContentLength("POST /insert HTTP/1.1\r\nContent-Length: abc\r\n\r\n"),
    );
}

test "rest: Authorization bearer parser" {
    const raw =
        "POST /insert HTTP/1.1\r\n" ++
        "Host: 127.0.0.1\r\n" ++
        "authorization:   Bearer local-token  \r\n" ++
        "Content-Length: 2\r\n\r\n{}";

    try std.testing.expect(hasBearerToken(raw, "local-token"));
    try std.testing.expect(!hasBearerToken(raw, "wrong-token"));
    try std.testing.expect(!hasBearerToken("POST /insert HTTP/1.1\r\n\r\n{}", "local-token"));
    try std.testing.expect(!hasBearerToken("POST /insert HTTP/1.1\r\nAuthorization: Basic nope\r\n\r\n{}", "local-token"));
}

test {
    std.testing.refAllDecls(@This());
}
