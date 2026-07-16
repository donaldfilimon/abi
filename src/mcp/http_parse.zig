const std = @import("std");

pub fn parseHttpPort(raw: []const u8) ?u16 {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    const port = std.fmt.parseInt(u16, trimmed, 10) catch return null;
    if (port == 0) return null;
    return port;
}

pub fn parseHttpToken(raw: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    return trimmed;
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

pub fn hasBearerToken(raw: []const u8, token: []const u8) bool {
    const value = headerValue(raw, "Authorization") orelse return false;
    const prefix = "Bearer ";
    if (!std.mem.startsWith(u8, value, prefix)) return false;
    return std.mem.eql(u8, value[prefix.len..], token);
}

test {
    std.testing.refAllDecls(@This());
}

test "MCP HTTP port parser accepts valid user ports" {
    try std.testing.expectEqual(@as(?u16, 18080), parseHttpPort("18080"));
    try std.testing.expectEqual(@as(?u16, 18080), parseHttpPort(" 18080\n"));
}

test "MCP HTTP port parser rejects invalid overrides" {
    try std.testing.expectEqual(@as(?u16, null), parseHttpPort(""));
    try std.testing.expectEqual(@as(?u16, null), parseHttpPort("0"));
    try std.testing.expectEqual(@as(?u16, null), parseHttpPort("65536"));
    try std.testing.expectEqual(@as(?u16, null), parseHttpPort("not-a-port"));
}

test "MCP HTTP token parser trims and rejects empty overrides" {
    try std.testing.expectEqualStrings("local-token", parseHttpToken(" local-token\n") orelse return error.MissingToken);
    try std.testing.expect(parseHttpToken("") == null);
    try std.testing.expect(parseHttpToken(" \t\r\n") == null);
}

test "MCP HTTP Content-Length header parser" {
    try std.testing.expectEqual(
        @as(?usize, 42),
        parseContentLength("POST /message HTTP/1.1\r\nContent-Length: 42\r\n\r\n"),
    );
    try std.testing.expectEqual(
        @as(?usize, 7),
        parseContentLength("POST /message HTTP/1.1\r\nHost: x\r\ncontent-length:   7  \r\n\r\n"),
    );
    try std.testing.expectEqual(
        @as(?usize, null),
        parseContentLength("POST /message HTTP/1.1\r\nHost: x\r\n\r\n"),
    );
    try std.testing.expectEqual(
        @as(?usize, null),
        parseContentLength("POST /message HTTP/1.1\r\nContent-Length: abc\r\n\r\n"),
    );
}

test "MCP HTTP request target rejects oversized Content-Length without overflow" {
    try std.testing.expectEqual(@as(?usize, 48), requestTargetWithinBuffer(40, 8, 64));
    try std.testing.expectEqual(@as(?usize, null), requestTargetWithinBuffer(40, 25, 64));
    try std.testing.expectEqual(@as(?usize, null), requestTargetWithinBuffer(40, std.math.maxInt(usize), 64));
    try std.testing.expectEqual(@as(?usize, null), requestTargetWithinBuffer(65, 0, 64));
}

test "MCP HTTP Authorization bearer parser" {
    const raw =
        "POST /message HTTP/1.1\r\n" ++
        "Host: 127.0.0.1\r\n" ++
        "authorization:   Bearer local-token  \r\n" ++
        "Content-Length: 2\r\n\r\n{}";

    try std.testing.expect(hasBearerToken(raw, "local-token"));
    try std.testing.expect(!hasBearerToken(raw, "wrong-token"));
    try std.testing.expect(!hasBearerToken("POST /message HTTP/1.1\r\n\r\n{}", "local-token"));
    try std.testing.expect(!hasBearerToken("POST /message HTTP/1.1\r\nAuthorization: Basic nope\r\n\r\n{}", "local-token"));
}
