const std = @import("std");
const foundation = @import("abi").foundation.http;

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

pub const parseContentLength = foundation.parseContentLength;
pub const requestTargetWithinBuffer = foundation.requestTargetWithinBuffer;
pub const headerValue = foundation.headerValue;
pub const hasBearerToken = foundation.hasBearerToken;

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
