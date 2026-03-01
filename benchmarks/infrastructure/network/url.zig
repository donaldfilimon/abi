//! URL parsing, encoding, and decoding benchmarks.

const std = @import("std");

pub const ParsedUrl = struct {
    scheme: []const u8,
    host: []const u8,
    port: ?u16,
    path: []const u8,
    query: []const u8,
    fragment: []const u8,
};

pub fn parseUrl(url: []const u8) ParsedUrl {
    var result = ParsedUrl{
        .scheme = "",
        .host = "",
        .port = null,
        .path = "/",
        .query = "",
        .fragment = "",
    };

    var rest = url;

    if (std.mem.indexOf(u8, rest, "://")) |idx| {
        result.scheme = rest[0..idx];
        rest = rest[idx + 3 ..];
    }
    if (std.mem.indexOf(u8, rest, "#")) |idx| {
        result.fragment = rest[idx + 1 ..];
        rest = rest[0..idx];
    }
    if (std.mem.indexOf(u8, rest, "?")) |idx| {
        result.query = rest[idx + 1 ..];
        rest = rest[0..idx];
    }
    if (std.mem.indexOf(u8, rest, "/")) |idx| {
        result.path = rest[idx..];
        rest = rest[0..idx];
    }
    if (std.mem.lastIndexOf(u8, rest, ":")) |idx| {
        if (std.fmt.parseInt(u16, rest[idx + 1 ..], 10)) |port| {
            result.port = port;
            rest = rest[0..idx];
        } else |_| {}
    }

    result.host = rest;
    return result;
}

pub fn urlEncode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var result = std.ArrayListUnmanaged(u8).empty;
    errdefer result.deinit(allocator);

    for (input) |c| {
        if (std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.' or c == '~') {
            try result.append(allocator, c);
        } else {
            try result.append(allocator, '%');
            try result.append(allocator, std.fmt.digitToChar(c >> 4, .upper));
            try result.append(allocator, std.fmt.digitToChar(c & 0x0F, .upper));
        }
    }
    return result.toOwnedSlice(allocator);
}

pub fn urlDecode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var result = std.ArrayListUnmanaged(u8).empty;
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < input.len) {
        if (input[i] == '%' and i + 2 < input.len) {
            const hi = std.fmt.charToDigit(input[i + 1], 16) catch {
                try result.append(allocator, input[i]);
                i += 1;
                continue;
            };
            const lo = std.fmt.charToDigit(input[i + 2], 16) catch {
                try result.append(allocator, input[i]);
                i += 1;
                continue;
            };
            try result.append(allocator, (hi << 4) | lo);
            i += 3;
        } else if (input[i] == '+') {
            try result.append(allocator, ' ');
            i += 1;
        } else {
            try result.append(allocator, input[i]);
            i += 1;
        }
    }
    return result.toOwnedSlice(allocator);
}

pub fn benchUrlParsing(url: []const u8) void {
    const parsed = parseUrl(url);
    std.mem.doNotOptimizeAway(&parsed);
}

pub fn benchUrlEncoding(allocator: std.mem.Allocator, input: []const u8) !void {
    const encoded = try urlEncode(allocator, input);
    defer allocator.free(encoded);
    std.mem.doNotOptimizeAway(encoded.ptr);
}

// ============================================================================
// Tests
// ============================================================================

test "url parsing" {
    const url = "https://api.example.com:8443/v1/users?page=1#section";
    const parsed = parseUrl(url);

    try std.testing.expectEqualStrings("https", parsed.scheme);
    try std.testing.expectEqualStrings("api.example.com", parsed.host);
    try std.testing.expectEqual(@as(?u16, 8443), parsed.port);
    try std.testing.expectEqualStrings("/v1/users", parsed.path);
    try std.testing.expectEqualStrings("page=1", parsed.query);
    try std.testing.expectEqualStrings("section", parsed.fragment);
}

test "url encoding" {
    const allocator = std.testing.allocator;
    const encoded = try urlEncode(allocator, "hello world");
    defer allocator.free(encoded);
    try std.testing.expectEqualStrings("hello%20world", encoded);

    const decoded = try urlDecode(allocator, "hello%20world");
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("hello world", decoded);
}
