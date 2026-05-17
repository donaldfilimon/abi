const std = @import("std");

pub fn trimWhitespace(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, &std.ascii.whitespace);
}

pub fn splitLines(s: []const u8, allocator: std.mem.Allocator) !std.ArrayListUnmanaged([]const u8) {
    var result = std.ArrayListUnmanaged([]const u8).empty;
    errdefer {
        for (result.items) |line| {
            allocator.free(line);
        }
        result.deinit(allocator);
    }

    var it = std.mem.splitScalar(u8, s, '\n');
    while (it.next()) |line| {
        const trimmed = trimWhitespace(line);
        const copy = try allocator.dupe(u8, trimmed);
        errdefer allocator.free(copy);
        try result.append(allocator, copy);
    }
    return result;
}

pub fn joinStrings(parts: []const []const u8, separator: []const u8, allocator: std.mem.Allocator) ![]const u8 {
    if (parts.len == 0) return try allocator.dupe(u8, "");

    var total_len: usize = 0;
    for (parts) |part| {
        total_len += part.len;
    }
    total_len += separator.len * (parts.len - 1);

    var result = std.ArrayListUnmanaged(u8).empty;
    errdefer result.deinit(allocator);

    try result.ensureTotalCapacity(allocator, total_len);

    for (parts, 0..) |part, i| {
        if (i > 0) {
            result.appendSlice(allocator, separator) catch |err| {
                result.deinit(allocator);
                return err;
            };
        }
        result.appendSlice(allocator, part) catch |err| {
            result.deinit(allocator);
            return err;
        };
    }

    return result.toOwnedSlice(allocator);
}

pub fn pathJoin(base: []const u8, relative: []const u8, allocator: std.mem.Allocator) ![]const u8 {
    if (base.len == 0) return try allocator.dupe(u8, relative);
    if (relative.len == 0) return try allocator.dupe(u8, base);

    const sep = if (base[base.len - 1] == '/' or base[base.len - 1] == '\\') "" else "/";
    const full_len = base.len + sep.len + relative.len;

    var buf = try allocator.alloc(u8, full_len);
    errdefer allocator.free(buf);

    @memcpy(buf[0..base.len], base);
    @memcpy(buf[base.len .. base.len + sep.len], sep);
    @memcpy(buf[base.len + sep.len ..], relative);

    return buf;
}

pub fn pathBasename(path: []const u8) []const u8 {
    var i: usize = path.len;
    while (i > 0) {
        i -= 1;
        if (path[i] == '/' or path[i] == '\\') {
            return path[i + 1 ..];
        }
    }
    return path;
}

pub fn pathDirname(path: []const u8) []const u8 {
    var i: usize = path.len;
    while (i > 0) {
        i -= 1;
        if (path[i] == '/' or path[i] == '\\') {
            if (i == 0) return "/";
            return path[0..i];
        }
    }
    return ".";
}

pub fn pathExt(path: []const u8) []const u8 {
    const base = pathBasename(path);
    var i: usize = base.len;
    while (i > 0) {
        i -= 1;
        if (base[i] == '.') {
            return base[i..];
        }
    }
    return "";
}

pub fn startsWith(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    return std.mem.eql(u8, haystack[0..needle.len], needle);
}

pub fn endsWith(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    return std.mem.eql(u8, haystack[haystack.len - needle.len ..], needle);
}

pub fn contains(haystack: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, haystack, needle) != null;
}

test {
    std.testing.refAllDecls(@This());
}

test "trimWhitespace" {
    try std.testing.expectEqualStrings("hello", trimWhitespace("  hello  "));
    try std.testing.expectEqualStrings("", trimWhitespace("   "));
    try std.testing.expectEqualStrings("a b", trimWhitespace("\ta b\n"));
}

test "splitLines" {
    var result = try splitLines("line1\nline2\nline3", std.testing.allocator);
    defer {
        for (result.items) |line| {
            std.testing.allocator.free(line);
        }
        result.deinit(std.testing.allocator);
    }

    try std.testing.expectEqual(@as(usize, 3), result.items.len);
    try std.testing.expectEqualStrings("line1", result.items[0]);
    try std.testing.expectEqualStrings("line2", result.items[1]);
    try std.testing.expectEqualStrings("line3", result.items[2]);
}

test "joinStrings" {
    const parts = [_][]const u8{ "a", "b", "c" };
    const result = try joinStrings(&parts, "-", std.testing.allocator);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqualStrings("a-b-c", result);
}

test "joinStrings empty" {
    const parts = [_][]const u8{};
    const result = try joinStrings(&parts, "-", std.testing.allocator);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqualStrings("", result);
}

test "pathJoin" {
    const result = try pathJoin("/home/user", "docs/file.txt", std.testing.allocator);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqualStrings("/home/user/docs/file.txt", result);
}

test "pathJoin with trailing slash" {
    const result = try pathJoin("/home/user/", "docs", std.testing.allocator);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqualStrings("/home/user/docs", result);
}

test "pathBasename" {
    try std.testing.expectEqualStrings("file.txt", pathBasename("/home/user/file.txt"));
    try std.testing.expectEqualStrings("file.txt", pathBasename("file.txt"));
    try std.testing.expectEqualStrings("", pathBasename("/"));
}

test "pathDirname" {
    try std.testing.expectEqualStrings("/home/user", pathDirname("/home/user/file.txt"));
    try std.testing.expectEqualStrings(".", pathDirname("file.txt"));
    try std.testing.expectEqualStrings("/", pathDirname("/file.txt"));
}

test "pathExt" {
    try std.testing.expectEqualStrings(".txt", pathExt("/home/user/file.txt"));
    try std.testing.expectEqualStrings("", pathExt("/home/user/file"));
    try std.testing.expectEqualStrings(".zig", pathExt("src/main.zig"));
}

test "startsWith" {
    try std.testing.expect(startsWith("hello world", "hello"));
    try std.testing.expect(!startsWith("hello", "hello world"));
    try std.testing.expect(startsWith("abc", ""));
}

test "endsWith" {
    try std.testing.expect(endsWith("hello world", "world"));
    try std.testing.expect(!endsWith("world", "hello world"));
    try std.testing.expect(endsWith("abc", ""));
}

test "contains" {
    try std.testing.expect(contains("hello world", "lo wo"));
    try std.testing.expect(!contains("hello", "xyz"));
    try std.testing.expect(contains("abc", ""));
}
