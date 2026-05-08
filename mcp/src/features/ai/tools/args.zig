//! Shared JSON argument helpers for agent tools.

const std = @import("std");

pub fn string(obj: std.json.ObjectMap, name: []const u8) ?[]const u8 {
    const value = obj.get(name) orelse return null;
    return switch (value) {
        .string => |s| s,
        else => null,
    };
}

pub fn int(obj: std.json.ObjectMap, name: []const u8) ?i64 {
    const value = obj.get(name) orelse return null;
    return switch (value) {
        .integer => |i| i,
        else => null,
    };
}

pub fn u16OrDefault(obj: std.json.ObjectMap, name: []const u8, default: u16) ?u16 {
    const raw = int(obj, name) orelse return default;
    if (raw <= 0 or raw > std.math.maxInt(u16)) return null;
    return @intCast(raw);
}

pub fn safeHost(host: []const u8) bool {
    if (host.len == 0 or host.len > 255) return false;
    for (host) |c| {
        if (std.ascii.isAlphanumeric(c)) continue;
        switch (c) {
            '.', '-', ':', '_' => {},
            else => return false,
        }
    }
    return true;
}

test "u16OrDefault returns default for missing values" {
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{}", .{});
    defer parsed.deinit();

    try std.testing.expectEqual(@as(?u16, 8080), u16OrDefault(parsed.value.object, "port", 8080));
}

test "u16OrDefault rejects invalid ports" {
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"port\":70000}", .{});
    defer parsed.deinit();

    try std.testing.expectEqual(@as(?u16, null), u16OrDefault(parsed.value.object, "port", 8080));
}

test "safeHost rejects shell metacharacters" {
    try std.testing.expect(safeHost("127.0.0.1"));
    try std.testing.expect(safeHost("localhost"));
    try std.testing.expect(!safeHost("127.0.0.1; rm -rf /"));
}
