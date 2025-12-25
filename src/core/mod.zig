const std = @import("std");

pub const Allocator = std.mem.Allocator;

pub const CoreError = error{
    InvalidState,
    OutOfMemory,
};

pub const Version = struct {
    major: u16,
    minor: u16,
    patch: u16,

    pub fn format(self: Version, buffer: []u8) ![]u8 {
        return formatVersion(buffer, self);
    }

    pub fn isZero(self: Version) bool {
        return self.major == 0 and self.minor == 0 and self.patch == 0;
    }

    pub fn toInt(self: Version) u64 {
        return (@as(u64, self.major) << 32) |
            (@as(u64, self.minor) << 16) |
            @as(u64, self.patch);
    }
};

pub const profile = @import("profile.zig");

pub fn parseVersion(text: []const u8) ?Version {
    var it = std.mem.splitScalar(u8, text, '.');
    const major = it.next() orelse return null;
    const minor = it.next() orelse return null;
    const patch = it.next() orelse return null;
    if (it.next() != null) return null;
    return Version{
        .major = std.fmt.parseInt(u16, major, 10) catch return null,
        .minor = std.fmt.parseInt(u16, minor, 10) catch return null,
        .patch = std.fmt.parseInt(u16, patch, 10) catch return null,
    };
}

pub fn parseVersionLoose(text: []const u8) ?Version {
    var trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return null;
    if (trimmed[0] == 'v' or trimmed[0] == 'V') {
        if (trimmed.len == 1) return null;
        trimmed = trimmed[1..];
    }
    const end = std.mem.indexOfAny(u8, trimmed, "-+") orelse trimmed.len;
    if (end == 0) return null;
    return parseVersion(trimmed[0..end]);
}

pub fn compareVersion(a: Version, b: Version) std.math.Order {
    if (a.major != b.major) return std.math.order(a.major, b.major);
    if (a.minor != b.minor) return std.math.order(a.minor, b.minor);
    return std.math.order(a.patch, b.patch);
}

pub fn formatVersion(buffer: []u8, version: Version) ![]u8 {
    return std.fmt.bufPrint(buffer, "{d}.{d}.{d}", .{ version.major, version.minor, version.patch });
}

pub fn formatVersionAlloc(allocator: std.mem.Allocator, version: Version) ![]u8 {
    return std.fmt.allocPrint(allocator, "{d}.{d}.{d}", .{ version.major, version.minor, version.patch });
}

pub fn isCompatible(required: Version, current: Version) bool {
    if (required.major != current.major) return false;
    return compareVersion(current, required) != .lt;
}

test "parse and format version" {
    const version = parseVersion("1.2.3") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u16, 1), version.major);
    try std.testing.expectEqual(@as(u16, 2), version.minor);
    try std.testing.expectEqual(@as(u16, 3), version.patch);

    var buffer: [32]u8 = undefined;
    const formatted = try formatVersion(&buffer, version);
    try std.testing.expectEqualStrings("1.2.3", formatted);
}

test "compare versions" {
    const a = Version{ .major = 1, .minor = 0, .patch = 0 };
    const b = Version{ .major = 1, .minor = 1, .patch = 0 };
    try std.testing.expect(compareVersion(a, b) == .lt);
    try std.testing.expect(compareVersion(b, a) == .gt);
    try std.testing.expect(compareVersion(a, a) == .eq);
}

test "loose version parsing and compatibility" {
    const version = parseVersionLoose("v2.3.4-beta1") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u16, 2), version.major);
    try std.testing.expectEqual(@as(u16, 3), version.minor);
    try std.testing.expectEqual(@as(u16, 4), version.patch);

    try std.testing.expect(isCompatible(.{ .major = 2, .minor = 0, .patch = 0 }, version));
    try std.testing.expect(!isCompatible(.{ .major = 3, .minor = 0, .patch = 0 }, version));

    const allocator = std.testing.allocator;
    const formatted = try formatVersionAlloc(allocator, version);
    defer allocator.free(formatted);
    try std.testing.expectEqualStrings("2.3.4", formatted);
    try std.testing.expect(!version.isZero());
    try std.testing.expect(version.toInt() > 0);
}
