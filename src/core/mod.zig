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
};

pub const profile = @import("profile.zig");

pub fn parseVersion(text: []const u8) ?Version {
    var it = std.mem.splitScalar(u8, text, '.');
    const major = it.next() orelse return null;
    const minor = it.next() orelse return null;
    const patch = it.next() orelse return null;
    return Version{
        .major = std.fmt.parseInt(u16, major, 10) catch return null,
        .minor = std.fmt.parseInt(u16, minor, 10) catch return null,
        .patch = std.fmt.parseInt(u16, patch, 10) catch return null,
    };
}
