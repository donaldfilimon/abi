const std = @import("std");

pub fn main() !void {
    const sin: std.c.sockaddr.in = .{
        .family = std.posix.AF.INET,
        .port = std.mem.nativeToBig(u16, 11435),
        .addr = @bitCast([4]u8{ 0, 0, 0, 0 }),
    };
    _ = sin;
}
