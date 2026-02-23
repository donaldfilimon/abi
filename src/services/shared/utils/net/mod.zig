//! Network Utilities
//!
//! Low-level network helper functions for parsing and validating
//! network addresses, host:port pairs, and related operations.

const std = @import("std");

pub const NetError = error{
    InvalidHostPort,
};

pub const HostPort = struct {
    host: []const u8,
    port: u16,

    pub fn deinit(self: *HostPort, allocator: std.mem.Allocator) void {
        allocator.free(self.host);
        self.* = undefined;
    }
};

pub fn parseHostPort(allocator: std.mem.Allocator, input: []const u8) !HostPort {
    if (input.len == 0) return NetError.InvalidHostPort;

    if (input[0] == '[') {
        const end = std.mem.indexOfScalar(u8, input, ']') orelse
            return NetError.InvalidHostPort;
        if (end + 1 >= input.len or input[end + 1] != ':')
            return NetError.InvalidHostPort;
        const host = try allocator.dupe(u8, input[1..end]);
        errdefer allocator.free(host);
        const port = std.fmt.parseInt(u16, input[end + 2 ..], 10) catch
            return NetError.InvalidHostPort;
        return .{
            .host = host,
            .port = port,
        };
    }

    if (std.mem.count(u8, input, ":") > 1) return NetError.InvalidHostPort;
    const index = std.mem.lastIndexOfScalar(u8, input, ':') orelse
        return NetError.InvalidHostPort;
    if (index == 0 or index + 1 >= input.len) return NetError.InvalidHostPort;

    const host = try allocator.dupe(u8, input[0..index]);
    errdefer allocator.free(host);
    const port = std.fmt.parseInt(u16, input[index + 1 ..], 10) catch
        return NetError.InvalidHostPort;

    return .{ .host = host, .port = port };
}

test "parseHostPort handles ipv4 and ipv6" {
    const allocator = std.testing.allocator;

    var hp = try parseHostPort(allocator, "127.0.0.1:8080");
    defer hp.deinit(allocator);
    try std.testing.expectEqualStrings("127.0.0.1", hp.host);
    try std.testing.expectEqual(@as(u16, 8080), hp.port);

    var hp6 = try parseHostPort(allocator, "[::1]:9090");
    defer hp6.deinit(allocator);
    try std.testing.expectEqualStrings("::1", hp6.host);
    try std.testing.expectEqual(@as(u16, 9090), hp6.port);
}

test {
    std.testing.refAllDecls(@This());
}
