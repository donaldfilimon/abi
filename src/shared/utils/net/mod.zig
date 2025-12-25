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
    const index = std.mem.lastIndexOfScalar(u8, input, ':') orelse
        return NetError.InvalidHostPort;
    if (index == 0 or index + 1 >= input.len) return NetError.InvalidHostPort;

    const host = try allocator.dupe(u8, input[0..index]);
    errdefer allocator.free(host);
    const port = std.fmt.parseInt(u16, input[index + 1 ..], 10) catch
        return NetError.InvalidHostPort;

    return HostPort{
        .host = host,
        .port = port,
    };
}
