const std = @import("std");
pub fn main() !void {
    _ = std.c.sendto;
    _ = std.c.recvfrom;
    _ = std.c.setsockopt;
    _ = std.posix.close;
    _ = std.c.bind;
}
