const std = @import("std");
const stdio = @import("../transport/stdio.zig");

pub fn run(self: anytype, io: std.Io) !void {
    return stdio.run(self, io);
}

pub fn runInfo(self: anytype) void {
    std.log.info("MCP server ready ({d} tools registered). Use run(io) with I/O backend.", .{self.tools.items.len});
}

test {
    std.testing.refAllDecls(@This());
}
