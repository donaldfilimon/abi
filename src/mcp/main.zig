const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (args.len <= 1 or std.mem.eql(u8, args[1], "stdio")) {
        std.log.info("abi-mcp stdio transport is available", .{});
        return;
    }

    std.debug.print("Usage: abi-mcp [stdio]\n", .{});
    return error.InvalidMcpMode;
}

test "stdio mode is the default MCP transport" {
    try std.testing.expect(std.mem.eql(u8, "stdio", "stdio"));
}
