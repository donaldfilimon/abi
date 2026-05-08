const std = @import("std");

const mcp_main = @import("mcp_main.zig");

test "cli: --debug flag detected" {
    const args = [_][]const u8{ "abi-mcp", "--debug" };
    const config = mcp_main.parseArgs(&args);
    try std.testing.expect(config.debug == true);
}
