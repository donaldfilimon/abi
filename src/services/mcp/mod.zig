//! MCP (Model Context Protocol) Service module switcher.

const build_options = @import("build_options");

pub usingnamespace if (build_options.feat_mcp)
    @import("real.zig")
else
    @import("stub.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
