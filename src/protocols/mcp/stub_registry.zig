//! MCP (Model Context Protocol) registry stub.
//!
//! Mirrors the full API of registry.zig, returning error.FeatureDisabled for all operations.

const std = @import("std");
const Server = @import("server.zig").Server;
const ToolHandler = @import("server.zig").ToolHandler;

/// Tool definition stub — mirrors the real registry module's public surface.
pub const ToolDef = struct {
    name: []const u8,
    description: []const u8,
    input_schema: []const u8,
    handler: ToolHandler,
};

/// Register tools stub — always returns error.FeatureDisabled.
pub fn registerTools(server: *Server, modules: anytype) !void {
    _ = server;
    _ = modules;
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
