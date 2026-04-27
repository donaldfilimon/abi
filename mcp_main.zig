//! ABI MCP Server — Standalone stdio entry point.
//!
//! Launches the ABI MCP server with database + ZLS tools exposed via
//! MCP Content-Length framed JSON-RPC over stdin/stdout for Claude Desktop,
//! Cursor, and other MCP clients.
//!
//! Usage:
//!   zig-out/bin/abi-mcp              # Start stdio server
//!   # Use an MCP client or send Content-Length framed JSON-RPC messages

const std = @import("std");
const build_options = @import("build_options");

const mcp = @import("protocols/mcp/mod.zig");

// Debug REPL removed for stability. MCP operates via JSON-RPC over stdin/stdout.

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // CLI arguments are not used in this minimal runtime; proceed to server startup

    const version = build_options.package_version;

    // Create combined MCP server with all tools
    var server = mcp.createCombinedServer(allocator, version) catch |err| {
        std.log.err("Failed to create MCP server: {}", .{err});
        return err;
    };
    defer server.deinit();

    // Run stdio event loop with Zig 0.16 I/O
    server.run(init.io) catch |err| {
        std.log.err("MCP server error: {}", .{err});
        return err;
    };
}
