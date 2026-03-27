//! ABI MCP Server — Standalone stdio entry point.
//!
//! Launches the ABI MCP server with database + ZLS tools exposed via
//! JSON-RPC 2.0 over stdin/stdout for use with Claude Desktop, Cursor, etc.
//!
//! Usage:
//!   zig-out/bin/abi-mcp              # Start stdio server
//!   echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}' | zig-out/bin/abi-mcp

const std = @import("std");
const build_options = @import("build_options");
const mcp = @import("protocols/mcp/mod.zig");

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const version = build_options.package_version;

    // Create combined MCP server with all tools
    var server = mcp.createCombinedServer(allocator, version) catch |err| {
        std.log.err("Failed to create MCP server: {}", .{err});
        return;
    };
    defer server.deinit();

    // Run stdio event loop with Zig 0.16 I/O
    server.run(init.io) catch |err| {
        std.log.err("MCP server error: {}", .{err});
    };
}
