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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    if (args.len > 1) {
        const arg = args[1];
        if (std.mem.eql(u8, arg, "--debug")) {
            try runDebugRepl(allocator, init.io);
            return;
        }
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help")) {
            const help_text =
                \\ABI MCP Server — Model Context Protocol stdio interface.
                \\
                \\Usage:
                \\  abi-mcp              Start the JSON-RPC 2.0 stdio server
                \\  abi-mcp --debug      Start in debug REPL mode for manual testing
                \\  abi-mcp --help       Show this help message
                \\
                \\This server exposes ABI framework tools (status, database, AI, ZLS)
                \\over stdin/stdout for use with MCP-compatible AI clients.
                \\
            ;
            std.debug.print("{s}", .{help_text});
            return;
        }
    }

fn runDebugRepl(allocator: std.mem.Allocator, io: anytype) !void {
    _ = io;
    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();
    try stdout.print("ABI MCP Debug Mode. Enter JSON-RPC 2.0 requests.\n", .{});

    var buf: [4096]u8 = undefined;
    while (stdin.readUntilDelimiterOrEof(&buf, '\n') catch null) |line| {
        // Echo input for testing framing
        try stdout.print("Echo: {s}\n", .{line});
    }
}

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
