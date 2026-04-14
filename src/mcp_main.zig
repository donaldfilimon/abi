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

pub const Config = struct {
    debug: bool = false,
};

pub fn parseArgs(args: []const []const u8) Config {
    var config = Config{};
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--debug")) {
            config.debug = true;
        }
    }
    return config;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    if (args.len > 1) {
        const arg = args[1];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help")) {
            const help_text =
                \\ABI MCP Server — Model Context Protocol stdio interface.
                \\
                \\Usage:
                \\  abi-mcp              Start the JSON-RPC 2.0 stdio server
                \\  abi-mcp --debug      Start the server in debug mode (REPL)
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

    const config = parseArgs(args);
    std.debug.print("DEBUG: args.len={}, config.debug={}\n", .{ args.len, config.debug });
    if (config.debug) {
        return runDebugRepl();
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

fn runDebugRepl() !void {
    std.debug.print("Debug mode active. Echoing input...\n", .{});
    const stdin_fd = std.posix.STDIN_FILENO;
    const stdout_fd = std.posix.STDOUT_FILENO;

    var buf: [1024]u8 = undefined;
    const prompt = "abi-mcp> ";

    while (true) {
        // Write prompt
        _ = std.c.write(stdout_fd, prompt.ptr, prompt.len);

        // Read line from stdin
        const n = std.c.read(stdin_fd, &buf, buf.len);
        if (n <= 0) break;

        const line_len = @as(usize, @intCast(n));
        // Remove trailing newline if present
        const content_len = if (line_len > 0 and buf[line_len - 1] == '\n')
            line_len - 1
        else
            line_len;

        // Echo back with prefix
        const echo_prefix = "Echo: ";
        _ = std.c.write(stdout_fd, echo_prefix.ptr, echo_prefix.len);
        _ = std.c.write(stdout_fd, &buf, content_len);
        const newline = "\n";
        _ = std.c.write(stdout_fd, newline.ptr, newline.len);
    }
}
