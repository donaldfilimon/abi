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

const TransportMode = enum {
    stdio,
    sse,
};

pub const ArgsConfig = struct {
    mode: TransportMode = .stdio,
    debug: bool = false,
    help: bool = false,
    host: []const u8 = "127.0.0.1",
    port: u16 = 8081,
};

fn envSlice(comptime name: [:0]const u8) ?[]const u8 {
    const value = std.c.getenv(name.ptr) orelse return null;
    return std.mem.span(value);
}

fn envPort(comptime name: [:0]const u8, default: u16) u16 {
    const value = envSlice(name) orelse return default;
    return std.fmt.parseInt(u16, value, 10) catch default;
}

pub fn parseArgs(args: anytype) ArgsConfig {
    var config = ArgsConfig{
        .host = envSlice("ABI_MCP_HOST") orelse "127.0.0.1",
        .port = envPort("ABI_MCP_PORT", 8081),
    };

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--debug")) {
            config.debug = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help")) {
            config.help = true;
        } else if (std.mem.eql(u8, arg, "stdio")) {
            config.mode = .stdio;
        } else if (std.mem.eql(u8, arg, "sse") or std.mem.eql(u8, arg, "http")) {
            config.mode = .sse;
        } else if (std.mem.eql(u8, arg, "--host") or std.mem.eql(u8, arg, "--addr")) {
            if (i + 1 < args.len) {
                i += 1;
                config.host = args[i];
            }
        } else if (std.mem.eql(u8, arg, "--port")) {
            if (i + 1 < args.len) {
                i += 1;
                config.port = std.fmt.parseInt(u16, args[i], 10) catch config.port;
            }
        }
    }

    return config;
}

// Top-level debug REPL so it can be invoked from `main`.
fn runDebugRepl(allocator: std.mem.Allocator, io: anytype) !void {
    // We intentionally ignore the generic `io` parameter; use std I/O handles for REPL.
    _ = allocator;
    var out_buf: [4096]u8 = undefined;
    var in_buf: [4096]u8 = undefined;
    const stdout_writer = std.Io.File.stdout().writer(io, &out_buf);
    var stdout = stdout_writer.interface;
    const stdin_reader = std.Io.File.stdin().reader(io, &in_buf);
    var stdin = stdin_reader.interface;
    try stdout.print("ABI MCP Debug Mode. Enter JSON-RPC 2.0 requests.\n", .{});

    while (stdin.takeDelimiterExclusive('\n')) |line| {
        // Echo input for testing framing
        try stdout.print("Echo: {s}\n", .{line});
    } else |err| {
        if (err != error.EndOfStream) {
            std.log.err("REPL error: {t}", .{err});
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const config = parseArgs(args);

    if (config.debug) {
        // Call the top-level REPL and return.
        try runDebugRepl(allocator, init.io);
        return;
    }
    if (config.help) {
        const help_text = "ABI MCP Server - Model Context Protocol interface.\n\nUsage:\n  abi-mcp                         Start the JSON-RPC 2.0 stdio server\n  abi-mcp stdio                   Start the JSON-RPC 2.0 stdio server\n  abi-mcp sse [--host H] [--port P] Start HTTP/SSE transport with /health\n  abi-mcp --debug                 Start in debug REPL mode for manual testing\n  abi-mcp --help                  Show this help message\n\nEnvironment:\n  ABI_MCP_HOST                    Host for sse mode (default 127.0.0.1)\n  ABI_MCP_PORT                    Port for sse mode (default 8081)\n\nThis server exposes ABI framework tools (status, database, AI, ZLS) for use with MCP-compatible AI clients.\n";
        std.debug.print("{s}", .{help_text});
        return;
    }

    const version = build_options.package_version;

    // Create combined MCP server with all tools
    var server = mcp.createCombinedServer(allocator, version) catch |err| {
        std.log.err("Failed to create MCP server: {}", .{err});
        return err;
    };
    defer server.deinit();

    switch (config.mode) {
        .stdio => server.run(init.io) catch |err| {
            std.log.err("MCP stdio server error: {}", .{err});
            return err;
        },
        .sse => mcp.transport.runSse(&server, init.io, .{
            .host = config.host,
            .port = config.port,
        }) catch |err| {
            std.log.err("MCP SSE server error: {}", .{err});
            return err;
        },
    }
}

test "parseArgs debug flag detected" {
    const args = [_][]const u8{ "abi-mcp", "--debug" };
    const config = parseArgs(&args);
    try std.testing.expect(config.debug);
}

test "parseArgs sse host and port" {
    const args = [_][]const u8{ "abi-mcp", "sse", "--host", "0.0.0.0", "--port", "50051" };
    const config = parseArgs(&args);
    try std.testing.expect(config.mode == .sse);
    try std.testing.expectEqualStrings("0.0.0.0", config.host);
    try std.testing.expectEqual(@as(u16, 50051), config.port);
}
