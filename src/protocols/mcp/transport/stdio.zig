//! MCP Stdio Transport — reads JSON-RPC messages from stdin, writes to stdout.
//!
//! Extracted from the original io_loop to support the transport abstraction.
//! This is the default transport for CLI usage with Claude Desktop, Cursor, etc.

const std = @import("std");
const types = @import("../types.zig");

/// Configuration for the stdio transport.
pub const Config = struct {
    /// Read buffer size for stdin.
    read_buf_size: usize = 65536,
    /// Write buffer size for stdout.
    write_buf_size: usize = 65536,
};

/// Run the stdio transport loop: read newline-delimited JSON-RPC from stdin,
/// dispatch via `server.processMessage`, write responses to stdout.
pub fn run(server: anytype, io: std.Io) !void {
    std.log.info("MCP stdio transport ready ({d} tools registered)", .{server.tools.items.len});

    var stdin_file = std.Io.File.stdin();
    var read_buf: [65536]u8 = undefined;
    var reader = stdin_file.reader(io, &read_buf);

    var stdout_file = std.Io.File.stdout();
    var write_buf: [65536]u8 = undefined;
    var writer = stdout_file.writer(io, &write_buf);

    while (true) {
        const line_opt = reader.interface.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => {
                try types.writeError(
                    &writer.interface,
                    null,
                    types.ErrorCode.parse_error,
                    "Message too large",
                );
                try writer.flush();
                continue;
            },
            else => break,
        };

        const line = line_opt orelse break;
        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed.len == 0) continue;

        server.processMessage(trimmed, &writer.interface) catch |err| {
            std.log.err("Error handling message: {t}", .{err});
            types.writeError(
                &writer.interface,
                null,
                types.ErrorCode.internal_error,
                "Internal error",
            ) catch |write_err| {
                std.log.err("MCP: failed to write error response: {t}", .{write_err});
                break;
            };
        };

        writer.flush() catch |flush_err| {
            std.log.err("MCP: flush error, closing connection: {t}", .{flush_err});
            break;
        };
    }

    writer.flush() catch |err| {
        std.log.warn("MCP: final flush failed: {t}", .{err});
    };
}

test {
    const testing = std.testing;
    testing.refAllDecls(@This());
}
