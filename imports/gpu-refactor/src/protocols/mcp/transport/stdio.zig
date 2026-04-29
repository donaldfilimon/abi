//! MCP Stdio Transport — reads framed JSON-RPC messages from stdin, writes to stdout.
//!
//! MCP stdio uses `Content-Length` framing, matching the LSP transport style.
//! This is the default transport for CLI usage with Claude Desktop, Cursor, etc.

const std = @import("std");
const types = @import("../types.zig");
const framing = @import("framing.zig");

/// Configuration for the stdio transport.
///
/// The transport currently uses fixed internal buffers, but this type stays
/// part of the public surface for compatibility with the stub module.
pub const Config = struct {
    /// Read buffer size for stdin.
    read_buf_size: usize = 65536,
    /// Write buffer size for stdout.
    write_buf_size: usize = 65536,
};

/// Maximum message size accepted by the stdio transport.
const MAX_MESSAGE_SIZE: usize = 4 * 1024 * 1024;

fn writeFramedError(
    allocator: std.mem.Allocator,
    writer: anytype,
    code: i32,
    message: []const u8,
) !void {
    var payload_writer: std.Io.Writer.Allocating = .init(allocator);
    errdefer payload_writer.deinit();

    try types.writeError(&payload_writer.writer, null, code, message);
    const payload = try payload_writer.toOwnedSlice();
    defer allocator.free(payload);

    try framing.writeMessage(writer, payload);
}

/// Run the stdio transport loop using MCP Content-Length framing.
pub fn run(server: anytype, io: std.Io) !void {
    std.log.info("MCP stdio transport ready ({d} tools registered)", .{server.tools.items.len});

    var stdin_file = std.Io.File.stdin();
    var read_buf: [65536]u8 = undefined;
    var reader = stdin_file.reader(io, &read_buf);

    var stdout_file = std.Io.File.stdout();
    var write_buf: [65536]u8 = undefined;
    var writer = stdout_file.writer(io, &write_buf);

    while (true) {
        const message_opt = framing.readMessageAlloc(server.allocator, &reader.interface, MAX_MESSAGE_SIZE) catch |err| switch (err) {
            error.StreamTooLong, error.PayloadTooLarge => {
                std.log.warn("MCP stdio: rejecting oversized message", .{});
                writeFramedError(
                    server.allocator,
                    &writer.interface,
                    types.ErrorCode.parse_error,
                    "Message too large",
                ) catch |write_err| {
                    std.log.err("MCP stdio: failed to write size error: {t}", .{write_err});
                    break;
                };
                writer.flush() catch |flush_err| {
                    std.log.err("MCP stdio: flush error after size error: {t}", .{flush_err});
                    break;
                };
                continue;
            },
            error.MissingContentLength, error.InvalidContentLength => {
                std.log.warn("MCP stdio: invalid framed message: {t}", .{err});
                writeFramedError(
                    server.allocator,
                    &writer.interface,
                    types.ErrorCode.parse_error,
                    "Parse error",
                ) catch |write_err| {
                    std.log.err("MCP stdio: failed to write parse error: {t}", .{write_err});
                    break;
                };
                writer.flush() catch |flush_err| {
                    std.log.err("MCP stdio: flush error after parse error: {t}", .{flush_err});
                    break;
                };
                continue;
            },
            else => |e| {
                std.log.err("MCP stdio: read error: {t}", .{e});
                break;
            },
        };

        const message = message_opt orelse break;
        defer server.allocator.free(message);

        var response_writer: std.Io.Writer.Allocating = .init(server.allocator);
        const processed = server.processMessage(message, &response_writer.writer);
        if (processed) |_| {
            const response = response_writer.toOwnedSlice() catch |err| {
                response_writer.deinit();
                return err;
            };
            defer server.allocator.free(response);

            if (response.len > 0) {
                framing.writeMessage(&writer.interface, response) catch |err| {
                    std.log.err("MCP stdio: failed to frame response: {t}", .{err});
                    break;
                };
                writer.flush() catch |err| {
                    std.log.err("MCP stdio: flush error, closing connection: {t}", .{err});
                    break;
                };
            }
        } else |err| {
            response_writer.deinit();
            std.log.err("MCP stdio: message handling error: {t}", .{err});
            writeFramedError(
                server.allocator,
                &writer.interface,
                types.ErrorCode.internal_error,
                "Internal error",
            ) catch |write_err| {
                std.log.err("MCP stdio: failed to write internal error response: {t}", .{write_err});
                break;
            };
            writer.flush() catch |flush_err| {
                std.log.err("MCP stdio: flush error after internal error: {t}", .{flush_err});
                break;
            };
        }
    }

    writer.flush() catch |err| {
        std.log.warn("MCP stdio: final flush failed: {t}", .{err});
    };
}

test {
    const testing = std.testing;
    testing.refAllDecls(@This());
}
