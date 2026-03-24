const std = @import("std");
const types = @import("../types.zig");

pub fn run(self: anytype, io: std.Io) !void {
    std.log.info("MCP server ready ({d} tools registered)", .{self.tools.items.len});

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

        self.processMessage(trimmed, &writer.interface) catch |err| {
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

pub fn runInfo(self: anytype) void {
    std.log.info("MCP server ready ({d} tools registered). Use run(io) with I/O backend.", .{self.tools.items.len});
}
