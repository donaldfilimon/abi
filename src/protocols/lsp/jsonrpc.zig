//! JSON-RPC framing helpers for LSP (Content-Length protocol).

const std = @import("std");

pub const HeaderError = error{
    MissingContentLength,
    InvalidContentLength,
    PayloadTooLarge,
};

pub const ReadError = HeaderError || std.Io.Reader.Error || error{StreamTooLong} || std.mem.Allocator.Error;

/// Write a JSON-RPC message using LSP Content-Length framing.
pub fn writeMessage(writer: anytype, payload: []const u8) !void {
    var header_buf: [64]u8 = undefined;
    const header = std.fmt.bufPrint(&header_buf, "Content-Length: {d}\r\n\r\n", .{payload.len}) catch return error.InvalidContentLength;
    try writer.writeAll(header);
    try writer.writeAll(payload);
}

/// Read a single JSON-RPC payload from an LSP stream (Content-Length framing).
/// Returns null on EOF.
pub fn readMessageAlloc(
    allocator: std.mem.Allocator,
    reader: *std.Io.Reader,
    max_bytes: usize,
) ReadError!?[]u8 {
    var content_len: ?usize = null;

    while (true) {
        const line_opt = reader.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => return error.StreamTooLong,
            error.ReadFailed => return error.ReadFailed,
        };
        if (line_opt == null) return null; // EOF

        const line = std.mem.trim(u8, line_opt.?, " \t\r\n");
        if (line.len == 0) break; // End of headers

        if (std.ascii.startsWithIgnoreCase(line, "Content-Length:")) {
            const rest = std.mem.trim(u8, line["Content-Length:".len..], " \t");
            const parsed = std.fmt.parseInt(usize, rest, 10) catch return error.InvalidContentLength;
            content_len = parsed;
        }
    }

    const len = content_len orelse return error.MissingContentLength;
    if (len > max_bytes) return error.PayloadTooLarge;

    const data = reader.readAlloc(allocator, len) catch |err| switch (err) {
        error.EndOfStream => return null,
        else => |e| return e,
    };
    return data;
}

test "writeMessage and readMessageAlloc" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const payload = "{\"jsonrpc\":\"2.0\",\"id\":1}";

    var buf: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeMessage(&writer, payload);

    var reader = std.Io.Reader.fixed(writer.buffered());
    const msg = try readMessageAlloc(arena.allocator(), &reader, 1024);
    try std.testing.expect(msg != null);
    try std.testing.expectEqualStrings(payload, msg.?);
}

test {
    std.testing.refAllDecls(@This());
}
