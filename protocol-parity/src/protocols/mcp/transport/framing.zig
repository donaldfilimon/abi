//! MCP stdio framing helpers.
//!
//! MCP stdio uses the same `Content-Length` framing style as LSP.

const std = @import("std");

pub const HeaderError = error{
    MissingContentLength,
    InvalidContentLength,
    PayloadTooLarge,
};

pub const ReadError = HeaderError || std.Io.Reader.Error || error{StreamTooLong} || std.mem.Allocator.Error;

/// Write a JSON-RPC payload using MCP/LSP Content-Length framing.
pub fn writeMessage(writer: anytype, payload: []const u8) !void {
    var header_buf: [64]u8 = undefined;
    const header = std.fmt.bufPrint(
        &header_buf,
        "Content-Length: {d}\r\n\r\n",
        .{payload.len},
    ) catch return error.InvalidContentLength;
    try writer.writeAll(header);
    try writer.writeAll(payload);
}

/// Read one framed JSON-RPC payload.
/// Returns null on EOF.
pub fn readMessageAlloc(
    allocator: std.mem.Allocator,
    reader: *std.Io.Reader,
    max_bytes: usize,
) ReadError!?[]u8 {
    const first_line_opt = reader.takeDelimiter('\n') catch |err| switch (err) {
        error.StreamTooLong => return error.StreamTooLong,
        error.ReadFailed => return error.ReadFailed,
    };
    if (first_line_opt == null) return null;
    return readMessageAllocFromFirstLine(allocator, reader, first_line_opt.?, max_bytes);
}

/// Read a framed payload when the first header line has already been consumed.
pub fn readMessageAllocFromFirstLine(
    allocator: std.mem.Allocator,
    reader: *std.Io.Reader,
    first_line: []const u8,
    max_bytes: usize,
) ReadError!?[]u8 {
    var content_len: ?usize = null;

    const trimmed_first = std.mem.trim(u8, first_line, " \t\r\n");
    if (trimmed_first.len > 0 and std.ascii.startsWithIgnoreCase(trimmed_first, "Content-Length:")) {
        const rest = std.mem.trim(u8, trimmed_first["Content-Length:".len..], " \t");
        const parsed = std.fmt.parseInt(usize, rest, 10) catch return error.InvalidContentLength;
        content_len = parsed;
    }

    while (true) {
        const line_opt = reader.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => return error.StreamTooLong,
            error.ReadFailed => return error.ReadFailed,
        };
        if (line_opt == null) return null;

        const line = std.mem.trim(u8, line_opt.?, " \t\r\n");
        if (line.len == 0) break;

        if (content_len == null and std.ascii.startsWithIgnoreCase(line, "Content-Length:")) {
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

test "readMessageAllocFromFirstLine accepts extra headers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const payload = "{\"jsonrpc\":\"2.0\",\"id\":1}";
    var input: [256]u8 = undefined;
    var stream = std.Io.Writer.fixed(&input);
    try stream.writeAll("Content-Type: application/json\r\n");
    try stream.print("Content-Length: {d}\r\n\r\n", .{payload.len});
    try stream.writeAll(payload);

    var reader = std.Io.Reader.fixed(stream.buffered());
    const msg = try readMessageAllocFromFirstLine(arena.allocator(), &reader, "Content-Type: application/json", 1024);
    try std.testing.expect(msg != null);
    try std.testing.expectEqualStrings(payload, msg.?);
}

test {
    std.testing.refAllDecls(@This());
}
