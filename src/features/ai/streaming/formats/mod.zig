//! Response Formats for Streaming API
//!
//! Provides format converters for different API standards:
//! - OpenAI API format (chat completions)
//! - Custom ABI format

const std = @import("std");
pub const openai = @import("openai.zig");

/// Format type enumeration
pub const FormatType = enum {
    openai,
    abi,

    pub fn fromString(s: []const u8) ?FormatType {
        const map = std.StaticStringMap(FormatType).initComptime(.{
            .{ "openai", .openai },
            .{ "abi", .abi },
        });
        return map.get(s);
    }
};

/// Generic streaming chunk
pub const StreamChunk = struct {
    text: []const u8,
    index: usize,
    is_end: bool,
    model: ?[]const u8,
    finish_reason: ?[]const u8,
};

/// Format a chunk based on format type
pub fn formatChunk(
    allocator: std.mem.Allocator,
    chunk: StreamChunk,
    format_type: FormatType,
) ![]u8 {
    return switch (format_type) {
        .openai => try openai.formatStreamChunk(
            allocator,
            chunk.text,
            chunk.model orelse "unknown",
            @intCast(chunk.index),
            chunk.is_end,
        ),
        .abi => try formatAbiChunk(allocator, chunk),
    };
}

/// Format a chunk in ABI format
fn formatAbiChunk(allocator: std.mem.Allocator, chunk: StreamChunk) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{\"text\":\"");

    // Escape text
    for (chunk.text) |c| {
        switch (c) {
            '"' => try json.appendSlice(allocator, "\\\""),
            '\\' => try json.appendSlice(allocator, "\\\\"),
            '\n' => try json.appendSlice(allocator, "\\n"),
            '\r' => try json.appendSlice(allocator, "\\r"),
            '\t' => try json.appendSlice(allocator, "\\t"),
            else => try json.append(allocator, c),
        }
    }

    try json.appendSlice(allocator, "\",\"index\":");
    try json.print(allocator, "{d}", .{chunk.index});

    if (chunk.is_end) {
        try json.appendSlice(allocator, ",\"done\":true");
    }

    if (chunk.finish_reason) |reason| {
        try json.appendSlice(allocator, ",\"finish_reason\":\"");
        try json.appendSlice(allocator, reason);
        try json.append(allocator, '"');
    }

    try json.append(allocator, '}');
    return json.toOwnedSlice(allocator);
}

// Tests
test "format type from string" {
    try std.testing.expectEqual(FormatType.openai, FormatType.fromString("openai").?);
    try std.testing.expectEqual(FormatType.abi, FormatType.fromString("abi").?);
    try std.testing.expect(FormatType.fromString("unknown") == null);
}

test "format abi chunk" {
    const allocator = std.testing.allocator;

    const chunk = StreamChunk{
        .text = "Hello",
        .index = 0,
        .is_end = false,
        .model = null,
        .finish_reason = null,
    };

    const formatted = try formatAbiChunk(allocator, chunk);
    defer allocator.free(formatted);

    try std.testing.expect(std.mem.indexOf(u8, formatted, "\"text\":\"Hello\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "\"index\":0") != null);
}

test {
    _ = openai;
}
