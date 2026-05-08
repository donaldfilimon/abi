const std = @import("std");
const types = @import("types.zig");

/// Server-Sent Events (SSE) formatter for streaming.
pub const SSEFormatter = struct {
    /// Format a token event as SSE data.
    pub fn formatTokenEvent(allocator: std.mem.Allocator, event: types.TokenEvent) ![]u8 {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(allocator);

        // SSE format: data: {json}\n\n
        try buffer.appendSlice(allocator, "data: {\"token_id\":");
        const id_str = try std.fmt.allocPrint(allocator, "{d}", .{event.token_id});
        defer allocator.free(id_str);
        try buffer.appendSlice(allocator, id_str);

        if (event.text) |text| {
            try buffer.appendSlice(allocator, ",\"text\":\"");
            // Escape JSON string
            for (text) |c| {
                switch (c) {
                    '"' => try buffer.appendSlice(allocator, "\\\""),
                    '\\' => try buffer.appendSlice(allocator, "\\\\"),
                    '\n' => try buffer.appendSlice(allocator, "\\n"),
                    '\r' => try buffer.appendSlice(allocator, "\\r"),
                    '\t' => try buffer.appendSlice(allocator, "\\t"),
                    else => try buffer.append(allocator, c),
                }
            }
            try buffer.appendSlice(allocator, "\"");
        }

        try buffer.appendSlice(allocator, ",\"position\":");
        const pos_str = try std.fmt.allocPrint(allocator, "{d}", .{event.position});
        defer allocator.free(pos_str);
        try buffer.appendSlice(allocator, pos_str);

        try buffer.appendSlice(allocator, ",\"is_final\":");
        try buffer.appendSlice(allocator, if (event.is_final) "true" else "false");

        try buffer.appendSlice(allocator, "}\n\n");

        return buffer.toOwnedSlice(allocator);
    }

    /// Format completion event as SSE.
    pub fn formatCompletion(allocator: std.mem.Allocator, stats: types.StreamingStats) ![]u8 {
        return std.fmt.allocPrint(allocator,
            \\data: {{"event":"complete","tokens_generated":{d},"tokens_per_second":{d:.1},"time_to_first_token_ms":{d:.1}}}
            \\
            \\
        , .{
            stats.tokens_generated,
            stats.tokensPerSecond(),
            stats.timeToFirstTokenMs(),
        });
    }

    /// Format error event as SSE.
    pub fn formatErrorEvent(allocator: std.mem.Allocator, err: types.StreamingError) ![]u8 {
        return std.fmt.allocPrint(allocator,
            \\data: {{"event":"error","error":"{t}"}}
            \\
            \\
        , .{err});
    }
};
