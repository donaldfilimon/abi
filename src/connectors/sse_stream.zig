const std = @import("std");
const connector = @import("connector.zig");

const ConnectorError = connector.ConnectorError;

/// Callback for SSE streaming chunks. `delta` contains the token text (empty when done).
/// Returns error to abort streaming.
pub const StreamCallback = *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void;

/// A single token delta from an SSE streaming response.
pub const StreamChunk = struct {
    delta: []const u8,
    done: bool,
};

fn processSseEvent(
    event_data: []const u8,
    allocator: std.mem.Allocator,
    on_chunk: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void,
    callback_ctx: *anyopaque,
    accumulated: *std.ArrayListUnmanaged(u8),
) ConnectorError!void {
    const trimmed_event = std.mem.trim(u8, event_data, " \t\n\r");
    if (trimmed_event.len == 0) return;

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed_event, .{}) catch return ConnectorError.InvalidResponse;
    defer parsed.deinit();

    const root = parsed.value.object;

    if (root.get("choices") == null) {
        if (root.get("delta")) |delta_val| {
            if (delta_val == .object) {
                if (delta_val.object.get("text")) |text_val| {
                    if (text_val == .string and text_val.string.len > 0) {
                        try accumulated.appendSlice(allocator, text_val.string);
                        try on_chunk(callback_ctx, .{ .delta = text_val.string, .done = false });
                    }
                }
            }
        }
        return;
    }

    const choices = root.get("choices") orelse return;
    if (choices.array.items.len == 0) return;

    const choice = choices.array.items[0].object;
    const delta = choice.get("delta") orelse return;
    const content = delta.object.get("content") orelse return;
    if (content != .string) return;
    const finish_reason = choice.get("finish_reason");

    const token = content.string;
    const done = if (finish_reason) |fr| fr != .null else false;

    if (token.len > 0) {
        try accumulated.appendSlice(allocator, token);
        try on_chunk(callback_ctx, .{ .delta = token, .done = false });
    }
    if (done) {
        try on_chunk(callback_ctx, .{ .delta = "", .done = true });
    }
}

pub fn parseSseStream(
    allocator: std.mem.Allocator,
    body: []const u8,
    on_chunk: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void,
    callback_ctx: *anyopaque,
) ConnectorError![]const u8 {
    var lines = std.mem.splitScalar(u8, body, '\n');
    var data_buffer = std.ArrayListUnmanaged(u8).empty;
    defer data_buffer.deinit(allocator);
    var accumulated = std.ArrayListUnmanaged(u8).empty;
    errdefer accumulated.deinit(allocator);

    while (lines.next()) |line| {
        const trimmed = if (line.len > 0 and line[line.len - 1] == '\r') line[0 .. line.len - 1] else line;

        if (trimmed.len == 0) {
            if (data_buffer.items.len > 0) {
                const event_data = try data_buffer.toOwnedSlice(allocator);
                defer allocator.free(event_data);
                try processSseEvent(event_data, allocator, on_chunk, callback_ctx, &accumulated);
                data_buffer = std.ArrayListUnmanaged(u8).empty;
            }
            continue;
        }

        if (std.mem.startsWith(u8, trimmed, "data:")) {
            const data = std.mem.trim(u8, trimmed[5..], " \t");
            if (std.mem.eql(u8, data, "[DONE]")) {
                try on_chunk(callback_ctx, .{ .delta = "", .done = true });
                continue;
            }
            try data_buffer.appendSlice(allocator, data);
            try data_buffer.append(allocator, '\n');
        }
    }

    if (data_buffer.items.len > 0) {
        const event_data = try data_buffer.toOwnedSlice(allocator);
        defer allocator.free(event_data);
        try processSseEvent(event_data, allocator, on_chunk, callback_ctx, &accumulated);
    }

    try on_chunk(callback_ctx, .{ .delta = "", .done = true });
    return try accumulated.toOwnedSlice(allocator);
}

pub fn parseSseStreamIncremental(
    allocator: std.mem.Allocator,
    reader: *std.Io.Reader,
    on_chunk: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void,
    callback_ctx: *anyopaque,
) ConnectorError![]const u8 {
    var accumulated = std.ArrayListUnmanaged(u8).empty;
    errdefer accumulated.deinit(allocator);

    var read_buf: [4096]u8 = undefined;
    var line_buf = std.ArrayListUnmanaged(u8).empty;
    defer line_buf.deinit(allocator);
    var data_buffer = std.ArrayListUnmanaged(u8).empty;
    defer data_buffer.deinit(allocator);

    while (true) {
        var n: usize = 0;
        blk: {
            n = try std.Io.Reader.readSliceShort(reader, read_buf[0..]);
            if (n == 0) break :blk;
        }
        if (n == 0) break;

        var i: usize = 0;
        while (i < n) {
            const byte = read_buf[i];
            i += 1;

            if (byte == '\n') {
                const line = std.mem.trim(u8, line_buf.items, "\r");
                if (line.len == 0) {
                    if (data_buffer.items.len > 0) {
                        const event_data = try data_buffer.toOwnedSlice(allocator);
                        defer allocator.free(event_data);
                        try processSseEvent(event_data, allocator, on_chunk, callback_ctx, &accumulated);
                        data_buffer = std.ArrayListUnmanaged(u8).empty;
                    }
                } else if (std.mem.startsWith(u8, line, "data:")) {
                    const data = std.mem.trim(u8, line[5..], " \t");
                    if (std.mem.eql(u8, data, "[DONE]")) {
                        try on_chunk(callback_ctx, .{ .delta = "", .done = true });
                    } else {
                        try data_buffer.appendSlice(allocator, data);
                        try data_buffer.append(allocator, '\n');
                    }
                }
                line_buf = std.ArrayListUnmanaged(u8).empty;
            } else {
                try line_buf.append(allocator, byte);
            }
        }
    }

    if (data_buffer.items.len > 0) {
        const event_data = try data_buffer.toOwnedSlice(allocator);
        defer allocator.free(event_data);
        try processSseEvent(event_data, allocator, on_chunk, callback_ctx, &accumulated);
    }

    try on_chunk(callback_ctx, .{ .delta = "", .done = true });
    return try accumulated.toOwnedSlice(allocator);
}

test "parseSseEvent parses token delta" {
    const allocator = std.testing.allocator;
    var context = struct {
        called: bool = false,
        last_len: usize = 0,
        last_done: bool = false,
    }{};
    var accumulated: std.ArrayListUnmanaged(u8) = .empty;
    defer accumulated.deinit(allocator);

    const callback: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void = struct {
        fn call(ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void {
            const self: *struct { called: bool, last_len: usize, last_done: bool } = @ptrCast(@alignCast(ctx));
            self.called = true;
            self.last_len = chunk.delta.len;
            self.last_done = chunk.done;
        }
    }.call;

    try processSseEvent(
        \\{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
    , allocator, callback, &context, &accumulated);

    try std.testing.expect(context.called);
    try std.testing.expectEqual(@as(usize, 5), context.last_len);
    try std.testing.expect(!context.last_done);
    try std.testing.expectEqualStrings("Hello", accumulated.items);
}

test "parseSseEvent handles finish_reason" {
    const allocator = std.testing.allocator;
    var context = struct {
        called: bool = false,
        last_done: bool = false,
    }{};
    var accumulated: std.ArrayListUnmanaged(u8) = .empty;
    defer accumulated.deinit(allocator);

    const callback: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void = struct {
        fn call(ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void {
            const self: *struct { called: bool, last_done: bool } = @ptrCast(@alignCast(ctx));
            self.called = true;
            self.last_done = chunk.done;
        }
    }.call;

    try processSseEvent(
        \\{"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}
    , allocator, callback, &context, &accumulated);

    try std.testing.expect(context.called);
    try std.testing.expect(context.last_done);
}

test "parseSseStream accumulates multi-token SSE and forwards callback_ctx" {
    const allocator = std.testing.allocator;
    var tokens = std.ArrayListUnmanaged(u8).empty;
    defer tokens.deinit(allocator);

    const callback: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void = struct {
        fn call(ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void {
            const acc: *std.ArrayListUnmanaged(u8) = @ptrCast(@alignCast(ctx));
            if (chunk.delta.len > 0) {
                acc.appendSlice(std.testing.allocator, chunk.delta) catch return ConnectorError.InvalidResponse;
            }
        }
    }.call;

    const body =
        \\data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}
        \\
        \\data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}
        \\
        \\data: [DONE]
        \\
    ;
    const full = try parseSseStream(allocator, body, callback, &tokens);
    defer allocator.free(full);
    try std.testing.expectEqualStrings("Hello", full);
    try std.testing.expectEqualStrings("Hello", tokens.items);
}

test "parseSseEvent parses Anthropic text delta" {
    const allocator = std.testing.allocator;
    var context = struct {
        called: bool = false,
        last_len: usize = 0,
    }{};
    const callback: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void = struct {
        fn call(ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void {
            const self: *@TypeOf(context) = @ptrCast(@alignCast(ctx));
            self.called = true;
            self.last_len = chunk.delta.len;
        }
    }.call;
    var accumulated = std.ArrayListUnmanaged(u8).empty;
    defer accumulated.deinit(allocator);
    try processSseEvent(
        "{\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}",
        allocator,
        callback,
        &context,
        &accumulated,
    );
    try std.testing.expect(context.called);
    try std.testing.expectEqual(@as(usize, 2), context.last_len);
    try std.testing.expectEqualStrings("Hi", accumulated.items);
}

test "parseSseStream accumulates Anthropic multi-token SSE" {
    const allocator = std.testing.allocator;
    var tokens = std.ArrayListUnmanaged(u8).empty;
    defer tokens.deinit(allocator);
    const callback: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void = struct {
        fn call(ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void {
            if (chunk.delta.len == 0) return;
            const list: *std.ArrayListUnmanaged(u8) = @ptrCast(@alignCast(ctx));
            try list.appendSlice(std.testing.allocator, chunk.delta);
        }
    }.call;
    const body =
        \\event: content_block_delta
        \\data: {"delta":{"text":"Hel"}}
        \\
        \\event: content_block_delta
        \\data: {"delta":{"text":"lo"}}
        \\
        \\event: message_stop
        \\data: {"type":"message_stop"}
        \\
    ;
    const full = try parseSseStream(allocator, body, callback, &tokens);
    defer allocator.free(full);
    try std.testing.expectEqualStrings("Hello", full);
    try std.testing.expectEqualStrings("Hello", tokens.items);
}

test {
    std.testing.refAllDecls(@This());
}
