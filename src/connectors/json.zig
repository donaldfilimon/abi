const std = @import("std");
const connector = @import("connector.zig");

const ConnectorError = connector.ConnectorError;

pub const ExpectedJsonKind = enum { array, object, any };

pub fn validateJsonValue(allocator: std.mem.Allocator, input: []const u8, expected: ExpectedJsonKind) ConnectorError!void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, input, .{}) catch return ConnectorError.InvalidResponse;
    defer parsed.deinit();
    switch (expected) {
        .any => {},
        .array => switch (parsed.value) {
            .array => {},
            else => return ConnectorError.InvalidResponse,
        },
        .object => switch (parsed.value) {
            .object => {},
            else => return ConnectorError.InvalidResponse,
        },
    }
}

pub fn appendJsonString(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try out.append(allocator, '"');
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0x00...0x07 => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            0x08 => try out.appendSlice(allocator, "\\b"),
            0x0c => try out.appendSlice(allocator, "\\f"),
            0x0b => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            0x0e...0x1f => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            else => try out.append(allocator, byte),
        }
    }
    try out.append(allocator, '"');
}

pub fn buildOpenAiBody(allocator: std.mem.Allocator, model: []const u8, messages: []const u8, stream: bool) ConnectorError![]u8 {
    try validateJsonValue(allocator, messages, .array);
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"model\":");
    try appendJsonString(&out, allocator, model);
    try out.appendSlice(allocator, ",\"messages\":");
    try out.appendSlice(allocator, messages);
    if (stream) try out.appendSlice(allocator, ",\"stream\":true");
    try out.append(allocator, '}');
    return try out.toOwnedSlice(allocator);
}

pub fn buildAnthropicBody(allocator: std.mem.Allocator, model: []const u8, prompt: []const u8, max_tokens: u32, stream: bool) ConnectorError![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"model\":");
    try appendJsonString(&out, allocator, model);
    try out.print(allocator, ",\"max_tokens\":{d}", .{max_tokens});
    if (stream) try out.appendSlice(allocator, ",\"stream\":true");
    try out.appendSlice(allocator, ",\"messages\":[{\"role\":\"user\",\"content\":");
    try appendJsonString(&out, allocator, prompt);
    try out.appendSlice(allocator, "}]}");
    return try out.toOwnedSlice(allocator);
}

pub fn buildDiscordMessageBody(allocator: std.mem.Allocator, content: []const u8) ConnectorError![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"content\":");
    try appendJsonString(&out, allocator, content);
    try out.append(allocator, '}');
    return try out.toOwnedSlice(allocator);
}

pub fn openAiLocalResponse(allocator: std.mem.Allocator, model: []const u8, messages: []const u8, body_len: usize) ![]u8 {
    const text = try std.fmt.allocPrint(
        allocator,
        "OpenAI-compatible local response model={s} messages_bytes={d} request_bytes={d}",
        .{ model, messages.len, body_len },
    );
    defer allocator.free(text);
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":");
    try appendJsonString(&out, allocator, text);
    try out.appendSlice(allocator, "}}]}");
    return try out.toOwnedSlice(allocator);
}

pub fn openAiLocalStream(allocator: std.mem.Allocator, model: []const u8, messages: []const u8, body_len: usize) ![]u8 {
    const content = try std.fmt.allocPrint(allocator, "OpenAI-compatible local stream model={s} messages_bytes={d} request_bytes={d}", .{ model, messages.len, body_len });
    defer allocator.free(content);
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "data: ");
    try appendOpenAiDelta(&out, allocator, content);
    try out.appendSlice(allocator, "\n\ndata: [DONE]\n\n");
    return try out.toOwnedSlice(allocator);
}

pub fn appendOpenAiDelta(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, content: []const u8) !void {
    try out.appendSlice(allocator, "{\"choices\":[{\"delta\":{\"content\":");
    try appendJsonString(out, allocator, content);
    try out.appendSlice(allocator, "}}]}");
}

pub fn anthropicLocalResponse(allocator: std.mem.Allocator, model: []const u8, prompt: []const u8, max_tokens: u32, body_len: usize) ![]u8 {
    const text = try std.fmt.allocPrint(
        allocator,
        "Anthropic-compatible local response model={s} prompt_bytes={d} max_tokens={d} request_bytes={d}",
        .{ model, prompt.len, max_tokens, body_len },
    );
    defer allocator.free(text);
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"content\":[{\"type\":\"text\",\"text\":");
    try appendJsonString(&out, allocator, text);
    try out.appendSlice(allocator, "}]}");
    return try out.toOwnedSlice(allocator);
}

pub fn anthropicLocalStream(allocator: std.mem.Allocator, model: []const u8, prompt: []const u8, max_tokens: u32, body_len: usize) ![]u8 {
    const content = try std.fmt.allocPrint(allocator, "Anthropic-compatible local stream model={s} prompt_bytes={d} max_tokens={d} request_bytes={d}", .{ model, prompt.len, max_tokens, body_len });
    defer allocator.free(content);
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "event: content_block_delta\ndata: ");
    try appendAnthropicDelta(&out, allocator, content);
    try out.appendSlice(allocator, "\n\nevent: message_stop\n\n");
    return try out.toOwnedSlice(allocator);
}

pub fn appendAnthropicDelta(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, content: []const u8) !void {
    try out.appendSlice(allocator, "{\"delta\":{\"text\":");
    try appendJsonString(out, allocator, content);
    try out.appendSlice(allocator, "}}");
}

pub fn discordLocalAck(allocator: std.mem.Allocator, channel_id: []const u8, content: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"status\":\"queued-local\",\"channel_id\":");
    try appendJsonString(&out, allocator, channel_id);
    try out.print(allocator, ",\"content_bytes\":{d}}}", .{content.len});
    return try out.toOwnedSlice(allocator);
}

test {
    std.testing.refAllDecls(@This());
}
