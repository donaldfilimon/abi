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

// Intentionally duplicated from `foundation/json.zig`: connectors are also
// compiled as an isolated module root in connector tests, so importing the
// foundation module here would violate that build boundary. Keep the escape
// table synchronized with `foundation.json.appendJsonString`.
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

pub fn buildDiscordIdentifyBody(allocator: std.mem.Allocator, token: []const u8, intents: u32) ConnectorError![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"op\":2,\"d\":{\"token\":");
    try appendJsonString(&out, allocator, token);
    try out.print(allocator, ",\"intents\":{d},\"properties\":{{\"os\":\"abi\",\"browser\":\"abi\",\"device\":\"abi\"}}}}}}", .{intents});
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

test "appendJsonString escapes quotes, backslashes, and common control characters" {
    const a = std.testing.allocator;
    var out: std.ArrayListUnmanaged(u8) = .empty;
    defer out.deinit(a);

    try appendJsonString(&out, a, "line1\nline2\ttab\r\"quoted\"\\backslash");
    try std.testing.expectEqualStrings(
        "\"line1\\nline2\\ttab\\r\\\"quoted\\\"\\\\backslash\"",
        out.items,
    );
}

test "appendJsonString escapes backspace, form feed, and low control bytes as \\u escapes" {
    const a = std.testing.allocator;
    var out: std.ArrayListUnmanaged(u8) = .empty;
    defer out.deinit(a);

    try appendJsonString(&out, a, &.{ 0x08, 0x0c, 0x01, 0x1f });
    try std.testing.expectEqualStrings("\"\\b\\f\\u0001\\u001F\"", out.items);
}

test "appendJsonString passes through plain ASCII unchanged" {
    const a = std.testing.allocator;
    var out: std.ArrayListUnmanaged(u8) = .empty;
    defer out.deinit(a);

    try appendJsonString(&out, a, "hello world 123");
    try std.testing.expectEqualStrings("\"hello world 123\"", out.items);
}

test "validateJsonValue enforces the expected top-level kind" {
    const a = std.testing.allocator;
    try validateJsonValue(a, "[1,2,3]", .array);
    try validateJsonValue(a, "{\"k\":1}", .object);
    try validateJsonValue(a, "\"anything\"", .any);

    try std.testing.expectError(ConnectorError.InvalidResponse, validateJsonValue(a, "{\"k\":1}", .array));
    try std.testing.expectError(ConnectorError.InvalidResponse, validateJsonValue(a, "[1]", .object));
    try std.testing.expectError(ConnectorError.InvalidResponse, validateJsonValue(a, "not json", .any));
}

test "buildOpenAiBody requires a JSON array for messages and embeds model/stream" {
    const a = std.testing.allocator;

    try std.testing.expectError(ConnectorError.InvalidResponse, buildOpenAiBody(a, "gpt-4", "{\"not\":\"array\"}", false));

    const body = try buildOpenAiBody(a, "gpt-4", "[{\"role\":\"user\",\"content\":\"hi\"}]", false);
    defer a.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"model\":\"gpt-4\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"stream\"") == null);
    try validateJsonValue(a, body, .object);

    const streamed = try buildOpenAiBody(a, "gpt-4", "[]", true);
    defer a.free(streamed);
    try std.testing.expect(std.mem.indexOf(u8, streamed, "\"stream\":true") != null);
}

test "buildAnthropicBody embeds model, max_tokens, and the user message" {
    const a = std.testing.allocator;

    const body = try buildAnthropicBody(a, "fable-5", "hello \"claude\"", 512, false);
    defer a.free(body);
    try validateJsonValue(a, body, .object);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"model\":\"fable-5\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"max_tokens\":512") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\\\"claude\\\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"stream\"") == null);

    const streamed = try buildAnthropicBody(a, "fable-5", "hi", 128, true);
    defer a.free(streamed);
    try std.testing.expect(std.mem.indexOf(u8, streamed, "\"stream\":true") != null);
}

test "buildDiscordMessageBody and buildDiscordIdentifyBody produce parseable JSON with expected fields" {
    const a = std.testing.allocator;

    const msg_body = try buildDiscordMessageBody(a, "hello discord");
    defer a.free(msg_body);
    try validateJsonValue(a, msg_body, .object);
    try std.testing.expect(std.mem.indexOf(u8, msg_body, "\"content\":\"hello discord\"") != null);

    const identify_body = try buildDiscordIdentifyBody(a, "tok123", 513);
    defer a.free(identify_body);
    try validateJsonValue(a, identify_body, .object);
    try std.testing.expect(std.mem.indexOf(u8, identify_body, "\"op\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, identify_body, "\"token\":\"tok123\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, identify_body, "\"intents\":513") != null);
}

test "openAiLocalResponse and openAiLocalStream embed model/byte-count metadata as parseable JSON" {
    const a = std.testing.allocator;

    const resp = try openAiLocalResponse(a, "gpt-4", "[1,2,3]", 42);
    defer a.free(resp);
    try validateJsonValue(a, resp, .object);
    try std.testing.expect(std.mem.indexOf(u8, resp, "model=gpt-4") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "messages_bytes=7") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "request_bytes=42") != null);

    const stream = try openAiLocalStream(a, "gpt-4", "[1,2,3]", 42);
    defer a.free(stream);
    try std.testing.expect(std.mem.startsWith(u8, stream, "data: "));
    try std.testing.expect(std.mem.endsWith(u8, stream, "data: [DONE]\n\n"));
}

test "anthropicLocalResponse and anthropicLocalStream embed model/prompt/max_tokens metadata" {
    const a = std.testing.allocator;

    const resp = try anthropicLocalResponse(a, "fable-5", "hi there", 256, 99);
    defer a.free(resp);
    try validateJsonValue(a, resp, .object);
    try std.testing.expect(std.mem.indexOf(u8, resp, "model=fable-5") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "prompt_bytes=8") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "max_tokens=256") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "request_bytes=99") != null);

    const stream = try anthropicLocalStream(a, "fable-5", "hi there", 256, 99);
    defer a.free(stream);
    try std.testing.expect(std.mem.startsWith(u8, stream, "event: content_block_delta\ndata: "));
    try std.testing.expect(std.mem.endsWith(u8, stream, "event: message_stop\n\n"));
}

test "discordLocalAck reports channel id and content byte length as parseable JSON" {
    const a = std.testing.allocator;
    const ack = try discordLocalAck(a, "chan-42", "hello");
    defer a.free(ack);
    try validateJsonValue(a, ack, .object);
    try std.testing.expect(std.mem.indexOf(u8, ack, "\"status\":\"queued-local\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, ack, "\"channel_id\":\"chan-42\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, ack, "\"content_bytes\":5") != null);
}

test {
    std.testing.refAllDecls(@This());
}
