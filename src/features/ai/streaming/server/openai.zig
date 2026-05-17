const std = @import("std");
const time = @import("../../../../foundation/time.zig");

pub const OpenAIRequest = struct {
    model: []const u8,
    stream: bool = false,
    messages: []Message = &.{},
    max_tokens: ?u32 = null,

    pub const Message = struct {
        role: []const u8,
        content: []const u8,
    };
};

pub const OpenAIStreamChunk = struct {
    id: []const u8,
    object: []const u8,
    created: i64,
    model: []const u8,
    choices: []StreamChoice,

    pub const StreamChoice = struct {
        index: u32,
        delta: Delta,
        finish_reason: ?[]const u8 = null,

        pub const Delta = struct {
            role: ?[]const u8 = null,
            content: ?[]const u8 = null,
        };
    };
};

pub fn parseRequest(allocator: std.mem.Allocator, request_body: []const u8) !std.json.Parsed(OpenAIRequest) {
    return try std.json.parseFromSlice(OpenAIRequest, allocator, request_body, .{
        .ignore_unknown_fields = true,
    });
}

pub fn handleOpenAIChatCompletions(
    allocator: std.mem.Allocator,
    request_body: []const u8,
    writer: anytype,
) !void {
    const parsed = try parseRequest(allocator, request_body);
    defer parsed.deinit();
    const req = parsed.value;

    if (req.stream) {
        try streamResponse(allocator, req, writer);
    } else {
        try nonStreamResponse(allocator, req, writer);
    }
}

fn streamResponse(
    allocator: std.mem.Allocator,
    req: OpenAIRequest,
    writer: anytype,
) !void {
    const request_id = "chatcmpl-abi-stream";
    const created = time.unixMs() / 1000;
    const response_content = try buildLocalResponse(allocator, req);
    defer allocator.free(response_content);

    var start: usize = 0;
    while (start < response_content.len) {
        const end = @min(start + 16, response_content.len);
        const chunk_text = response_content[start..end];

        var json_buf = std.ArrayListUnmanaged(u8).empty;
        defer json_buf.deinit(allocator);
        var stream = std.json.Stringify{
            .writer = &json_buf.writer(allocator),
            .options = .{ .whitespace = .minified },
        };
        try stream.beginObject();
        try stream.objectField("id");
        try stream.write(request_id);
        try stream.objectField("object");
        try stream.write("chat.completion.chunk");
        try stream.objectField("created");
        try stream.write(created);
        try stream.objectField("model");
        try stream.write(req.model);
        try stream.objectField("choices");
        try stream.beginArray();
        try stream.beginObject();
        try stream.objectField("index");
        try stream.write(@as(u32, 0));
        try stream.objectField("delta");
        try stream.beginObject();
        try stream.objectField("content");
        try stream.write(chunk_text);
        try stream.endObject();
        try stream.endObject();
        try stream.endArray();
        try stream.endObject();

        try writer.print("data: {s}\n\n", .{json_buf.items});
        start = end;
    }

    var final_buf = std.ArrayListUnmanaged(u8).empty;
    defer final_buf.deinit(allocator);
    var final_stream = std.json.Stringify{
        .writer = &final_buf.writer(allocator),
        .options = .{ .whitespace = .minified },
    };
    try final_stream.beginObject();
    try final_stream.objectField("id");
    try final_stream.write(request_id);
    try final_stream.objectField("object");
    try final_stream.write("chat.completion.chunk");
    try final_stream.objectField("created");
    try final_stream.write(created);
    try final_stream.objectField("model");
    try final_stream.write(req.model);
    try final_stream.objectField("choices");
    try final_stream.beginArray();
    try final_stream.beginObject();
    try final_stream.objectField("index");
    try final_stream.write(@as(u32, 0));
    try final_stream.objectField("delta");
    try final_stream.beginObject();
    try final_stream.endObject();
    try final_stream.objectField("finish_reason");
    try final_stream.write("stop");
    try final_stream.endObject();
    try final_stream.endArray();
    try final_stream.endObject();
    try writer.print("data: {s}\n\n", .{final_buf.items});

    try writer.writeAll("data: [DONE]\n\n");
}

fn nonStreamResponse(
    allocator: std.mem.Allocator,
    req: OpenAIRequest,
    writer: anytype,
) !void {
    const response_content = try buildLocalResponse(allocator, req);
    defer allocator.free(response_content);

    var json_buf = std.ArrayListUnmanaged(u8).empty;
    defer json_buf.deinit(allocator);
    var stream = std.json.Stringify{
        .writer = &json_buf.writer(allocator),
        .options = .{ .whitespace = .minified },
    };
    try stream.beginObject();
    try stream.objectField("id");
    try stream.write("chatcmpl-abi-nonstream");
    try stream.objectField("object");
    try stream.write("chat.completion");
    try stream.objectField("created");
    try stream.write(time.unixMs() / 1000);
    try stream.objectField("model");
    try stream.write(req.model);
    try stream.objectField("choices");
    try stream.beginArray();
    try stream.beginObject();
    try stream.objectField("index");
    try stream.write(@as(u32, 0));
    try stream.objectField("message");
    try stream.beginObject();
    try stream.objectField("role");
    try stream.write("assistant");
    try stream.objectField("content");
    try stream.write(response_content);
    try stream.endObject();
    try stream.objectField("finish_reason");
    try stream.write("stop");
    try stream.endObject();
    try stream.endArray();
    try stream.objectField("usage");
    try stream.beginObject();
    try stream.objectField("prompt_tokens");
    try stream.write(estimatePromptTokens(req));
    try stream.objectField("completion_tokens");
    try stream.write(estimateTokens(response_content));
    try stream.objectField("total_tokens");
    try stream.write(estimatePromptTokens(req) + estimateTokens(response_content));
    try stream.endObject();
    try stream.endObject();

    try writer.writeAll(json_buf.items);
}

fn buildLocalResponse(allocator: std.mem.Allocator, req: OpenAIRequest) ![]u8 {
    return try std.fmt.allocPrint(allocator, "ABI local response: {s}", .{lastUserMessage(req)});
}

fn lastUserMessage(req: OpenAIRequest) []const u8 {
    var selected: []const u8 = "No user message provided";
    for (req.messages) |message| {
        if (std.ascii.eqlIgnoreCase(message.role, "user")) selected = message.content;
    }
    return selected;
}

fn estimatePromptTokens(req: OpenAIRequest) u32 {
    var total: u32 = 0;
    for (req.messages) |message| total += estimateTokens(message.content);
    return total;
}

fn estimateTokens(text: []const u8) u32 {
    var count: u32 = 0;
    var parts = std.mem.splitAny(u8, text, " \t\r\n");
    while (parts.next()) |part| {
        if (part.len > 0) count += 1;
    }
    return count;
}

test {
    std.testing.refAllDecls(@This());
}

test "parseRequest extracts model and stream flag" {
    const allocator = std.testing.allocator;
    const json =
        \\{"model": "gpt-4", "stream": true}
    ;
    const parsed = try parseRequest(allocator, json);
    defer parsed.deinit();
    const req = parsed.value;
    try std.testing.expect(std.mem.eql(u8, req.model, "gpt-4"));
    try std.testing.expect(req.stream == true);
}

test "streamResponse writes SSE format" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    const messages = [_]OpenAIRequest.Message{.{ .role = "user", .content = "stream this" }};
    const req = OpenAIRequest{
        .model = "abi-test",
        .stream = true,
        .messages = @constCast(&messages),
    };

    try streamResponse(allocator, req, buf.writer(allocator));
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "data:") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "stream this") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "[DONE]") != null);
}

test "nonStreamResponse writes JSON" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    const messages = [_]OpenAIRequest.Message{.{ .role = "user", .content = "hello local model" }};
    const req = OpenAIRequest{
        .model = "abi-test",
        .stream = false,
        .messages = @constCast(&messages),
    };

    try nonStreamResponse(allocator, req, buf.writer(allocator));
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"object\":\"chat.completion\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"model\":\"abi-test\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "hello local model") != null);
}
