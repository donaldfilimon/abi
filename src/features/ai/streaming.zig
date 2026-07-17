const std = @import("std");
const time = @import("../../foundation/time.zig");
const json = @import("../../foundation/json.zig");

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
    const created = @divTrunc(time.unixMs(), 1000);
    const response_content = try buildLocalResponse(allocator, req);
    defer allocator.free(response_content);

    var start: usize = 0;
    while (start < response_content.len) {
        const end = @min(start + 16, response_content.len);
        const chunk_text = response_content[start..end];

        try writer.writeAll("data: {\"id\":\"");
        try writer.writeAll(request_id);
        try writer.writeAll("\",\"object\":\"chat.completion.chunk\",\"created\":");
        const created_str = try std.fmt.allocPrint(allocator, "{d}", .{created});
        defer allocator.free(created_str);
        try writer.writeAll(created_str);
        try writer.writeAll(",\"model\":");
        try json.writeJsonString(writer, req.model);
        try writer.writeAll(",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
        try json.writeJsonString(writer, chunk_text);
        try writer.writeAll("}}]}\n\n");
        start = end;
    }

    try writer.writeAll("data: {\"id\":\"");
    try writer.writeAll(request_id);
    try writer.writeAll("\",\"object\":\"chat.completion.chunk\",\"created\":");
    const final_created_str = try std.fmt.allocPrint(allocator, "{d}", .{created});
    defer allocator.free(final_created_str);
    try writer.writeAll(final_created_str);
    try writer.writeAll(",\"model\":");
    try json.writeJsonString(writer, req.model);
    try writer.writeAll(",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n");
    try writer.writeAll("data: [DONE]\n\n");
}

fn nonStreamResponse(
    allocator: std.mem.Allocator,
    req: OpenAIRequest,
    writer: anytype,
) !void {
    const response_content = try buildLocalResponse(allocator, req);
    defer allocator.free(response_content);

    const created_str = try std.fmt.allocPrint(allocator, "{d}", .{@divTrunc(time.unixMs(), 1000)});
    defer allocator.free(created_str);
    const prompt_tokens_str = try std.fmt.allocPrint(allocator, "{d}", .{estimatePromptTokens(req)});
    defer allocator.free(prompt_tokens_str);
    const completion_tokens_str = try std.fmt.allocPrint(allocator, "{d}", .{estimateTokens(response_content)});
    defer allocator.free(completion_tokens_str);
    const total_tokens_str = try std.fmt.allocPrint(allocator, "{d}", .{estimatePromptTokens(req) + estimateTokens(response_content)});
    defer allocator.free(total_tokens_str);

    try writer.writeAll("{\"id\":\"chatcmpl-abi-nonstream\",\"object\":\"chat.completion\",\"created\":");
    try writer.writeAll(created_str);
    try writer.writeAll(",\"model\":");
    try json.writeJsonString(writer, req.model);
    try writer.writeAll(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    try json.writeJsonString(writer, response_content);
    try writer.writeAll("},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":");
    try writer.writeAll(prompt_tokens_str);
    try writer.writeAll(",\"completion_tokens\":");
    try writer.writeAll(completion_tokens_str);
    try writer.writeAll(",\"total_tokens\":");
    try writer.writeAll(total_tokens_str);
    try writer.writeAll("}}");
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
    const input_json =
        \\{"model": "gpt-4", "stream": true}
    ;
    const parsed = try parseRequest(allocator, input_json);
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

    const BufWriter = struct {
        b: *std.ArrayListUnmanaged(u8),
        a: std.mem.Allocator,
        pub fn writeAll(self: *@This(), data: []const u8) !void {
            try self.b.appendSlice(self.a, data);
        }
    };
    var tw = BufWriter{ .b = &buf, .a = allocator };

    try streamResponse(allocator, req, &tw);
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

    const BufWriter = struct {
        b: *std.ArrayListUnmanaged(u8),
        a: std.mem.Allocator,
        pub fn writeAll(self: *@This(), data: []const u8) !void {
            try self.b.appendSlice(self.a, data);
        }
    };
    var tw = BufWriter{ .b = &buf, .a = allocator };

    try nonStreamResponse(allocator, req, &tw);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"object\":\"chat.completion\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"model\":\"abi-test\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "hello local model") != null);
}

test "responses escape JSON strings" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    const messages = [_]OpenAIRequest.Message{.{ .role = "user", .content = "hello \"local\"\nmodel" }};
    const req = OpenAIRequest{
        .model = "abi\\test\"model",
        .stream = false,
        .messages = @constCast(&messages),
    };

    const BufWriter = struct {
        b: *std.ArrayListUnmanaged(u8),
        a: std.mem.Allocator,
        pub fn writeAll(self: *@This(), data: []const u8) !void {
            try self.b.appendSlice(self.a, data);
        }
    };
    var tw = BufWriter{ .b = &buf, .a = allocator };

    try nonStreamResponse(allocator, req, &tw);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"model\":\"abi\\\\test\\\"model\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "hello \\\"local\\\"\\nmodel") != null);
}
