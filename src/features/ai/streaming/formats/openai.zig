//! OpenAI API Format Compatibility
//!
//! Provides request/response formatting compatible with OpenAI's Chat Completions API.
//! This allows clients using OpenAI SDKs to work with ABI's streaming API.
//!
//! Reference: https://platform.openai.com/docs/api-reference/chat

const std = @import("std");
const time = @import("../../../../services/shared/time.zig");
const sync = @import("../../../../services/shared/sync.zig");
const backends = @import("../backends/mod.zig");

/// Chat message role
pub const Role = enum {
    system,
    user,
    assistant,
    tool,

    pub fn toString(self: Role) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .tool => "tool",
        };
    }

    pub fn fromString(s: []const u8) ?Role {
        const map = std.StaticStringMap(Role).initComptime(.{
            .{ "system", .system },
            .{ "user", .user },
            .{ "assistant", .assistant },
            .{ "tool", .tool },
        });
        return map.get(s);
    }
};

/// Chat message
pub const ChatMessage = struct {
    role: Role,
    content: []const u8,

    pub fn deinit(self: *ChatMessage, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
        self.* = undefined;
    }
};

/// Chat completion request (OpenAI format)
pub const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []ChatMessage,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    stream: bool,
    stop: ?[]const []const u8,
    presence_penalty: f32,
    frequency_penalty: f32,

    pub fn deinit(self: *const ChatCompletionRequest, allocator: std.mem.Allocator) void {
        for (self.messages) |*msg| {
            var m = msg.*;
            m.deinit(allocator);
        }
        allocator.free(self.messages);
        allocator.free(self.model);
        if (self.stop) |stops| {
            for (stops) |s| allocator.free(s);
            allocator.free(stops);
        }
    }

    /// Format messages as a single prompt string
    pub fn formatPrompt(self: *const ChatCompletionRequest, allocator: std.mem.Allocator) ![]u8 {
        var prompt = std.ArrayListUnmanaged(u8).empty;
        errdefer prompt.deinit(allocator);

        for (self.messages) |msg| {
            if (prompt.items.len > 0) {
                try prompt.appendSlice(allocator, "\n\n");
            }

            switch (msg.role) {
                .system => {
                    try prompt.appendSlice(allocator, "System: ");
                    try prompt.appendSlice(allocator, msg.content);
                },
                .user => {
                    try prompt.appendSlice(allocator, "User: ");
                    try prompt.appendSlice(allocator, msg.content);
                },
                .assistant => {
                    try prompt.appendSlice(allocator, "Assistant: ");
                    try prompt.appendSlice(allocator, msg.content);
                },
                .tool => {
                    try prompt.appendSlice(allocator, "Tool: ");
                    try prompt.appendSlice(allocator, msg.content);
                },
            }
        }

        // Add assistant prefix for response
        try prompt.appendSlice(allocator, "\n\nAssistant: ");

        return prompt.toOwnedSlice(allocator);
    }

    /// Convert to backend generation config
    pub fn toGenerationConfig(self: *const ChatCompletionRequest) backends.GenerationConfig {
        return .{
            .max_tokens = self.max_tokens,
            .temperature = self.temperature,
            .top_p = self.top_p,
            .model = self.model,
        };
    }
};

/// Parse chat completion request from JSON
pub fn parseRequest(allocator: std.mem.Allocator, json: []const u8) !ChatCompletionRequest {
    // Extract model
    const model_str = extractString(json, "model") orelse "gpt-4";
    const model = try allocator.dupe(u8, model_str);
    errdefer allocator.free(model);

    // Extract messages array
    var messages = std.ArrayListUnmanaged(ChatMessage).empty;
    errdefer {
        for (messages.items) |*msg| {
            allocator.free(msg.content);
        }
        messages.deinit(allocator);
    }

    // Find messages array
    if (std.mem.indexOf(u8, json, "\"messages\"")) |msg_start| {
        const array_start = std.mem.indexOfPos(u8, json, msg_start, "[") orelse return error.InvalidRequest;
        const array_end = findMatchingBracket(json[array_start..]) orelse return error.InvalidRequest;
        const messages_json = json[array_start .. array_start + array_end + 1];

        // Parse individual messages
        var pos: usize = 1; // Skip opening [
        while (pos < messages_json.len - 1) {
            // Find next object
            const obj_start = std.mem.indexOfPos(u8, messages_json, pos, "{") orelse break;
            const obj_end = findMatchingBrace(messages_json[obj_start..]) orelse break;
            const msg_json = messages_json[obj_start .. obj_start + obj_end + 1];

            // Extract role and content
            const role_str = extractString(msg_json, "role") orelse "user";
            const role = Role.fromString(role_str) orelse .user;
            const content_str = extractString(msg_json, "content") orelse "";
            const content = try allocator.dupe(u8, content_str);

            try messages.append(allocator, .{
                .role = role,
                .content = content,
            });

            pos = obj_start + obj_end + 1;
        }
    }

    // If no messages parsed, create a default user message
    if (messages.items.len == 0) {
        const content = try allocator.dupe(u8, "Hello");
        try messages.append(allocator, .{
            .role = .user,
            .content = content,
        });
    }

    // Extract other parameters
    const max_tokens = extractInt(json, "max_tokens") orelse 1024;
    const temperature = extractFloat(json, "temperature") orelse 0.7;
    const top_p = extractFloat(json, "top_p") orelse 1.0;
    const stream = extractBool(json, "stream") orelse false;
    const presence_penalty = extractFloat(json, "presence_penalty") orelse 0.0;
    const frequency_penalty = extractFloat(json, "frequency_penalty") orelse 0.0;

    return .{
        .model = model,
        .messages = try messages.toOwnedSlice(allocator),
        .max_tokens = @intCast(max_tokens),
        .temperature = @floatCast(temperature),
        .top_p = @floatCast(top_p),
        .stream = stream,
        .stop = null,
        .presence_penalty = @floatCast(presence_penalty),
        .frequency_penalty = @floatCast(frequency_penalty),
    };
}

/// Format a streaming chunk in OpenAI format
pub fn formatStreamChunk(
    allocator: std.mem.Allocator,
    content: []const u8,
    model: []const u8,
    index: u32,
    is_end: bool,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    // Generate a simple ID using timer - Zig 0.16 compatible
    // Since we just need a unique-ish ID, use timer elapsed nanoseconds
    var timer = time.Timer.start() catch return error.OutOfMemory;
    const id: i128 = @intCast(timer.read());

    try json.appendSlice(allocator, "{\"id\":\"chatcmpl-");
    try json.print(allocator, "{d}", .{id});
    try json.appendSlice(allocator, "\",\"object\":\"chat.completion.chunk\",\"created\":");
    try json.print(allocator, "{d}", .{@divFloor(id, 1000)});
    try json.appendSlice(allocator, ",\"model\":\"");
    try json.appendSlice(allocator, model);
    try json.appendSlice(allocator, "\",\"choices\":[{\"index\":");
    try json.print(allocator, "{d}", .{index});
    try json.appendSlice(allocator, ",\"delta\":{");

    if (content.len > 0) {
        try json.appendSlice(allocator, "\"content\":\"");
        // Escape content
        for (content) |c| {
            switch (c) {
                '"' => try json.appendSlice(allocator, "\\\""),
                '\\' => try json.appendSlice(allocator, "\\\\"),
                '\n' => try json.appendSlice(allocator, "\\n"),
                '\r' => try json.appendSlice(allocator, "\\r"),
                '\t' => try json.appendSlice(allocator, "\\t"),
                else => try json.append(allocator, c),
            }
        }
        try json.append(allocator, '"');
    }

    try json.appendSlice(allocator, "},\"finish_reason\":");
    if (is_end) {
        try json.appendSlice(allocator, "\"stop\"");
    } else {
        try json.appendSlice(allocator, "null");
    }
    try json.appendSlice(allocator, "}]}");

    return json.toOwnedSlice(allocator);
}

/// Format a complete (non-streaming) response in OpenAI format
pub fn formatResponse(
    allocator: std.mem.Allocator,
    content: []const u8,
    model: []const u8,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    // Generate a simple ID using timer - Zig 0.16 compatible
    var timer = time.Timer.start() catch return error.OutOfMemory;
    const id: i128 = @intCast(timer.read());
    const prompt_tokens: u32 = @intCast(@max(1, @divFloor(content.len, 4))); // ~4 chars/token estimate
    const completion_tokens: u32 = @intCast(@divFloor(content.len, 4)); // Rough estimate

    try json.appendSlice(allocator, "{\"id\":\"chatcmpl-");
    try json.print(allocator, "{d}", .{id});
    try json.appendSlice(allocator, "\",\"object\":\"chat.completion\",\"created\":");
    try json.print(allocator, "{d}", .{@divFloor(id, 1000)});
    try json.appendSlice(allocator, ",\"model\":\"");
    try json.appendSlice(allocator, model);
    try json.appendSlice(allocator, "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"");

    // Escape content
    for (content) |c| {
        switch (c) {
            '"' => try json.appendSlice(allocator, "\\\""),
            '\\' => try json.appendSlice(allocator, "\\\\"),
            '\n' => try json.appendSlice(allocator, "\\n"),
            '\r' => try json.appendSlice(allocator, "\\r"),
            '\t' => try json.appendSlice(allocator, "\\t"),
            else => try json.append(allocator, c),
        }
    }

    try json.appendSlice(allocator, "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":");
    try json.print(allocator, "{d}", .{prompt_tokens});
    try json.appendSlice(allocator, ",\"completion_tokens\":");
    try json.print(allocator, "{d}", .{completion_tokens});
    try json.appendSlice(allocator, ",\"total_tokens\":");
    try json.print(allocator, "{d}", .{prompt_tokens + completion_tokens});
    try json.appendSlice(allocator, "}}");

    return json.toOwnedSlice(allocator);
}

// Helper functions for JSON parsing

fn extractString(json: []const u8, key: []const u8) ?[]const u8 {
    // Build search key without allocation using a fixed buffer
    var key_buf: [256]u8 = undefined;
    const search_key = std.fmt.bufPrint(&key_buf, "\"{s}\":", .{key}) catch return null;

    const key_pos = std.mem.indexOf(u8, json, search_key) orelse return null;
    var pos = key_pos + search_key.len;

    // Skip whitespace
    while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t' or json[pos] == '\n')) : (pos += 1) {}

    if (pos >= json.len or json[pos] != '"') return null;
    pos += 1;

    const str_start = pos;
    while (pos < json.len and json[pos] != '"') : (pos += 1) {
        if (json[pos] == '\\' and pos + 1 < json.len) pos += 1;
    }

    return json[str_start..pos];
}

fn extractInt(json: []const u8, key: []const u8) ?i64 {
    // Build search key without allocation using a fixed buffer
    var key_buf: [256]u8 = undefined;
    const search_key = std.fmt.bufPrint(&key_buf, "\"{s}\":", .{key}) catch return null;

    const key_pos = std.mem.indexOf(u8, json, search_key) orelse return null;
    var pos = key_pos + search_key.len;

    while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t')) : (pos += 1) {}

    const num_start = pos;
    while (pos < json.len and ((json[pos] >= '0' and json[pos] <= '9') or json[pos] == '-')) : (pos += 1) {}

    if (pos == num_start) return null;
    return std.fmt.parseInt(i64, json[num_start..pos], 10) catch null;
}

fn extractFloat(json: []const u8, key: []const u8) ?f64 {
    // Build search key without allocation using a fixed buffer
    var key_buf: [256]u8 = undefined;
    const search_key = std.fmt.bufPrint(&key_buf, "\"{s}\":", .{key}) catch return null;

    const key_pos = std.mem.indexOf(u8, json, search_key) orelse return null;
    var pos = key_pos + search_key.len;

    while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t')) : (pos += 1) {}

    const num_start = pos;
    while (pos < json.len and ((json[pos] >= '0' and json[pos] <= '9') or json[pos] == '.' or json[pos] == '-')) : (pos += 1) {}

    if (pos == num_start) return null;
    return std.fmt.parseFloat(f64, json[num_start..pos]) catch null;
}

fn extractBool(json: []const u8, key: []const u8) ?bool {
    // Build search key without allocation using a fixed buffer
    var key_buf: [256]u8 = undefined;
    const search_key = std.fmt.bufPrint(&key_buf, "\"{s}\":", .{key}) catch return null;

    const key_pos = std.mem.indexOf(u8, json, search_key) orelse return null;
    var pos = key_pos + search_key.len;

    while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t')) : (pos += 1) {}

    if (pos + 4 <= json.len and std.mem.eql(u8, json[pos..][0..4], "true")) return true;
    if (pos + 5 <= json.len and std.mem.eql(u8, json[pos..][0..5], "false")) return false;
    return null;
}

fn findMatchingBracket(json: []const u8) ?usize {
    if (json.len == 0 or json[0] != '[') return null;
    var depth: usize = 0;
    var in_string = false;

    for (json, 0..) |c, i| {
        if (c == '"' and (i == 0 or json[i - 1] != '\\')) {
            in_string = !in_string;
        } else if (!in_string) {
            if (c == '[') depth += 1;
            if (c == ']') {
                depth -= 1;
                if (depth == 0) return i;
            }
        }
    }
    return null;
}

fn findMatchingBrace(json: []const u8) ?usize {
    if (json.len == 0 or json[0] != '{') return null;
    var depth: usize = 0;
    var in_string = false;

    for (json, 0..) |c, i| {
        if (c == '"' and (i == 0 or json[i - 1] != '\\')) {
            in_string = !in_string;
        } else if (!in_string) {
            if (c == '{') depth += 1;
            if (c == '}') {
                depth -= 1;
                if (depth == 0) return i;
            }
        }
    }
    return null;
}

// Tests
test "parse simple request" {
    const allocator = std.testing.allocator;

    const json =
        \\{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}],"max_tokens":100,"stream":true}
    ;

    var request = try parseRequest(allocator, json);
    defer request.deinit(allocator);

    try std.testing.expectEqualStrings("gpt-4", request.model);
    try std.testing.expectEqual(@as(usize, 1), request.messages.len);
    try std.testing.expectEqual(Role.user, request.messages[0].role);
    try std.testing.expectEqualStrings("Hello", request.messages[0].content);
    try std.testing.expect(request.stream);
}

test "format stream chunk" {
    const allocator = std.testing.allocator;

    const chunk = try formatStreamChunk(allocator, "Hello", "gpt-4", 0, false);
    defer allocator.free(chunk);

    try std.testing.expect(std.mem.indexOf(u8, chunk, "chat.completion.chunk") != null);
    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"content\":\"Hello\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"finish_reason\":null") != null);
}

test "format complete response" {
    const allocator = std.testing.allocator;

    const response = try formatResponse(allocator, "Hello world", "gpt-4");
    defer allocator.free(response);

    try std.testing.expect(std.mem.indexOf(u8, response, "chat.completion") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "Hello world") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"finish_reason\":\"stop\"") != null);
}

test "format prompt from messages" {
    const allocator = std.testing.allocator;

    const messages = try allocator.alloc(ChatMessage, 2);
    messages[0] = .{ .role = .system, .content = try allocator.dupe(u8, "You are helpful") };
    messages[1] = .{ .role = .user, .content = try allocator.dupe(u8, "Hi") };

    const request = ChatCompletionRequest{
        .model = try allocator.dupe(u8, "gpt-4"),
        .messages = messages,
        .max_tokens = 100,
        .temperature = 0.7,
        .top_p = 1.0,
        .stream = false,
        .stop = null,
        .presence_penalty = 0.0,
        .frequency_penalty = 0.0,
    };
    defer request.deinit(allocator);

    const prompt = try request.formatPrompt(allocator);
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "System: You are helpful") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "User: Hi") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Assistant:") != null);
}

test {
    std.testing.refAllDecls(@This());
}
