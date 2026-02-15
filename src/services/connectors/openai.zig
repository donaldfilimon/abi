//! OpenAI API connector.
//!
//! Provides integration with OpenAI's models via their Chat Completions API.
//! Supports both synchronous and streaming chat completions.
//!
//! ## Environment Variables
//!
//! - `ABI_OPENAI_API_KEY` or `OPENAI_API_KEY`: API key (required)
//! - `ABI_OPENAI_BASE_URL` or `OPENAI_BASE_URL`: Base URL (default: https://api.openai.com/v1)
//! - `ABI_OPENAI_MODEL` or `OPENAI_MODEL`: Default model (default: gpt-4)
//!
//! ## Example
//!
//! ```zig
//! const openai = @import("abi").connectors.openai;
//!
//! var client = try openai.createClient(allocator);
//! defer client.deinit();
//!
//! const response = try client.chatSimple("Hello, world!");
//! ```

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const async_http = @import("../shared/utils.zig").async_http;
const json_utils = @import("../shared/utils.zig").json;

/// Errors that can occur when interacting with the OpenAI API.
pub const OpenAIError = error{
    /// API key was not provided via environment variable.
    MissingApiKey,
    /// The API request failed (network error or non-2xx status).
    ApiRequestFailed,
    /// The API response could not be parsed.
    InvalidResponse,
    /// Rate limit exceeded (HTTP 429). Retry after backoff.
    RateLimitExceeded,
};

pub const Config = struct {
    api_key: []u8,
    base_url: []u8,
    model: []const u8 = "gpt-4",
    model_owned: bool = false,
    timeout_ms: u32 = 60_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        shared.deinitConfig(allocator, self.api_key, self.base_url);
        if (self.model_owned) allocator.free(@constCast(self.model));
        self.* = undefined;
    }
};

pub const Message = shared.ChatMessage;

pub const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []Message,
    temperature: f32 = 0.7,
    max_tokens: ?u32 = null,
    stream: bool = false,
};

pub const StreamingChunk = struct {
    id: []const u8,
    object: []const u8,
    created: u64,
    model: []const u8,
    choices: []StreamingChoice,
    delta: ?StreamingDelta = null,
};

pub const StreamingChoice = struct {
    index: u32,
    delta: ?StreamingDelta,
    finish_reason: ?[]const u8 = null,
};

pub const StreamingDelta = struct {
    role: ?[]const u8 = null,
    content: ?[]const u8 = null,
};

pub const ChatCompletionResponse = struct {
    id: []const u8,
    object: []const u8,
    created: u64,
    model: []const u8,
    choices: []Choice,
    usage: Usage,
};

pub const Choice = struct {
    index: u32,
    message: Message,
    finish_reason: []const u8,
};

pub const Usage = struct {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
};

pub const Client = struct {
    allocator: std.mem.Allocator,
    config: Config,
    http: async_http.AsyncHttpClient,

    pub fn init(allocator: std.mem.Allocator, config: Config) !Client {
        const http = try async_http.AsyncHttpClient.init(allocator);
        errdefer http.deinit();

        return .{
            .allocator = allocator,
            .config = config,
            .http = http,
        };
    }

    pub fn deinit(self: *Client) void {
        self.http.deinit();
        self.config.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn chatCompletion(self: *Client, request: ChatCompletionRequest) !ChatCompletionResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/chat/completions", .{self.config.base_url});
        defer self.allocator.free(url);

        const json = try self.encodeChatRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .POST, url);
        defer http_req.deinit();

        try http_req.setBearerToken(self.config.api_key);
        try http_req.setJsonBody(json);

        const http_res = try self.http.fetchJsonWithRetry(&http_req, shared.DEFAULT_RETRY_OPTIONS);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            if (http_res.status_code == 429) {
                return OpenAIError.RateLimitExceeded;
            }
            return OpenAIError.ApiRequestFailed;
        }

        return try self.decodeChatResponse(http_res.body);
    }

    pub fn chat(self: *Client, messages: []Message) !ChatCompletionResponse {
        return try self.chatCompletion(.{
            .model = self.config.model,
            .messages = messages,
        });
    }

    pub fn chatSimple(self: *Client, prompt: []const u8) !ChatCompletionResponse {
        const messages = [_]Message{
            .{ .role = "user", .content = prompt },
        };
        return try self.chat(&messages);
    }

    pub fn chatCompletionStreaming(self: *Client, request: ChatCompletionRequest) !async_http.StreamingResponse {
        var req = request;
        req.stream = true;

        const url = try std.fmt.allocPrint(self.allocator, "{s}/chat/completions", .{self.config.base_url});
        defer self.allocator.free(url);

        const json = try self.encodeChatRequest(req);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .POST, url);
        defer http_req.deinit();

        try http_req.setBearerToken(self.config.api_key);
        try http_req.setJsonBody(json);

        return try self.http.fetchStreaming(&http_req);
    }

    pub fn encodeChatRequest(self: *Client, request: ChatCompletionRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8).empty;
        errdefer json_str.deinit(self.allocator);

        try json_str.appendSlice(self.allocator, "{\"model\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &json_str, request.model);
        try json_str.appendSlice(self.allocator, "\",\"messages\":[");

        try shared.encodeMessageArray(self.allocator, &json_str, request.messages);

        try json_str.print(self.allocator, "],\"temperature\":{d:.2}", .{request.temperature});

        if (request.max_tokens) |max_tokens| {
            try json_str.print(self.allocator, ",\"max_tokens\":{d}", .{max_tokens});
        }

        try json_str.append(self.allocator, '}');

        return json_str.toOwnedSlice(self.allocator);
    }

    pub fn decodeChatResponse(self: *Client, json: []const u8) !ChatCompletionResponse {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        const id = try json_utils.parseStringField(object, "id", self.allocator);
        errdefer self.allocator.free(id);

        const obj = try json_utils.parseStringField(object, "object", self.allocator);
        errdefer self.allocator.free(obj);

        const created = try json_utils.parseUintField(object, "created");

        const model = try json_utils.parseStringField(object, "model", self.allocator);
        errdefer self.allocator.free(model);

        const choices_array = try json_utils.parseArrayField(object, "choices");
        if (choices_array.items.len == 0) {
            return OpenAIError.InvalidResponse;
        }

        var choices = try self.allocator.alloc(Choice, choices_array.items.len);
        errdefer {
            for (choices) |*choice| {
                self.allocator.free(choice.message.role);
                self.allocator.free(choice.message.content);
                self.allocator.free(choice.finish_reason);
            }
            self.allocator.free(choices);
        }

        for (choices_array.items, 0..) |choice_value, i| {
            const choice_obj = try json_utils.getRequiredObject(choice_value);
            const index: u32 = @intCast(try json_utils.parseIntField(choice_obj, "index"));

            const message_obj = try json_utils.parseObjectField(choice_obj, "message");
            const role = try json_utils.parseStringField(message_obj, "role", self.allocator);
            const content = try json_utils.parseStringField(message_obj, "content", self.allocator);

            const finish_reason = try json_utils.parseStringField(choice_obj, "finish_reason", self.allocator);

            choices[i] = .{
                .index = index,
                .message = .{
                    .role = role,
                    .content = content,
                },
                .finish_reason = finish_reason,
            };
        }

        const usage_obj = try json_utils.parseObjectField(object, "usage");
        const usage = Usage{
            .prompt_tokens = @intCast(try json_utils.parseIntField(usage_obj, "prompt_tokens")),
            .completion_tokens = @intCast(try json_utils.parseIntField(usage_obj, "completion_tokens")),
            .total_tokens = @intCast(try json_utils.parseIntField(usage_obj, "total_tokens")),
        };

        return ChatCompletionResponse{
            .id = id,
            .object = obj,
            .created = created,
            .model = model,
            .choices = choices,
            .usage = usage,
        };
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const api_key_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OPENAI_API_KEY",
        "OPENAI_API_KEY",
    });
    const api_key = api_key_raw orelse return OpenAIError.MissingApiKey;
    // Treat empty string as missing (e.g., OPENAI_API_KEY="")
    if (api_key.len == 0) {
        allocator.free(api_key);
        return OpenAIError.MissingApiKey;
    }
    errdefer allocator.free(api_key);

    const base_url_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
    });
    // Treat empty base URL as unset — fall through to default
    const base_url = if (base_url_raw) |u| blk: {
        if (u.len == 0) {
            allocator.free(u);
            break :blk try allocator.dupe(u8, "https://api.openai.com/v1");
        }
        break :blk u;
    } else try allocator.dupe(u8, "https://api.openai.com/v1");
    errdefer allocator.free(base_url);

    const model_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OPENAI_MODEL",
        "OPENAI_MODEL",
    });
    // Treat empty model as unset — fall through to default
    const model = if (model_raw) |m| blk: {
        if (m.len == 0) {
            allocator.free(m);
            break :blk try allocator.dupe(u8, "gpt-4");
        }
        break :blk m;
    } else try allocator.dupe(u8, "gpt-4");

    return .{
        .api_key = api_key,
        .base_url = base_url,
        .model = model,
        .model_owned = true,
        .timeout_ms = 60_000,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}

/// Check if the OpenAI connector is available (API key env var is set).
/// This is a zero-allocation health check suitable for status dashboards.
pub fn isAvailable() bool {
    return shared.anyEnvIsSet(&.{
        "ABI_OPENAI_API_KEY",
        "OPENAI_API_KEY",
    });
}

// ============================================================================
// Tests
// ============================================================================

test "chat completion request default values" {
    var messages = [_]Message{.{
        .role = shared.Role.USER,
        .content = "Hello",
    }};
    const request = ChatCompletionRequest{
        .model = "gpt-4",
        .messages = messages[0..],
    };
    try std.testing.expectEqual(@as(f32, 0.7), request.temperature);
    try std.testing.expectEqual(@as(?u32, null), request.max_tokens);
    try std.testing.expect(!request.stream);
}

test "usage struct" {
    const usage = Usage{
        .prompt_tokens = 10,
        .completion_tokens = 20,
        .total_tokens = 30,
    };
    try std.testing.expectEqual(@as(u32, 10), usage.prompt_tokens);
    try std.testing.expectEqual(@as(u32, 20), usage.completion_tokens);
    try std.testing.expectEqual(@as(u32, 30), usage.total_tokens);
}

test "streaming delta default values" {
    const delta = StreamingDelta{};
    try std.testing.expectEqual(@as(?[]const u8, null), delta.role);
    try std.testing.expectEqual(@as(?[]const u8, null), delta.content);
}

test "streaming choice default finish_reason" {
    const choice = StreamingChoice{
        .index = 0,
        .delta = null,
    };
    try std.testing.expectEqual(@as(?[]const u8, null), choice.finish_reason);
}

test "isAvailable returns bool" {
    // Just verify it returns without crashing - actual availability depends on env
    _ = isAvailable();
}
