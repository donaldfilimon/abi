//! xAI Grok API connector.
//!
//! Provides integration with xAI's Grok models via their Chat Completions API.
//! Supports both synchronous and streaming chat completions.
//!
//! ## Environment Variables
//!
//! - `ABI_GROK_API_KEY` or `GROK_API_KEY`: API key (required)
//! - `ABI_GROK_BASE_URL` or `GROK_BASE_URL`: Base URL (default: https://api.x.ai/v1)
//! - `ABI_GROK_MODEL` or `GROK_MODEL`: Default model (default: grok-2)
//!
//! ## Example
//!
//! ```zig
//! const grok = @import("abi").connectors.grok;
//!
//! var client = try grok.createClient(allocator);
//! defer client.deinit();
//!
//! const response = try client.chatSimple("Hello, Grok!");
//! ```

const std = @import("std");
const shared = @import("shared.zig");
const async_http = @import("../foundation/mod.zig").utils.async_http;
const json_utils = @import("../foundation/mod.zig").utils.json;

/// Errors that can occur when interacting with the xAI API.
pub const GrokError = shared.ProviderError || error{
    MissingApiKey,
};

pub const Config = struct {
    api_key: []u8,
    base_url: []u8,
    model: []const u8 = "grok-2",
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
    messages: []const Message,
    temperature: f32 = 0.7,
    top_p: f32 = 0.95,
    max_tokens: ?u32 = null,
    stream: bool = false,
};

pub const Usage = struct {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
};

pub const MessageDelta = struct {
    role: ?[]const u8 = null,
    content: ?[]const u8 = null,
};

pub const Choice = struct {
    index: u32,
    message: Message,
    finish_reason: ?[]const u8 = null,
};

pub const StreamingChoice = struct {
    index: u32,
    delta: MessageDelta,
    finish_reason: ?[]const u8 = null,
};

pub const ChatCompletionResponse = struct {
    id: []const u8,
    object: []const u8,
    created: u64,
    model: []const u8,
    choices: []Choice,
    usage: Usage,
};

pub const StreamingChunk = struct {
    id: []const u8,
    object: []const u8,
    created: u64,
    model: []const u8,
    choices: []StreamingChoice,
    delta: ?MessageDelta = null,
};

pub const Client = struct {
    allocator: std.mem.Allocator,
    config: Config,
    http: async_http.AsyncHttpClient,

    pub fn init(allocator: std.mem.Allocator, config: Config) !Client {
        const http = try async_http.AsyncHttpClient.init(allocator);
        errdefer http.deinit();
        return .{ .allocator = allocator, .config = config, .http = http };
    }

    pub fn deinit(self: *Client) void {
        self.http.deinit();
        self.config.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn chatSimple(self: *Client, prompt: []const u8) ![]const u8 {
        const messages = [_]Message{
            .{ .role = shared.Role.USER, .content = prompt },
        };
        const request = ChatCompletionRequest{
            .model = self.config.model,
            .messages = messages[0..],
        };
        const response = try self.chatComplete(request);
        if (response.choices.len == 0) return error.InvalidResponse;
        return response.choices[0].message.content;
    }

    pub fn chatComplete(self: *Client, request: ChatCompletionRequest) !ChatCompletionResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/chat/completions", .{self.config.base_url});
        defer self.allocator.free(url);

        const json = try self.encodeChatRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .post, url);
        defer http_req.deinit();

        try http_req.setBearerToken(self.config.api_key);
        try http_req.setJsonBody(json);

        var http_res = try self.http.fetchJsonWithRetry(&http_req, shared.DEFAULT_RETRY_OPTIONS);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            if (http_res.status_code == 429) {
                return GrokError.RateLimitExceeded;
            }
            return GrokError.ApiRequestFailed;
        }

        return try decodeChatResponse(self, http_res.body);
    }

    fn encodeChatRequest(self: *Client, request: ChatCompletionRequest) ![]u8 {
        var json = std.ArrayListUnmanaged(u8).empty;
        errdefer json.deinit(self.allocator);

        try json.appendSlice(self.allocator, "{\"model\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &json, request.model);
        try json.appendSlice(self.allocator, "\",\"messages\":[");
        for (request.messages, 0..) |msg, i| {
            if (i > 0) try json.append(self.allocator, ',');
            try json.appendSlice(self.allocator, "{\"role\":\"");
            try json_utils.appendJsonEscaped(self.allocator, &json, msg.role);
            try json.appendSlice(self.allocator, "\",\"content\":\"");
            try json_utils.appendJsonEscaped(self.allocator, &json, msg.content);
            try json.appendSlice(self.allocator, "\"}");
        }
        try json.print(self.allocator, "],\"temperature\":{d:.2}", .{request.temperature});
        if (request.max_tokens) |max| {
            try json.print(self.allocator, ",\"max_tokens\":{d}", .{max});
        }
        if (request.stream) {
            try json.appendSlice(self.allocator, ",\"stream\":true");
        }
        try json.append(self.allocator, '}');
        return json.toOwnedSlice(self.allocator);
    }

    fn decodeChatResponse(self: *Client, json: []const u8) !ChatCompletionResponse {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json, .{
            .ignore_unknown_fields = true,
        });
        defer parsed.deinit();

        const obj = try json_utils.getRequiredObject(parsed.value);
        const id = try json_utils.parseStringField(obj, "id", self.allocator);
        errdefer self.allocator.free(id);
        const object = try json_utils.parseStringField(obj, "object", self.allocator);
        errdefer self.allocator.free(object);
        const created = try json_utils.parseUintField(obj, "created");
        const model = try json_utils.parseStringField(obj, "model", self.allocator);
        errdefer self.allocator.free(model);

        const choices_array = try json_utils.getRequiredArray(try json_utils.getRequiredField(obj, "choices"));
        var choices = try self.allocator.alloc(Choice, choices_array.items.len);
        var choices_filled: usize = 0;
        errdefer {
            for (choices[0..choices_filled]) |choice| {
                self.allocator.free(choice.message.role);
                self.allocator.free(choice.message.content);
                if (choice.finish_reason) |reason| self.allocator.free(reason);
            }
            self.allocator.free(choices);
        }

        for (choices_array.items, 0..) |item, i| {
            const choice_obj = try json_utils.getRequiredObject(item);
            const message_obj = try json_utils.getRequiredObject(try json_utils.getRequiredField(choice_obj, "message"));
            choices[i] = .{
                .index = @intCast(try json_utils.parseIntField(choice_obj, "index")),
                .message = .{
                    .role = try json_utils.parseStringField(message_obj, "role", self.allocator),
                    .content = try json_utils.parseStringField(message_obj, "content", self.allocator),
                },
                .finish_reason = try json_utils.parseOptionalStringField(choice_obj, "finish_reason", self.allocator),
            };
            choices_filled += 1;
        }

        const usage_obj = try json_utils.getRequiredObject(try json_utils.getRequiredField(obj, "usage"));
        const usage = Usage{
            .prompt_tokens = @intCast(try json_utils.parseIntField(usage_obj, "prompt_tokens")),
            .completion_tokens = @intCast(try json_utils.parseIntField(usage_obj, "completion_tokens")),
            .total_tokens = @intCast(try json_utils.parseIntField(usage_obj, "total_tokens")),
        };

        return .{
            .id = id,
            .object = object,
            .created = created,
            .model = model,
            .choices = choices,
            .usage = usage,
        };
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const loaded = try shared.loadConfigFromEnv(allocator, .{
        .api_key_env = &.{ "ABI_GROK_API_KEY", "GROK_API_KEY" },
        .base_url_env = &.{ "ABI_GROK_BASE_URL", "GROK_BASE_URL" },
        .model_env = &.{ "ABI_GROK_MODEL", "GROK_MODEL" },
        .default_base_url = "https://api.x.ai/v1",
        .default_model = "grok-2",
        .api_key_required = true,
    }, GrokError.MissingApiKey);

    return .{
        .api_key = loaded.api_key.?,
        .base_url = loaded.base_url,
        .model = loaded.model,
        .model_owned = true,
        .timeout_ms = 60_000,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}

pub fn isAvailable() bool {
    return shared.anyEnvIsSet(&.{
        "ABI_GROK_API_KEY",
        "GROK_API_KEY",
    });
}

test {
    std.testing.refAllDecls(@This());
}
