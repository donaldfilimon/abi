//! vLLM API connector.
//!
//! Provides integration with vLLM inference servers via the OpenAI-compatible
//! API at `/v1/chat/completions`.
//!
//! ## Environment Variables
//!
//! - `ABI_VLLM_HOST`: Host URL (default: http://localhost:8000)
//! - `ABI_VLLM_MODEL`: Default model name
//! - `ABI_VLLM_API_KEY`: Optional API key
//!
//! ## Example
//!
//! ```zig
//! const vllm = @import("abi").connectors.vllm;
//!
//! var client = try vllm.createClient(allocator);
//! defer client.deinit();
//!
//! const response = try client.chatSimple("Hello, world!");
//! ```

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const async_http = @import("../shared/utils.zig").async_http;
const json_utils = @import("../shared/utils.zig").json;

/// Errors that can occur when interacting with vLLM.
pub const VLLMError = error{
    /// The API request failed (network error or non-2xx status).
    ApiRequestFailed,
    /// The API response could not be parsed.
    InvalidResponse,
    /// Rate limit exceeded (HTTP 429).
    RateLimitExceeded,
};

pub const Config = struct {
    host: []u8,
    api_key: ?[]u8 = null,
    model: []const u8 = "default",
    model_owned: bool = false,
    timeout_ms: u32 = 120_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.host);
        if (self.api_key) |key| {
            shared.secureFree(allocator, key);
        }
        if (self.model_owned) allocator.free(@constCast(self.model));
        self.* = undefined;
    }
};

pub const Message = shared.ChatMessage;

pub const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []const Message,
    temperature: f32 = 0.7,
    max_tokens: ?u32 = null,
    top_p: f32 = 1.0,
    stream: bool = false,
};

pub const ChatCompletionResponse = struct {
    id: []const u8,
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
        const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/chat/completions", .{self.config.host});
        defer self.allocator.free(url);

        const json = try self.encodeChatRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .post, url);
        defer http_req.deinit();

        if (self.config.api_key) |key| {
            try http_req.setBearerToken(key);
        }
        try http_req.setJsonBody(json);

        var http_res = try self.http.fetchJson(&http_req);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            if (http_res.status_code == 429) {
                return VLLMError.RateLimitExceeded;
            }
            return VLLMError.ApiRequestFailed;
        }

        return try self.decodeChatResponse(http_res.body);
    }

    pub fn chat(self: *Client, messages: []const Message) !ChatCompletionResponse {
        return try self.chatCompletion(.{
            .model = self.config.model,
            .messages = messages,
        });
    }

    pub fn chatSimple(self: *Client, prompt: []const u8) !ChatCompletionResponse {
        const msgs = [_]Message{
            .{ .role = "user", .content = prompt },
        };
        return try self.chat(&msgs);
    }

    fn encodeChatRequest(self: *Client, request: ChatCompletionRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8){};
        errdefer json_str.deinit(self.allocator);

        try json_str.appendSlice(self.allocator, "{\"model\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &json_str, request.model);
        try json_str.appendSlice(self.allocator, "\",\"messages\":[");

        try shared.encodeMessageArray(self.allocator, &json_str, request.messages);

        try json_str.print(self.allocator, "],\"temperature\":{d:.2},\"top_p\":{d:.2}", .{
            request.temperature,
            request.top_p,
        });

        if (request.max_tokens) |max_tokens| {
            try json_str.print(self.allocator, ",\"max_tokens\":{d}", .{max_tokens});
        }

        if (request.stream) {
            try json_str.appendSlice(self.allocator, ",\"stream\":true");
        }

        try json_str.append(self.allocator, '}');

        return json_str.toOwnedSlice(self.allocator);
    }

    fn decodeChatResponse(self: *Client, json: []const u8) !ChatCompletionResponse {
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

        const model = try json_utils.parseStringField(object, "model", self.allocator);
        errdefer self.allocator.free(model);

        const choices_array = try json_utils.parseArrayField(object, "choices");
        if (choices_array.items.len == 0) {
            return VLLMError.InvalidResponse;
        }

        var choices = try self.allocator.alloc(Choice, choices_array.items.len);
        errdefer self.allocator.free(choices);

        for (choices_array.items, 0..) |choice_value, i| {
            const choice_obj = try json_utils.getRequiredObject(choice_value);
            const index: u32 = @intCast(try json_utils.parseIntField(choice_obj, "index"));

            const message_obj = try json_utils.parseObjectField(choice_obj, "message");
            const role = try json_utils.parseStringField(message_obj, "role", self.allocator);
            const content = try json_utils.parseStringField(message_obj, "content", self.allocator);

            const finish_reason = try json_utils.parseStringField(choice_obj, "finish_reason", self.allocator);

            choices[i] = .{
                .index = index,
                .message = .{ .role = role, .content = content },
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
            .model = model,
            .choices = choices,
            .usage = usage,
        };
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const host = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_VLLM_HOST",
        "VLLM_HOST",
    })) orelse try allocator.dupe(u8, "http://localhost:8000");
    errdefer allocator.free(host);

    const api_key = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_VLLM_API_KEY",
        "VLLM_API_KEY",
    });
    errdefer if (api_key) |k| shared.secureFree(allocator, k);

    const model = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_VLLM_MODEL",
        "VLLM_MODEL",
    })) orelse try allocator.dupe(u8, "default");

    return .{
        .host = host,
        .api_key = api_key,
        .model = model,
        .model_owned = true,
        .timeout_ms = 120_000,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}

/// Check if the vLLM connector is available (host env var is set).
/// This is a zero-allocation health check suitable for status dashboards.
pub fn isAvailable() bool {
    return shared.anyEnvIsSet(&.{
        "ABI_VLLM_HOST",
        "VLLM_HOST",
    });
}

test "vllm config deinit" {
    const allocator = std.testing.allocator;
    var config = Config{
        .host = try allocator.dupe(u8, "http://localhost:8000"),
    };
    config.deinit(allocator);
}

test "vllm config deinit with api key" {
    const allocator = std.testing.allocator;
    var config = Config{
        .host = try allocator.dupe(u8, "http://localhost:8000"),
        .api_key = try allocator.dupe(u8, "test-key"),
    };
    config.deinit(allocator);
}

test "vllm chat request encoding" {
    const allocator = std.testing.allocator;

    const config = Config{
        .host = try allocator.dupe(u8, "http://localhost:8000"),
    };
    var client = try Client.init(allocator, config);
    defer client.deinit();

    const msgs = [_]Message{
        .{ .role = "user", .content = "Hello" },
    };

    const json = try client.encodeChatRequest(.{
        .model = "meta-llama/Llama-3-8B",
        .messages = &msgs,
    });
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"model\":\"meta-llama/Llama-3-8B\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"role\":\"user\"") != null);
}

test "isAvailable returns bool" {
    _ = isAvailable();
}
