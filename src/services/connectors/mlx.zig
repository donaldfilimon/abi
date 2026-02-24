//! MLX API connector.
//!
//! Provides integration with locally-running MLX inference servers via
//! the OpenAI-compatible API at `/v1/chat/completions`.
//!
//! MLX is Apple's machine learning framework optimized for Apple Silicon.
//! The `mlx-lm` Python package includes a server (`python -m mlx_lm.server`)
//! that exposes an OpenAI-compatible endpoint.
//!
//! ## Environment Variables
//!
//! - `ABI_MLX_HOST`: Host URL (default: http://localhost:8080)
//! - `ABI_MLX_MODEL`: Default model name
//! - `ABI_MLX_API_KEY`: Optional API key
//!
//! ## Example
//!
//! ```zig
//! const mlx = @import("abi").connectors.mlx;
//!
//! var client = try mlx.createClient(allocator);
//! defer client.deinit();
//!
//! const response = try client.chatSimple("Hello, world!");
//! ```

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const async_http = @import("../shared/utils.zig").async_http;
const json_utils = @import("../shared/utils.zig").json;

/// Errors that can occur when interacting with MLX.
pub const MLXError = error{
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

        var http_res = try self.http.fetchJsonWithRetry(&http_req, shared.DEFAULT_RETRY_OPTIONS);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            if (http_res.status_code == 429) {
                return MLXError.RateLimitExceeded;
            }
            return MLXError.ApiRequestFailed;
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

    /// Generate text (convenience wrapper around chat completion).
    /// Returns the first choice's message content.
    pub fn generate(self: *Client, prompt: []const u8, max_tokens: ?u32) ![]u8 {
        const msgs = [_]Message{
            .{ .role = "user", .content = prompt },
        };
        const response = try self.chatCompletion(.{
            .model = self.config.model,
            .messages = &msgs,
            .max_tokens = max_tokens,
        });
        // The response fields are allocated — caller owns the content string.
        // We need to free the other parts and return just the content.
        defer self.allocator.free(response.id);
        defer self.allocator.free(response.model);
        defer {
            for (response.choices) |choice| {
                self.allocator.free(choice.finish_reason);
                self.allocator.free(choice.message.role);
            }
            self.allocator.free(response.choices);
        }

        if (response.choices.len == 0) return MLXError.InvalidResponse;
        // Caller takes ownership of content
        return @constCast(response.choices[0].message.content);
    }

    fn encodeChatRequest(self: *Client, request: ChatCompletionRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8).empty;
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
            return MLXError.InvalidResponse;
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
    const host_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_MLX_HOST",
        "MLX_HOST",
    });
    // Treat empty host as unset — fall through to default
    const host = if (host_raw) |h| blk: {
        if (h.len == 0) {
            allocator.free(h);
            break :blk try allocator.dupe(u8, "http://localhost:8080");
        }
        break :blk h;
    } else try allocator.dupe(u8, "http://localhost:8080");
    errdefer allocator.free(host);

    const api_key_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_MLX_API_KEY",
        "MLX_API_KEY",
    });
    // Treat empty API key as unset (optional field)
    const api_key: ?[]u8 = if (api_key_raw) |k| blk: {
        if (k.len == 0) {
            shared.secureFree(allocator, k);
            break :blk null;
        }
        break :blk k;
    } else null;
    errdefer if (api_key) |k| shared.secureFree(allocator, k);

    const model_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_MLX_MODEL",
        "MLX_MODEL",
    });
    const model = if (model_raw) |m| blk: {
        if (m.len == 0) {
            allocator.free(m);
            break :blk try allocator.dupe(u8, "default");
        }
        break :blk m;
    } else try allocator.dupe(u8, "default");

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

/// Check if the MLX connector is available (host env var is set).
/// This is a zero-allocation health check suitable for status dashboards.
pub fn isAvailable() bool {
    return shared.anyEnvIsSet(&.{
        "ABI_MLX_HOST",
        "MLX_HOST",
    });
}

test "mlx config deinit" {
    const allocator = std.testing.allocator;
    var config = Config{
        .host = try allocator.dupe(u8, "http://localhost:8080"),
    };
    config.deinit(allocator);
}

test "mlx config deinit with api key" {
    const allocator = std.testing.allocator;
    var config = Config{
        .host = try allocator.dupe(u8, "http://localhost:8080"),
        .api_key = try allocator.dupe(u8, "test-key"),
    };
    config.deinit(allocator);
}

test "mlx chat request encoding" {
    const allocator = std.testing.allocator;

    const config = Config{
        .host = try allocator.dupe(u8, "http://localhost:8080"),
    };
    var client = try Client.init(allocator, config);
    defer client.deinit();

    const msgs = [_]Message{
        .{ .role = "user", .content = "Hello" },
    };

    const json = try client.encodeChatRequest(.{
        .model = "test-model",
        .messages = &msgs,
    });
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"model\":\"test-model\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"role\":\"user\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "/v1/chat/completions") == null);
}

test "isAvailable returns bool" {
    _ = isAvailable();
}

test {
    std.testing.refAllDecls(@This());
}
