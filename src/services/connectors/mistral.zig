//! Mistral AI API connector.
//!
//! Provides integration with Mistral AI models via their OpenAI-compatible API.
//! Supports chat completions, streaming, and embeddings.

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const async_http = @import("../shared/utils.zig").async_http;
const json_utils = @import("../shared/utils.zig").json;

/// Errors that can occur when interacting with the Mistral API.
pub const MistralError = error{
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
    model: []const u8 = "mistral-large-latest",
    timeout_ms: u32 = 60_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        // Use shared secure cleanup helper
        shared.deinitConfig(allocator, self.api_key, self.base_url);
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
    safe_prompt: bool = false,
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

pub const EmbeddingRequest = struct {
    model: []const u8 = "mistral-embed",
    input: []const []const u8,
    encoding_format: []const u8 = "float",
};

pub const EmbeddingResponse = struct {
    id: []const u8,
    object: []const u8,
    model: []const u8,
    data: []EmbeddingData,
    usage: EmbeddingUsage,
};

pub const EmbeddingData = struct {
    object: []const u8,
    index: u32,
    embedding: []f32,
};

pub const EmbeddingUsage = struct {
    prompt_tokens: u32,
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

        const http_res = try self.http.fetchJson(&http_req);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            if (http_res.status_code == 429) {
                return MistralError.RateLimitExceeded;
            }
            return MistralError.ApiRequestFailed;
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

    pub fn embeddings(self: *Client, request: EmbeddingRequest) !EmbeddingResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/embeddings", .{self.config.base_url});
        defer self.allocator.free(url);

        const json = try self.encodeEmbeddingRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .POST, url);
        defer http_req.deinit();

        try http_req.setBearerToken(self.config.api_key);
        try http_req.setJsonBody(json);

        const http_res = try self.http.fetchJson(&http_req);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            return MistralError.ApiRequestFailed;
        }

        return try self.decodeEmbeddingResponse(http_res.body);
    }

    pub fn embed(self: *Client, texts: []const []const u8) !EmbeddingResponse {
        return try self.embeddings(.{
            .model = "mistral-embed",
            .input = texts,
        });
    }

    pub fn embedSingle(self: *Client, text: []const u8) ![]f32 {
        const texts = [_][]const u8{text};
        const response = try self.embed(&texts);
        defer {
            for (response.data) |data| {
                self.allocator.free(data.embedding);
            }
            self.allocator.free(response.data);
        }

        if (response.data.len == 0) {
            return MistralError.InvalidResponse;
        }

        return try self.allocator.dupe(f32, response.data[0].embedding);
    }

    fn encodeChatRequest(self: *Client, request: ChatCompletionRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8){};
        errdefer json_str.deinit(self.allocator);

        try json_str.print(self.allocator, "{{\"model\":\"{s}\",\"messages\":[", .{request.model});

        for (request.messages, 0..) |msg, i| {
            if (i > 0) try json_str.append(self.allocator, ',');
            try json_str.print(
                self.allocator,
                "{{\"role\":\"{s}\",\"content\":\"{}\"}}",
                .{ msg.role, json_utils.jsonEscape(msg.content) },
            );
        }

        try json_str.print(self.allocator, "],\"temperature\":{d:.2},\"top_p\":{d:.2}", .{
            request.temperature,
            request.top_p,
        });

        if (request.max_tokens) |max_tokens| {
            try json_str.print(self.allocator, ",\"max_tokens\":{d}", .{max_tokens});
        }

        if (request.safe_prompt) {
            try json_str.appendSlice(self.allocator, ",\"safe_prompt\":true");
        }

        if (request.stream) {
            try json_str.appendSlice(self.allocator, ",\"stream\":true");
        }

        try json_str.append(self.allocator, '}');

        return json_str.toOwnedSlice(self.allocator);
    }

    fn encodeEmbeddingRequest(self: *Client, request: EmbeddingRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8){};
        errdefer json_str.deinit(self.allocator);

        try json_str.print(self.allocator, "{{\"model\":\"{s}\",\"input\":[", .{request.model});

        for (request.input, 0..) |text, i| {
            if (i > 0) try json_str.append(self.allocator, ',');
            try json_str.print(self.allocator, "\"{}\"", .{json_utils.jsonEscape(text)});
        }

        try json_str.print(self.allocator, "],\"encoding_format\":\"{s}\"}}", .{request.encoding_format});

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

        const obj = try json_utils.parseStringField(object, "object", self.allocator);
        errdefer self.allocator.free(obj);

        const created = try json_utils.parseUintField(object, "created");

        const model = try json_utils.parseStringField(object, "model", self.allocator);
        errdefer self.allocator.free(model);

        const choices_array = try json_utils.parseArrayField(object, "choices");
        if (choices_array.items.len == 0) {
            return MistralError.InvalidResponse;
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
            .object = obj,
            .created = created,
            .model = model,
            .choices = choices,
            .usage = usage,
        };
    }

    fn decodeEmbeddingResponse(self: *Client, json: []const u8) !EmbeddingResponse {
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

        const model = try json_utils.parseStringField(object, "model", self.allocator);
        errdefer self.allocator.free(model);

        const data_array = try json_utils.parseArrayField(object, "data");
        var data = try self.allocator.alloc(EmbeddingData, data_array.items.len);
        errdefer self.allocator.free(data);

        for (data_array.items, 0..) |item, i| {
            const data_obj = try json_utils.getRequiredObject(item);
            const data_object = try json_utils.parseStringField(data_obj, "object", self.allocator);
            const index: u32 = @intCast(try json_utils.parseIntField(data_obj, "index"));

            const embedding_array = try json_utils.parseArrayField(data_obj, "embedding");
            var embedding = try self.allocator.alloc(f32, embedding_array.items.len);

            for (embedding_array.items, 0..) |val, j| {
                embedding[j] = @floatCast(val.float);
            }

            data[i] = .{
                .object = data_object,
                .index = index,
                .embedding = embedding,
            };
        }

        const usage_obj = try json_utils.parseObjectField(object, "usage");
        const usage = EmbeddingUsage{
            .prompt_tokens = @intCast(try json_utils.parseIntField(usage_obj, "prompt_tokens")),
            .total_tokens = @intCast(try json_utils.parseIntField(usage_obj, "total_tokens")),
        };

        return EmbeddingResponse{
            .id = id,
            .object = obj,
            .model = model,
            .data = data,
            .usage = usage,
        };
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const api_key = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_MISTRAL_API_KEY",
        "MISTRAL_API_KEY",
    })) orelse return MistralError.MissingApiKey;

    const base_url = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_MISTRAL_BASE_URL",
        "MISTRAL_BASE_URL",
    })) orelse try allocator.dupe(u8, "https://api.mistral.ai/v1");

    const model = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_MISTRAL_MODEL",
        "MISTRAL_MODEL",
    })) orelse try allocator.dupe(u8, "mistral-large-latest");
    errdefer allocator.free(model);

    return .{
        .api_key = api_key,
        .base_url = base_url,
        .model = model,
        .timeout_ms = 60_000,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}

test "mistral config deinit" {
    const allocator = std.testing.allocator;
    var config = Config{
        .api_key = try allocator.dupe(u8, "test-key"),
        .base_url = try allocator.dupe(u8, "https://api.mistral.ai/v1"),
    };
    config.deinit(allocator);
}

test "mistral chat request encoding" {
    const allocator = std.testing.allocator;

    const config = Config{
        .api_key = try allocator.dupe(u8, "test-key"),
        .base_url = try allocator.dupe(u8, "https://api.mistral.ai/v1"),
    };
    var client = try Client.init(allocator, config);
    defer client.deinit();

    const msgs = [_]Message{
        .{ .role = "user", .content = "Hello" },
    };

    const json = try client.encodeChatRequest(.{
        .model = "mistral-large-latest",
        .messages = &msgs,
    });
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"model\":\"mistral-large-latest\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"role\":\"user\"") != null);
}

test "mistral embedding request encoding" {
    const allocator = std.testing.allocator;

    const config = Config{
        .api_key = try allocator.dupe(u8, "test-key"),
        .base_url = try allocator.dupe(u8, "https://api.mistral.ai/v1"),
    };
    var client = try Client.init(allocator, config);
    defer client.deinit();

    const texts = [_][]const u8{ "Hello", "World" };
    const json = try client.encodeEmbeddingRequest(.{
        .model = "mistral-embed",
        .input = &texts,
    });
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"model\":\"mistral-embed\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"Hello\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"World\"") != null);
}
