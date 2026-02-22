//! Anthropic Claude API connector.
//!
//! Provides integration with Anthropic's Claude models via the Messages API.
//! Supports chat completions, streaming, and embeddings.

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const async_http = @import("../shared/utils.zig").async_http;
const json_utils = @import("../shared/utils.zig").json;

/// Errors that can occur when interacting with the Anthropic API.
pub const AnthropicError = error{
    /// API key was not provided via environment variable.
    MissingApiKey,
    /// The API request failed (network error or non-2xx status).
    ApiRequestFailed,
    /// The API response could not be parsed.
    InvalidResponse,
    /// Rate limit exceeded (HTTP 429). Retry after backoff.
    RateLimitExceeded,
    /// Content was filtered by Anthropic's safety systems.
    ContentFiltered,
};

pub const Config = struct {
    api_key: []u8,
    base_url: []u8,
    model: []const u8 = "claude-3-5-sonnet-20241022",
    model_owned: bool = false,
    max_tokens: u32 = 4096,
    timeout_ms: u32 = 120_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        shared.deinitConfig(allocator, self.api_key, self.base_url);
        if (self.model_owned) allocator.free(@constCast(self.model));
        self.* = undefined;
    }
};

pub const Message = shared.ChatMessage;

pub const MessagesRequest = struct {
    model: []const u8,
    messages: []const Message,
    max_tokens: u32 = 4096,
    temperature: f32 = 0.7,
    system: ?[]const u8 = null,
    stream: bool = false,
};

pub const ContentBlock = struct {
    type: []const u8,
    text: []const u8,
};

pub const MessagesResponse = struct {
    id: []const u8,
    type: []const u8,
    role: []const u8,
    content: []ContentBlock,
    model: []const u8,
    stop_reason: ?[]const u8,
    usage: Usage,
};

pub const Usage = struct {
    input_tokens: u32,
    output_tokens: u32,
};

pub const EmbeddingRequest = struct {
    model: []const u8 = "voyage-3",
    input: []const []const u8,
    input_type: []const u8 = "document",
};

pub const EmbeddingResponse = struct {
    object: []const u8,
    data: []EmbeddingData,
    model: []const u8,
    usage: EmbeddingUsage,
};

pub const EmbeddingData = struct {
    object: []const u8,
    index: u32,
    embedding: []f32,
};

pub const EmbeddingUsage = struct {
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

    pub fn messages(self: *Client, request: MessagesRequest) !MessagesResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/messages", .{self.config.base_url});
        defer self.allocator.free(url);

        const json = try self.encodeMessagesRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .post, url);
        defer http_req.deinit();

        try http_req.setHeader("x-api-key", self.config.api_key);
        try http_req.setHeader("anthropic-version", "2023-06-01");
        try http_req.setJsonBody(json);

        var http_res = try self.http.fetchJsonWithRetry(&http_req, shared.DEFAULT_RETRY_OPTIONS);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            if (http_res.status_code == 429) {
                return AnthropicError.RateLimitExceeded;
            }
            return AnthropicError.ApiRequestFailed;
        }

        return try self.decodeMessagesResponse(http_res.body);
    }

    pub fn chat(self: *Client, user_messages: []const Message) !MessagesResponse {
        return try self.messages(.{
            .model = self.config.model,
            .messages = user_messages,
            .max_tokens = self.config.max_tokens,
        });
    }

    pub fn chatSimple(self: *Client, prompt: []const u8) !MessagesResponse {
        const msgs = [_]Message{
            .{ .role = "user", .content = prompt },
        };
        return try self.chat(&msgs);
    }

    pub fn chatWithSystem(self: *Client, system_prompt: []const u8, user_messages: []const Message) !MessagesResponse {
        return try self.messages(.{
            .model = self.config.model,
            .messages = user_messages,
            .max_tokens = self.config.max_tokens,
            .system = system_prompt,
        });
    }

    pub fn messagesStreaming(self: *Client, request: MessagesRequest) !async_http.StreamingResponse {
        var req = request;
        req.stream = true;

        const url = try std.fmt.allocPrint(self.allocator, "{s}/messages", .{self.config.base_url});
        defer self.allocator.free(url);

        const json = try self.encodeMessagesRequest(req);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .post, url);
        defer http_req.deinit();

        try http_req.setHeader("x-api-key", self.config.api_key);
        try http_req.setHeader("anthropic-version", "2023-06-01");
        try http_req.setJsonBody(json);

        return try self.http.fetchStreaming(&http_req);
    }

    fn encodeMessagesRequest(self: *Client, request: MessagesRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8).empty;
        errdefer json_str.deinit(self.allocator);

        try json_str.appendSlice(self.allocator, "{\"model\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &json_str, request.model);
        try json_str.print(self.allocator, "\",\"max_tokens\":{d},\"messages\":[", .{
            request.max_tokens,
        });

        try shared.encodeMessageArray(self.allocator, &json_str, request.messages);

        try json_str.appendSlice(self.allocator, "]");

        if (request.system) |system| {
            try json_str.print(self.allocator, ",\"system\":\"{}\"", .{json_utils.jsonEscape(system)});
        }

        try json_str.print(self.allocator, ",\"temperature\":{d:.2}", .{request.temperature});

        if (request.stream) {
            try json_str.appendSlice(self.allocator, ",\"stream\":true");
        }

        try json_str.append(self.allocator, '}');

        return json_str.toOwnedSlice(self.allocator);
    }

    fn decodeMessagesResponse(self: *Client, json: []const u8) !MessagesResponse {
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

        const type_str = try json_utils.parseStringField(object, "type", self.allocator);
        errdefer self.allocator.free(type_str);

        const role = try json_utils.parseStringField(object, "role", self.allocator);
        errdefer self.allocator.free(role);

        const model = try json_utils.parseStringField(object, "model", self.allocator);
        errdefer self.allocator.free(model);

        const stop_reason = json_utils.parseOptionalStringField(object, "stop_reason", self.allocator) catch null;
        errdefer if (stop_reason) |sr| self.allocator.free(sr);

        const content_array = try json_utils.parseArrayField(object, "content");
        var content_blocks = try self.allocator.alloc(ContentBlock, content_array.items.len);
        errdefer self.allocator.free(content_blocks);

        for (content_array.items, 0..) |item, i| {
            const content_obj = try json_utils.getRequiredObject(item);
            const block_type = try json_utils.parseStringField(content_obj, "type", self.allocator);
            errdefer self.allocator.free(block_type);
            const text = try json_utils.parseStringField(content_obj, "text", self.allocator);
            content_blocks[i] = .{ .type = block_type, .text = text };
        }

        const usage_obj = try json_utils.parseObjectField(object, "usage");
        const usage = Usage{
            .input_tokens = @intCast(try json_utils.parseIntField(usage_obj, "input_tokens")),
            .output_tokens = @intCast(try json_utils.parseIntField(usage_obj, "output_tokens")),
        };

        return MessagesResponse{
            .id = id,
            .type = type_str,
            .role = role,
            .content = content_blocks,
            .model = model,
            .stop_reason = stop_reason,
            .usage = usage,
        };
    }

    /// Get text content from response (combines all text blocks)
    pub fn getResponseText(self: *Client, response: MessagesResponse) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        for (response.content) |block| {
            if (std.mem.eql(u8, block.type, "text")) {
                try result.appendSlice(self.allocator, block.text);
            }
        }

        return result.toOwnedSlice(self.allocator);
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const api_key_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
    });
    const api_key = api_key_raw orelse return AnthropicError.MissingApiKey;
    // Treat empty string as missing (e.g., ANTHROPIC_API_KEY="")
    if (api_key.len == 0) {
        allocator.free(api_key);
        return AnthropicError.MissingApiKey;
    }
    errdefer allocator.free(api_key);

    const base_url_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_ANTHROPIC_BASE_URL",
        "ANTHROPIC_BASE_URL",
    });
    // Treat empty base URL as unset — fall through to default
    const base_url = if (base_url_raw) |u| blk: {
        if (u.len == 0) {
            allocator.free(u);
            break :blk try allocator.dupe(u8, "https://api.anthropic.com/v1");
        }
        break :blk u;
    } else try allocator.dupe(u8, "https://api.anthropic.com/v1");
    errdefer allocator.free(base_url);

    const model_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_ANTHROPIC_MODEL",
        "ANTHROPIC_MODEL",
    });
    // Treat empty model as unset — fall through to default
    const model = if (model_raw) |m| blk: {
        if (m.len == 0) {
            allocator.free(m);
            break :blk try allocator.dupe(u8, "claude-3-5-sonnet-20241022");
        }
        break :blk m;
    } else try allocator.dupe(u8, "claude-3-5-sonnet-20241022");

    return .{
        .api_key = api_key,
        .base_url = base_url,
        .model = model,
        .model_owned = true,
        .max_tokens = 4096,
        .timeout_ms = 120_000,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}

/// Check if the Anthropic connector is available (API key env var is set).
/// This is a zero-allocation health check suitable for status dashboards.
pub fn isAvailable() bool {
    return shared.anyEnvIsSet(&.{
        "ABI_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
    });
}

test "anthropic config deinit" {
    const allocator = std.testing.allocator;
    var config = Config{
        .api_key = try allocator.dupe(u8, "test-key"),
        .base_url = try allocator.dupe(u8, "https://api.anthropic.com/v1"),
    };
    config.deinit(allocator);
}

test "anthropic message encoding" {
    const allocator = std.testing.allocator;

    const config = Config{
        .api_key = try allocator.dupe(u8, "test-key"),
        .base_url = try allocator.dupe(u8, "https://api.anthropic.com/v1"),
    };
    var client = try Client.init(allocator, config);
    defer client.deinit();

    const msgs = [_]Message{
        .{ .role = "user", .content = "Hello" },
    };

    const json = try client.encodeMessagesRequest(.{
        .model = "claude-3-5-sonnet-20241022",
        .messages = &msgs,
        .max_tokens = 1024,
    });
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"model\":\"claude-3-5-sonnet-20241022\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"max_tokens\":1024") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"role\":\"user\"") != null);
}

test "isAvailable returns bool" {
    // Just verify it returns without crashing - actual availability depends on env
    _ = isAvailable();
}
