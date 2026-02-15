//! Ollama API connector.
//!
//! Provides integration with locally-running Ollama instances for LLM inference.
//! Supports both text generation and chat completions.
//!
//! ## Environment Variables
//!
//! - `ABI_OLLAMA_HOST` or `OLLAMA_HOST`: Host URL (default: http://127.0.0.1:11434)
//! - `ABI_OLLAMA_MODEL` or `OLLAMA_MODEL`: Default model (default: gpt-oss)
//!
//! ## Example
//!
//! ```zig
//! const ollama = @import("abi").connectors.ollama;
//!
//! var client = try ollama.createClient(allocator);
//! defer client.deinit();
//!
//! const response = try client.chatSimple("Hello, world!");
//! ```

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const async_http = @import("../shared/utils.zig").async_http;
const json_utils = @import("../shared/utils.zig").json;

/// Errors that can occur when interacting with the Ollama API.
pub const OllamaError = error{
    /// The API request failed (network error or non-2xx status).
    ApiRequestFailed,
    /// The API response could not be parsed.
    InvalidResponse,
    /// The model is not available or still loading.
    ModelNotAvailable,
    /// Rate limit exceeded (HTTP 429). Retry after backoff.
    RateLimitExceeded,
};

/// Configuration for connecting to an Ollama instance.
pub const Config = struct {
    host: []u8,
    model: []const u8 = "gpt-oss",
    model_owned: bool = false,
    timeout_ms: u32 = 120_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.host);
        if (self.model_owned) {
            allocator.free(@constCast(self.model));
        }
        self.* = undefined;
    }
};

pub const Message = shared.ChatMessage;

pub const GenerateRequest = struct {
    model: []const u8,
    prompt: []const u8,
    stream: bool = false,
    options: ?Options = null,
};

pub const ChatRequest = struct {
    model: []const u8,
    messages: []const Message,
    stream: bool = false,
};

pub const Options = struct {
    temperature: f32 = 0.7,
    num_predict: u32 = 128,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
};

pub const GenerateResponse = struct {
    model: []const u8,
    response: []const u8,
    done: bool,
    context: ?[]u64 = null,
    total_duration_ns: ?u64 = null,
    load_duration_ns: ?u64 = null,
    prompt_eval_count: ?u32 = null,
    prompt_eval_duration_ns: ?u64 = null,
    eval_count: ?u32 = null,
    eval_duration_ns: ?u64 = null,

    pub fn deinit(self: *GenerateResponse, allocator: std.mem.Allocator) void {
        allocator.free(self.model);
        allocator.free(self.response);
        if (self.context) |ctx| {
            allocator.free(ctx);
        }
        self.* = undefined;
    }
};

pub const ChatResponse = struct {
    model: []const u8,
    message: Message,
    done: bool,

    pub fn deinit(self: *ChatResponse, allocator: std.mem.Allocator) void {
        allocator.free(self.model);
        allocator.free(self.message.role);
        allocator.free(self.message.content);
        self.* = undefined;
    }
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

    pub fn generate(self: *Client, request: GenerateRequest) !GenerateResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/api/generate", .{self.config.host});
        defer self.allocator.free(url);

        const json = try self.encodeGenerateRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .post, url);
        defer http_req.deinit();

        try http_req.setJsonBody(json);

        var http_res = try self.http.fetchJsonWithRetry(&http_req, shared.DEFAULT_RETRY_OPTIONS);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            if (http_res.status_code == 404) {
                return OllamaError.ModelNotAvailable;
            }
            return OllamaError.ApiRequestFailed;
        }

        return try self.decodeGenerateResponse(http_res.body);
    }

    pub fn generateSimple(self: *Client, prompt: []const u8) !GenerateResponse {
        return try self.generate(.{
            .model = self.config.model,
            .prompt = prompt,
        });
    }

    pub fn chat(self: *Client, request: ChatRequest) !ChatResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/api/chat", .{self.config.host});
        defer self.allocator.free(url);

        const json = try self.encodeChatRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .post, url);
        defer http_req.deinit();

        try http_req.setJsonBody(json);

        var http_res = try self.http.fetchJsonWithRetry(&http_req, shared.DEFAULT_RETRY_OPTIONS);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            if (http_res.status_code == 404) {
                return OllamaError.ModelNotAvailable;
            }
            return OllamaError.ApiRequestFailed;
        }

        return try self.decodeChatResponse(http_res.body);
    }

    pub fn chatSimple(self: *Client, prompt: []const u8) !ChatResponse {
        const messages = [_]Message{
            .{ .role = "user", .content = prompt },
        };
        return try self.chat(.{
            .model = self.config.model,
            .messages = messages[0..],
        });
    }

    pub fn encodeGenerateRequest(self: *Client, request: GenerateRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8).empty;
        errdefer json_str.deinit(self.allocator);

        try json_str.appendSlice(self.allocator, "{\"model\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &json_str, request.model);
        try json_str.appendSlice(self.allocator, "\",\"prompt\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &json_str, request.prompt);
        try json_str.appendSlice(self.allocator, "\",\"stream\":");
        try json_str.appendSlice(self.allocator, if (request.stream) "true}" else "false}");

        return json_str.toOwnedSlice(self.allocator);
    }

    pub fn encodeChatRequest(self: *Client, request: ChatRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8).empty;
        errdefer json_str.deinit(self.allocator);

        try json_str.appendSlice(self.allocator, "{\"model\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &json_str, request.model);
        try json_str.appendSlice(self.allocator, "\",\"messages\":[");

        try shared.encodeMessageArray(self.allocator, &json_str, request.messages);

        try json_str.appendSlice(self.allocator, "]}");

        return json_str.toOwnedSlice(self.allocator);
    }

    pub fn decodeGenerateResponse(self: *Client, json: []const u8) !GenerateResponse {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        const model = try json_utils.parseStringField(object, "model", self.allocator);
        errdefer self.allocator.free(model);
        const response = try json_utils.parseStringField(object, "response", self.allocator);
        errdefer self.allocator.free(response);
        const done = try json_utils.parseBoolField(object, "done");

        var context: ?[]u64 = null;
        errdefer if (context) |ctx| self.allocator.free(ctx);
        if (object.get("context")) |context_val| {
            if (context_val == .array) {
                context = try self.allocator.alloc(u64, context_val.array.items.len);
                for (context_val.array.items, 0..) |item, i| {
                    const num = try json_utils.parseUint(item);
                    context.?[i] = num;
                }
            }
        }

        return GenerateResponse{
            .model = model,
            .response = response,
            .done = done,
            .context = context,
            .total_duration_ns = json_utils.parseOptionalUintField(object, "total_duration"),
            .load_duration_ns = json_utils.parseOptionalUintField(object, "load_duration"),
            .prompt_eval_count = parseOptionalU32(object, "prompt_eval_count"),
            .prompt_eval_duration_ns = json_utils.parseOptionalUintField(object, "prompt_eval_duration"),
            .eval_count = parseOptionalU32(object, "eval_count"),
            .eval_duration_ns = json_utils.parseOptionalUintField(object, "eval_duration"),
        };
    }

    pub fn decodeChatResponse(self: *Client, json: []const u8) !ChatResponse {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        const model = try json_utils.parseStringField(object, "model", self.allocator);
        errdefer self.allocator.free(model);
        const message_obj = try json_utils.parseObjectField(object, "message");
        const role = try json_utils.parseStringField(message_obj, "role", self.allocator);
        errdefer self.allocator.free(role);
        const content = try json_utils.parseStringField(message_obj, "content", self.allocator);
        errdefer self.allocator.free(content);
        const done = try json_utils.parseBoolField(object, "done");

        return ChatResponse{
            .model = model,
            .message = .{
                .role = role,
                .content = content,
            },
            .done = done,
        };
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const host_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OLLAMA_HOST",
        "OLLAMA_HOST",
    });
    // Treat empty host as unset — fall through to default
    const host = if (host_raw) |h| blk: {
        if (h.len == 0) {
            allocator.free(h);
            break :blk try allocator.dupe(u8, "http://127.0.0.1:11434");
        }
        break :blk h;
    } else try allocator.dupe(u8, "http://127.0.0.1:11434");
    errdefer allocator.free(host);

    const model_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OLLAMA_MODEL",
        "OLLAMA_MODEL",
    });
    // Treat empty model as unset — fall through to default
    const model = if (model_raw) |m| blk: {
        if (m.len == 0) {
            allocator.free(m);
            break :blk try allocator.dupe(u8, "gpt-oss");
        }
        break :blk m;
    } else try allocator.dupe(u8, "gpt-oss");

    return .{
        .host = host,
        .model = model,
        .model_owned = true,
        .timeout_ms = 120_000,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}

/// Check if the Ollama connector is available (host env var is set).
/// Ollama doesn't require an API key, only a reachable host.
/// This is a zero-allocation health check suitable for status dashboards.
pub fn isAvailable() bool {
    return shared.anyEnvIsSet(&.{
        "ABI_OLLAMA_HOST",
        "OLLAMA_HOST",
    });
}

fn parseOptionalU32(object: std.json.ObjectMap, field: []const u8) ?u32 {
    const raw = json_utils.parseOptionalUintField(object, field) orelse return null;
    return std.math.cast(u32, raw);
}

test "ollama endpoint join" {
    var config = Config{ .host = try std.testing.allocator.dupe(u8, "http://localhost:11434") };
    defer config.deinit(std.testing.allocator);

    const url = try std.fmt.allocPrint(std.testing.allocator, "{s}/api/generate", .{config.host});
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings("http://localhost:11434/api/generate", url);
}

test "isAvailable returns bool" {
    // Just verify it returns without crashing - actual availability depends on env
    _ = isAvailable();
}

test "ollama config lifecycle" {
    const allocator = std.testing.allocator;
    var config = Config{
        .host = try allocator.dupe(u8, "http://myhost:11434"),
        .model = try allocator.dupe(u8, "llama3"),
        .model_owned = true,
        .timeout_ms = 60_000,
    };
    defer config.deinit(allocator);
    try std.testing.expectEqualStrings("http://myhost:11434", config.host);
    try std.testing.expectEqualStrings("llama3", config.model);
    try std.testing.expect(config.model_owned);
    try std.testing.expectEqual(@as(u32, 60_000), config.timeout_ms);
}

test "ollama config defaults" {
    const allocator = std.testing.allocator;
    var config = Config{
        .host = try allocator.dupe(u8, "http://localhost:11434"),
    };
    defer config.deinit(allocator);
    // Default model should be gpt-oss, not owned
    try std.testing.expectEqualStrings("gpt-oss", config.model);
    try std.testing.expect(!config.model_owned);
    try std.testing.expectEqual(@as(u32, 120_000), config.timeout_ms);
}

test "ollama chat endpoint" {
    const allocator = std.testing.allocator;
    var config = Config{ .host = try allocator.dupe(u8, "http://ollama.local:11434") };
    defer config.deinit(allocator);

    const url = try std.fmt.allocPrint(allocator, "{s}/api/chat", .{config.host});
    defer allocator.free(url);
    try std.testing.expectEqualStrings("http://ollama.local:11434/api/chat", url);
}

test "ollama encodeGenerateRequest" {
    const allocator = std.testing.allocator;
    var config = Config{ .host = try allocator.dupe(u8, "http://localhost:11434") };
    var client = Client{
        .allocator = allocator,
        .config = config,
        .http = undefined,
    };

    const json = try client.encodeGenerateRequest(.{
        .model = "llama3",
        .prompt = "Hello, world!",
    });
    defer allocator.free(json);

    // Verify it's valid JSON by parsing
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();

    const obj = try json_utils.getRequiredObject(parsed.value);
    const model_str = try json_utils.parseStringField(obj, "model", allocator);
    defer allocator.free(model_str);
    try std.testing.expectEqualStrings("llama3", model_str);
    const prompt_str = try json_utils.parseStringField(obj, "prompt", allocator);
    defer allocator.free(prompt_str);
    try std.testing.expectEqualStrings("Hello, world!", prompt_str);
    const stream = try json_utils.parseBoolField(obj, "stream");
    try std.testing.expect(!stream);

    config.deinit(allocator);
}

test "ollama encodeChatRequest" {
    const allocator = std.testing.allocator;
    var config = Config{ .host = try allocator.dupe(u8, "http://localhost:11434") };
    var client = Client{
        .allocator = allocator,
        .config = config,
        .http = undefined,
    };

    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Hi!" },
    };
    const json = try client.encodeChatRequest(.{
        .model = "llama3",
        .messages = messages[0..],
    });
    defer allocator.free(json);

    // Verify valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();

    const obj = try json_utils.getRequiredObject(parsed.value);
    const model_str = try json_utils.parseStringField(obj, "model", allocator);
    defer allocator.free(model_str);
    try std.testing.expectEqualStrings("llama3", model_str);
    const arr = try json_utils.parseArrayField(obj, "messages");
    try std.testing.expectEqual(@as(usize, 2), arr.items.len);

    config.deinit(allocator);
}

test "ollama decodeGenerateResponse" {
    const allocator = std.testing.allocator;
    var config = Config{ .host = try allocator.dupe(u8, "http://localhost:11434") };
    var client = Client{
        .allocator = allocator,
        .config = config,
        .http = undefined,
    };

    const json =
        \\{"model":"llama3","response":"Hello!","done":true,"total_duration":1234567}
    ;
    var resp = try client.decodeGenerateResponse(json);
    defer resp.deinit(allocator);

    try std.testing.expectEqualStrings("llama3", resp.model);
    try std.testing.expectEqualStrings("Hello!", resp.response);
    try std.testing.expect(resp.done);
    try std.testing.expectEqual(@as(?u64, 1234567), resp.total_duration_ns);
    try std.testing.expect(resp.context == null);

    config.deinit(allocator);
}

test "ollama decodeChatResponse" {
    const allocator = std.testing.allocator;
    var config = Config{ .host = try allocator.dupe(u8, "http://localhost:11434") };
    var client = Client{
        .allocator = allocator,
        .config = config,
        .http = undefined,
    };

    const json =
        \\{"model":"llama3","message":{"role":"assistant","content":"Hi there!"},"done":true}
    ;
    var resp = try client.decodeChatResponse(json);
    defer resp.deinit(allocator);

    try std.testing.expectEqualStrings("llama3", resp.model);
    try std.testing.expectEqualStrings("assistant", resp.message.role);
    try std.testing.expectEqualStrings("Hi there!", resp.message.content);
    try std.testing.expect(resp.done);

    config.deinit(allocator);
}

test "ollama encodeGenerateRequest with special chars" {
    const allocator = std.testing.allocator;
    var config = Config{ .host = try allocator.dupe(u8, "http://localhost:11434") };
    var client = Client{
        .allocator = allocator,
        .config = config,
        .http = undefined,
    };

    const json = try client.encodeGenerateRequest(.{
        .model = "llama3",
        .prompt = "Say \"hello\" with\nnewlines",
    });
    defer allocator.free(json);

    // Should be valid JSON even with special characters
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();

    const obj = try json_utils.getRequiredObject(parsed.value);
    const prompt = try json_utils.parseStringField(obj, "prompt", allocator);
    defer allocator.free(prompt);
    try std.testing.expectEqualStrings("Say \"hello\" with\nnewlines", prompt);

    config.deinit(allocator);
}

test "ollama parseOptionalU32" {
    const allocator = std.testing.allocator;
    const json_text = "{\"count\":42,\"big\":999999999999}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const obj = try json_utils.getRequiredObject(parsed.value);
    const count = parseOptionalU32(obj, "count");
    try std.testing.expectEqual(@as(?u32, 42), count);

    const missing = parseOptionalU32(obj, "nonexistent");
    try std.testing.expect(missing == null);
}
