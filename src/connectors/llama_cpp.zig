//! llama.cpp API connector.
//!
//! Provides integration with llama.cpp inference servers via the OpenAI-compatible
//! API at `/v1/chat/completions`.
//!
//! ## Environment Variables
//!
//! - `ABI_LLAMA_CPP_HOST`: Host URL (default: http://localhost:8080)
//! - `ABI_LLAMA_CPP_MODEL`: Default model name
//! - `ABI_LLAMA_CPP_API_KEY`: Optional API key

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const async_http = @import("../foundation/mod.zig").utils.async_http;

/// Errors that can occur when interacting with llama.cpp.
pub const LlamaCppError = error{
    ApiRequestFailed,
    InvalidResponse,
    RateLimitExceeded,
};

// Re-export shared OpenAI-compatible types for backward compatibility.
pub const Config = shared.OpenAICompatConfig;
pub const Message = shared.ChatMessage;
pub const ChatCompletionRequest = shared.OpenAICompatChatRequest;
pub const ChatCompletionResponse = shared.OpenAICompatChatResponse;
pub const Choice = shared.OpenAICompatChoice;
pub const Usage = shared.OpenAICompatUsage;

/// llama.cpp client using shared OpenAI-compatible types and encode/decode helpers.
/// Keeps the same field layout (`allocator`, `config`, `http`) for backward
/// compatibility with code that accesses `client.config` directly.
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
        return try shared.openaiCompatChatCompletion(
            self.allocator,
            &self.http,
            &self.config,
            request,
        );
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

    /// Generate text from a prompt, returning just the response content.
    pub fn generate(self: *Client, prompt: []const u8, max_tokens: ?u32) ![]u8 {
        const msgs = [_]Message{
            .{ .role = "user", .content = prompt },
        };

        var response = try shared.openaiCompatChatCompletion(
            self.allocator,
            &self.http,
            &self.config,
            .{
                .model = self.config.model,
                .messages = &msgs,
                .max_tokens = max_tokens,
                .temperature = 0.7,
                .top_p = 0.95,
                .stream = false,
            },
        );
        defer shared.openaiCompatDeinitChatResponse(self.allocator, &response);

        if (response.choices.len == 0) return LlamaCppError.InvalidResponse;
        return try self.allocator.dupe(u8, response.choices[0].message.content);
    }

    /// Encode a chat request to JSON (exposed for testing).
    pub fn encodeChatRequest(self: *Client, request: ChatCompletionRequest) ![]u8 {
        return try shared.openaiCompatEncodeChatRequest(self.allocator, request);
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const host_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_LLAMA_CPP_HOST",
        "LLAMA_CPP_HOST",
    });

    const host = if (host_raw) |h| blk: {
        if (h.len == 0) {
            allocator.free(h);
            break :blk try allocator.dupe(u8, "http://localhost:8080");
        }
        break :blk h;
    } else try allocator.dupe(u8, "http://localhost:8080");
    errdefer allocator.free(host);

    const api_key_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_LLAMA_CPP_API_KEY",
        "LLAMA_CPP_API_KEY",
    });

    const api_key: ?[]u8 = if (api_key_raw) |k| blk: {
        if (k.len == 0) {
            shared.secureFree(allocator, k);
            break :blk null;
        }
        break :blk k;
    } else null;
    errdefer if (api_key) |k| shared.secureFree(allocator, k);

    const model_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_LLAMA_CPP_MODEL",
        "LLAMA_CPP_MODEL",
    });

    const model = if (model_raw) |m| blk: {
        if (m.len == 0) {
            allocator.free(m);
            break :blk try allocator.dupe(u8, "llama");
        }
        break :blk m;
    } else try allocator.dupe(u8, "llama");

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

pub fn isAvailable() bool {
    return shared.anyEnvIsSet(&.{
        "ABI_LLAMA_CPP_HOST",
        "LLAMA_CPP_HOST",
    });
}

test "llama_cpp config deinit" {
    const allocator = std.testing.allocator;
    var config = Config{ .host = try allocator.dupe(u8, "http://localhost:8080") };
    config.deinit(allocator);
}

test "llama_cpp config deinit with api key" {
    const allocator = std.testing.allocator;
    var config = Config{
        .host = try allocator.dupe(u8, "http://localhost:8080"),
        .api_key = try allocator.dupe(u8, "test-key"),
    };
    config.deinit(allocator);
}

test "llama_cpp chat request encoding" {
    const allocator = std.testing.allocator;

    const config = Config{ .host = try allocator.dupe(u8, "http://localhost:8080") };
    var client = try Client.init(allocator, config);
    defer client.deinit();

    const msgs = [_]Message{
        .{ .role = "user", .content = "Hello" },
    };

    const json = try client.encodeChatRequest(.{ .model = "meta-llama/Llama-3-8B", .messages = &msgs });
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"model\":\"meta-llama/Llama-3-8B\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"role\":\"user\"") != null);
}

test "llama_cpp isAvailable returns bool" {
    _ = isAvailable();
}

test {
    std.testing.refAllDecls(@This());
}
