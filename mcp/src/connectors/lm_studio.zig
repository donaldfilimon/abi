//! LM Studio API connector.
//!
//! Provides integration with locally-running LM Studio instances via
//! the OpenAI-compatible API at `/v1/chat/completions`.
//!
//! ## Environment Variables
//!
//! - `ABI_LM_STUDIO_HOST`: Host URL (default: http://localhost:1234)
//! - `ABI_LM_STUDIO_MODEL`: Default model name
//! - `ABI_LM_STUDIO_API_KEY`: Optional API key
//!
//! ## Example
//!
//! ```zig
//! const lm_studio = @import("abi").connectors.lm_studio;
//!
//! var client = try lm_studio.createClient(allocator);
//! defer client.deinit();
//!
//! const response = try client.chatSimple("Hello, world!");
//! ```

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const async_http = @import("../foundation/mod.zig").utils.async_http;

/// Errors that can occur when interacting with LM Studio.
pub const LMStudioError = error{
    /// The API request failed (network error or non-2xx status).
    ApiRequestFailed,
    /// The API response could not be parsed.
    InvalidResponse,
    /// Rate limit exceeded (HTTP 429).
    RateLimitExceeded,
};

// Re-export shared OpenAI-compatible types for backward compatibility.
pub const Config = shared.OpenAICompatConfig;
pub const Message = shared.ChatMessage;
pub const ChatCompletionRequest = shared.OpenAICompatChatRequest;
pub const ChatCompletionResponse = shared.OpenAICompatChatResponse;
pub const Choice = shared.OpenAICompatChoice;
pub const Usage = shared.OpenAICompatUsage;

/// LM Studio client using shared OpenAI-compatible types and encode/decode helpers.
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

    /// Encode a chat request to JSON (exposed for testing).
    pub fn encodeChatRequest(self: *Client, request: ChatCompletionRequest) ![]u8 {
        return try shared.openaiCompatEncodeChatRequest(self.allocator, request);
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const host_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_LM_STUDIO_HOST",
        "LM_STUDIO_HOST",
    });
    // Treat empty host as unset — fall through to default
    const host = if (host_raw) |h| blk: {
        if (h.len == 0) {
            allocator.free(h);
            break :blk try allocator.dupe(u8, "http://localhost:1234");
        }
        break :blk h;
    } else try allocator.dupe(u8, "http://localhost:1234");
    errdefer allocator.free(host);

    const api_key_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_LM_STUDIO_API_KEY",
        "LM_STUDIO_API_KEY",
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
        "ABI_LM_STUDIO_MODEL",
        "LM_STUDIO_MODEL",
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

/// Check if the LM Studio connector is available (host env var is set).
/// This is a zero-allocation health check suitable for status dashboards.
pub fn isAvailable() bool {
    return shared.anyEnvIsSet(&.{
        "ABI_LM_STUDIO_HOST",
        "LM_STUDIO_HOST",
    });
}

test "lm_studio config deinit" {
    const allocator = std.testing.allocator;
    var config = Config{
        .host = try allocator.dupe(u8, "http://localhost:1234"),
    };
    config.deinit(allocator);
}

test "lm_studio config deinit with api key" {
    const allocator = std.testing.allocator;
    var config = Config{
        .host = try allocator.dupe(u8, "http://localhost:1234"),
        .api_key = try allocator.dupe(u8, "test-key"),
    };
    config.deinit(allocator);
}

test "lm_studio chat request encoding" {
    const allocator = std.testing.allocator;

    const config = Config{
        .host = try allocator.dupe(u8, "http://localhost:1234"),
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
    try std.testing.expect(std.mem.indexOf(u8, json, "/v1/chat/completions") == null); // Not a URL
}

test "isAvailable returns bool" {
    _ = isAvailable();
}

test {
    std.testing.refAllDecls(@This());
}
