//! OpenCode API connector.
//!
//! OpenAI-compatible chat-completions connector wired with OpenCode-specific
//! environment variables.

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const openai = @import("openai.zig");

pub const OpenCodeError = openai.OpenAIError;
pub const Config = openai.Config;
pub const Message = openai.Message;
pub const ChatCompletionRequest = openai.ChatCompletionRequest;
pub const ChatCompletionResponse = openai.ChatCompletionResponse;
pub const StreamingChunk = openai.StreamingChunk;
pub const StreamingChoice = openai.StreamingChoice;
pub const StreamingDelta = openai.StreamingDelta;
pub const Client = openai.Client;

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const loaded = try shared.loadConfigFromEnv(allocator, .{
        .api_key_env = &.{ "ABI_OPENCODE_API_KEY", "OPENCODE_API_KEY" },
        .base_url_env = &.{ "ABI_OPENCODE_BASE_URL", "OPENCODE_BASE_URL" },
        .model_env = &.{ "ABI_OPENCODE_MODEL", "OPENCODE_MODEL" },
        .default_base_url = "https://api.openai.com/v1",
        .default_model = "gpt-4o-mini",
        .api_key_required = true,
    }, OpenCodeError.MissingApiKey);

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
        "ABI_OPENCODE_API_KEY",
        "OPENCODE_API_KEY",
    });
}

test "opencode availability returns bool" {
    _ = isAvailable();
}

test {
    std.testing.refAllDecls(@This());
}
