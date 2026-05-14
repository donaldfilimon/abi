//! Codex API connector.
//!
//! OpenAI-compatible chat-completions connector wired with Codex-specific
//! environment variables.

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const openai = @import("openai.zig");

pub const CodexError = openai.OpenAIError;
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
        .api_key_env = &.{ "ABI_CODEX_API_KEY", "CODEX_API_KEY", "ABI_OPENAI_API_KEY", "OPENAI_API_KEY" },
        .base_url_env = &.{ "ABI_CODEX_BASE_URL", "CODEX_BASE_URL", "ABI_OPENAI_BASE_URL", "OPENAI_BASE_URL" },
        .model_env = &.{ "ABI_CODEX_MODEL", "CODEX_MODEL" },
        .default_base_url = "https://api.openai.com/v1",
        .default_model = "gpt-5-codex",
        .api_key_required = true,
    }, CodexError.MissingApiKey);

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
        "ABI_CODEX_API_KEY",
        "CODEX_API_KEY",
        "ABI_OPENAI_API_KEY",
        "OPENAI_API_KEY",
    });
}

test "codex availability returns bool" {
    _ = isAvailable();
}

test {
    std.testing.refAllDecls(@This());
}
