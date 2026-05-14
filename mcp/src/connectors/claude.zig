//! Claude API connector.
//!
//! Anthropic-compatible connector wired with Claude-specific environment
//! variables and fallback to Anthropic env names.

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const anthropic = @import("anthropic.zig");

pub const ClaudeError = anthropic.AnthropicError;
pub const Config = anthropic.Config;
pub const Message = anthropic.Message;
pub const MessagesRequest = anthropic.MessagesRequest;
pub const ContentBlock = anthropic.ContentBlock;
pub const MessagesResponse = anthropic.MessagesResponse;
pub const Usage = anthropic.Usage;
pub const EmbeddingRequest = anthropic.EmbeddingRequest;
pub const EmbeddingResponse = anthropic.EmbeddingResponse;
pub const EmbeddingData = anthropic.EmbeddingData;
pub const EmbeddingUsage = anthropic.EmbeddingUsage;
pub const Client = anthropic.Client;

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const loaded = try shared.loadConfigFromEnv(allocator, .{
        .api_key_env = &.{ "ABI_CLAUDE_API_KEY", "CLAUDE_API_KEY", "ABI_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY" },
        .base_url_env = &.{ "ABI_CLAUDE_BASE_URL", "CLAUDE_BASE_URL", "ABI_ANTHROPIC_BASE_URL", "ANTHROPIC_BASE_URL" },
        .model_env = &.{ "ABI_CLAUDE_MODEL", "CLAUDE_MODEL", "ABI_ANTHROPIC_MODEL", "ANTHROPIC_MODEL" },
        .default_base_url = "https://api.anthropic.com/v1",
        .default_model = "claude-3-5-sonnet-20241022",
        .api_key_required = true,
    }, ClaudeError.MissingApiKey);

    return .{
        .api_key = loaded.api_key.?,
        .base_url = loaded.base_url,
        .model = loaded.model,
        .model_owned = true,
        .max_tokens = 4096,
        .timeout_ms = 120_000,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}

pub fn isAvailable() bool {
    return shared.anyEnvIsSet(&.{
        "ABI_CLAUDE_API_KEY",
        "CLAUDE_API_KEY",
        "ABI_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
    });
}

test "claude availability returns bool" {
    _ = isAvailable();
}

test {
    std.testing.refAllDecls(@This());
}
