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
    const api_key_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_CLAUDE_API_KEY",
        "CLAUDE_API_KEY",
        "ABI_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
    });
    const api_key = api_key_raw orelse return ClaudeError.MissingApiKey;
    if (api_key.len == 0) {
        allocator.free(api_key);
        return ClaudeError.MissingApiKey;
    }
    errdefer allocator.free(api_key);

    const base_url_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_CLAUDE_BASE_URL",
        "CLAUDE_BASE_URL",
        "ABI_ANTHROPIC_BASE_URL",
        "ANTHROPIC_BASE_URL",
    });
    const base_url = if (base_url_raw) |u| blk: {
        if (u.len == 0) {
            allocator.free(u);
            break :blk try allocator.dupe(u8, "https://api.anthropic.com/v1");
        }
        break :blk u;
    } else try allocator.dupe(u8, "https://api.anthropic.com/v1");
    errdefer allocator.free(base_url);

    const model_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_CLAUDE_MODEL",
        "CLAUDE_MODEL",
        "ABI_ANTHROPIC_MODEL",
        "ANTHROPIC_MODEL",
    });
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
