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
    const api_key_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OPENCODE_API_KEY",
        "OPENCODE_API_KEY",
    });
    const api_key = api_key_raw orelse return OpenCodeError.MissingApiKey;
    if (api_key.len == 0) {
        allocator.free(api_key);
        return OpenCodeError.MissingApiKey;
    }
    errdefer allocator.free(api_key);

    const base_url_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OPENCODE_BASE_URL",
        "OPENCODE_BASE_URL",
    });
    const base_url = if (base_url_raw) |u| blk: {
        if (u.len == 0) {
            allocator.free(u);
            break :blk try allocator.dupe(u8, "https://api.openai.com/v1");
        }
        break :blk u;
    } else try allocator.dupe(u8, "https://api.openai.com/v1");
    errdefer allocator.free(base_url);

    const model_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OPENCODE_MODEL",
        "OPENCODE_MODEL",
    });
    const model = if (model_raw) |m| blk: {
        if (m.len == 0) {
            allocator.free(m);
            break :blk try allocator.dupe(u8, "gpt-4o-mini");
        }
        break :blk m;
    } else try allocator.dupe(u8, "gpt-4o-mini");

    return .{
        .api_key = api_key,
        .base_url = base_url,
        .model = model,
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
