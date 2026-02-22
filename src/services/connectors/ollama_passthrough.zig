//! Ollama passthrough connector.
//!
//! OpenAI-compatible connector for Ollama passthrough endpoints.

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const vllm = @import("vllm.zig");

pub const OllamaPassthroughError = vllm.VLLMError;
pub const Config = vllm.Config;
pub const Message = vllm.Message;
pub const ChatCompletionRequest = vllm.ChatCompletionRequest;
pub const ChatCompletionResponse = vllm.ChatCompletionResponse;
pub const Choice = vllm.Choice;
pub const Usage = vllm.Usage;
pub const Client = vllm.Client;

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const host_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OLLAMA_PASSTHROUGH_URL",
        "OLLAMA_PASSTHROUGH_URL",
    });
    const host = host_raw orelse return error.MissingApiKey;
    if (host.len == 0) {
        allocator.free(host);
        return error.MissingApiKey;
    }
    errdefer allocator.free(host);

    const api_key_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OLLAMA_PASSTHROUGH_API_KEY",
        "OLLAMA_PASSTHROUGH_API_KEY",
    });
    const api_key: ?[]u8 = if (api_key_raw) |key| blk: {
        if (key.len == 0) {
            shared.secureFree(allocator, key);
            break :blk null;
        }
        break :blk key;
    } else null;
    errdefer if (api_key) |key| shared.secureFree(allocator, key);

    const model_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OLLAMA_PASSTHROUGH_MODEL",
        "OLLAMA_PASSTHROUGH_MODEL",
    });
    const model = if (model_raw) |m| blk: {
        if (m.len == 0) {
            allocator.free(m);
            break :blk try allocator.dupe(u8, "gpt-oss");
        }
        break :blk m;
    } else try allocator.dupe(u8, "gpt-oss");

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
        "ABI_OLLAMA_PASSTHROUGH_URL",
        "OLLAMA_PASSTHROUGH_URL",
    });
}

test "ollama_passthrough availability returns bool" {
    _ = isAvailable();
}
