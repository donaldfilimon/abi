const std = @import("std");
const types = @import("types.zig");
const model_profiles = @import("model_profiles.zig");

pub const all_providers = [_]types.ProviderId{
    .local_gguf,
    .llama_cpp,
    .mlx,
    .ollama,
    .lm_studio,
    .vllm,
    .anthropic,
    .openai,
    .codex,
    .opencode,
    .claude,
    .gemini,
    .ollama_passthrough,
    .plugin_http,
    .plugin_native,
};

pub const file_model_chain = [_]types.ProviderId{
    .local_gguf,
    .llama_cpp,
    .mlx,
    .ollama,
    .ollama_passthrough,
    .lm_studio,
    .vllm,
    .codex,
    .opencode,
    .claude,
    .gemini,
    .plugin_http,
    .plugin_native,
};

pub const model_name_chain = [_]types.ProviderId{
    .llama_cpp,
    .mlx,
    .ollama,
    .ollama_passthrough,
    .lm_studio,
    .vllm,
    .codex,
    .opencode,
    .claude,
    .gemini,
    .plugin_http,
    .plugin_native,
    .anthropic,
    .openai,
};

pub const sync_round_robin_chain = [_]types.ProviderId{
    .codex,
    .opencode,
    .claude,
    .gemini,
    .ollama_passthrough,
    .ollama,
};

/// Look up a built-in model profile by name.
pub fn getModelProfile(name: []const u8) ?*const model_profiles.ModelProfile {
    return model_profiles.getProfile(name);
}

pub fn looksLikeModelPath(value: []const u8) bool {
    if (value.len == 0) return false;
    if (std.mem.indexOfScalar(u8, value, '/')) |_| return true;
    if (std.mem.indexOfScalar(u8, value, '\\')) |_| return true;

    return std.mem.endsWith(u8, value, ".gguf") or
        std.mem.endsWith(u8, value, ".bin") or
        std.mem.endsWith(u8, value, ".safetensors") or
        std.mem.endsWith(u8, value, ".mlx");
}

test "sync round robin chain includes requested providers" {
    try std.testing.expectEqual(types.ProviderId.codex, sync_round_robin_chain[0]);
    try std.testing.expectEqual(types.ProviderId.opencode, sync_round_robin_chain[1]);
    try std.testing.expectEqual(types.ProviderId.claude, sync_round_robin_chain[2]);
    try std.testing.expectEqual(types.ProviderId.gemini, sync_round_robin_chain[3]);
    try std.testing.expectEqual(types.ProviderId.ollama_passthrough, sync_round_robin_chain[4]);
    try std.testing.expectEqual(types.ProviderId.ollama, sync_round_robin_chain[5]);
}
