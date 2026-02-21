const std = @import("std");
const types = @import("types.zig");

pub const all_providers = [_]types.ProviderId{
    .local_gguf,
    .llama_cpp,
    .mlx,
    .ollama,
    .lm_studio,
    .vllm,
    .plugin_http,
    .plugin_native,
};

pub const file_model_chain = [_]types.ProviderId{
    .local_gguf,
    .llama_cpp,
    .mlx,
    .ollama,
    .lm_studio,
    .vllm,
    .plugin_http,
    .plugin_native,
};

pub const model_name_chain = [_]types.ProviderId{
    .llama_cpp,
    .mlx,
    .ollama,
    .lm_studio,
    .vllm,
    .plugin_http,
    .plugin_native,
};

pub fn looksLikeModelPath(value: []const u8) bool {
    if (value.len == 0) return false;
    if (std.mem.indexOfScalar(u8, value, '/')) |_| return true;
    if (std.mem.indexOfScalar(u8, value, '\\')) |_| return true;

    return std.mem.endsWith(u8, value, ".gguf") or
        std.mem.endsWith(u8, value, ".bin") or
        std.mem.endsWith(u8, value, ".safetensors") or
        std.mem.endsWith(u8, value, ".mlx");
}
