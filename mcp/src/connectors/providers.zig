const std = @import("std");
const env = @import("env.zig");

pub const ProviderInfo = struct {
    name: []const u8,
    display_name: []const u8,
    env_key: []const u8,
    base_url: []const u8,
    is_alias: bool,
};

pub const ProviderRegistry = struct {
    pub const providers = [_]ProviderInfo{
        .{ .name = "ollama", .display_name = "Ollama", .env_key = env.ENV_VARS.ollama.host[0], .base_url = "http://127.0.0.1:11434", .is_alias = false },
        .{ .name = "ollama_passthrough", .display_name = "Ollama Passthrough", .env_key = "ABI_OLLAMA_PASSTHROUGH_URL", .base_url = "http://127.0.0.1:11434", .is_alias = false },
        .{ .name = "openai", .display_name = "OpenAI", .env_key = env.ENV_VARS.openai.api_key[0], .base_url = "https://api.openai.com/v1", .is_alias = false },
        .{ .name = "anthropic", .display_name = "Anthropic", .env_key = env.ENV_VARS.anthropic.api_key[0], .base_url = "https://api.anthropic.com/v1", .is_alias = false },
        .{ .name = "claude", .display_name = "Claude", .env_key = env.ENV_VARS.anthropic.api_key[0], .base_url = "https://api.anthropic.com/v1", .is_alias = true },
        .{ .name = "codex", .display_name = "Codex", .env_key = env.ENV_VARS.openai.api_key[0], .base_url = "https://api.openai.com/v1", .is_alias = true },
        .{ .name = "opencode", .display_name = "OpenCode", .env_key = "ABI_OPENCODE_API_KEY", .base_url = "https://api.openai.com/v1", .is_alias = true },
        .{ .name = "gemini", .display_name = "Google Gemini", .env_key = env.ENV_VARS.gemini.api_key[0], .base_url = "https://generativelanguage.googleapis.com/v1beta", .is_alias = false },
        .{ .name = "huggingface", .display_name = "HuggingFace", .env_key = env.ENV_VARS.huggingface.api_key[0], .base_url = "https://api-inference.huggingface.co", .is_alias = false },
        .{ .name = "mistral", .display_name = "Mistral AI", .env_key = env.ENV_VARS.mistral.api_key[0], .base_url = "https://api.mistral.ai/v1", .is_alias = false },
        .{ .name = "cohere", .display_name = "Cohere", .env_key = env.ENV_VARS.cohere.api_key[0], .base_url = "https://api.cohere.ai/v1", .is_alias = false },
        .{ .name = "lm_studio", .display_name = "LM Studio", .env_key = "ABI_LM_STUDIO_HOST", .base_url = "http://localhost:1234", .is_alias = false },
        .{ .name = "vllm", .display_name = "vLLM", .env_key = "ABI_VLLM_HOST", .base_url = "http://localhost:8000", .is_alias = false },
        .{ .name = "mlx", .display_name = "MLX", .env_key = "ABI_MLX_HOST", .base_url = "http://localhost:8080", .is_alias = false },
        .{ .name = "llama_cpp", .display_name = "llama.cpp", .env_key = "ABI_LLAMA_CPP_HOST", .base_url = "http://localhost:8080", .is_alias = false },
        .{ .name = "discord", .display_name = "Discord", .env_key = env.ENV_VARS.discord.bot_token[0], .base_url = "https://discord.com/api/v10", .is_alias = false },
    };

    pub fn listAll() []const ProviderInfo {
        return &providers;
    }

    pub fn listAvailable() []const ProviderInfo {
        return &providers;
    }

    pub fn getByName(name: []const u8) ?ProviderInfo {
        for (providers) |provider| {
            if (std.mem.eql(u8, provider.name, name)) return provider;
        }
        return null;
    }
};

test "ProviderRegistry.listAll returns 16 providers" {
    const all = ProviderRegistry.listAll();
    try std.testing.expectEqual(@as(usize, 16), all.len);
}

test "ProviderRegistry keeps local-first inference providers first" {
    const all = ProviderRegistry.listAll();
    try std.testing.expect(all.len >= 3);
    try std.testing.expectEqualStrings("ollama", all[0].name);
    try std.testing.expectEqualStrings("ollama_passthrough", all[1].name);
    try std.testing.expectEqualStrings("openai", all[2].name);
}

test "ProviderRegistry.getByName finds openai" {
    const info = ProviderRegistry.getByName("openai");
    try std.testing.expect(info != null);
    try std.testing.expectEqualStrings("OpenAI", info.?.display_name);
    try std.testing.expectEqualStrings(env.ENV_VARS.openai.api_key[0], info.?.env_key);
    try std.testing.expect(!info.?.is_alias);
}

test "ProviderRegistry.getByName finds anthropic" {
    const info = ProviderRegistry.getByName("anthropic");
    try std.testing.expect(info != null);
    try std.testing.expectEqualStrings("Anthropic", info.?.display_name);
}

test "ProviderRegistry.getByName returns null for nonexistent" {
    const info = ProviderRegistry.getByName("nonexistent");
    try std.testing.expect(info == null);
}

test "ProviderRegistry.getByName identifies aliases" {
    const claude_info = ProviderRegistry.getByName("claude");
    try std.testing.expect(claude_info != null);
    try std.testing.expect(claude_info.?.is_alias);

    const codex_info = ProviderRegistry.getByName("codex");
    try std.testing.expect(codex_info != null);
    try std.testing.expect(codex_info.?.is_alias);
}

test "ProviderRegistry.listAvailable returns all providers" {
    const available = ProviderRegistry.listAvailable();
    try std.testing.expectEqual(@as(usize, 16), available.len);
}

test "ProviderRegistry env_key uses ABI-namespaced primary" {
    for (ProviderRegistry.providers) |provider| {
        if (provider.is_alias) continue;
        try std.testing.expect(std.mem.startsWith(u8, provider.env_key, "ABI_"));
    }
}

test "ProviderRegistry env keys stay aligned with ENV_VARS" {
    try std.testing.expectEqualStrings(env.ENV_VARS.openai.api_key[0], ProviderRegistry.getByName("openai").?.env_key);
    try std.testing.expectEqualStrings(env.ENV_VARS.anthropic.api_key[0], ProviderRegistry.getByName("anthropic").?.env_key);
    try std.testing.expectEqualStrings(env.ENV_VARS.gemini.api_key[0], ProviderRegistry.getByName("gemini").?.env_key);
    try std.testing.expectEqualStrings(env.ENV_VARS.huggingface.api_key[0], ProviderRegistry.getByName("huggingface").?.env_key);
    try std.testing.expectEqualStrings(env.ENV_VARS.ollama.host[0], ProviderRegistry.getByName("ollama").?.env_key);
    try std.testing.expectEqualStrings(env.ENV_VARS.mistral.api_key[0], ProviderRegistry.getByName("mistral").?.env_key);
    try std.testing.expectEqualStrings(env.ENV_VARS.cohere.api_key[0], ProviderRegistry.getByName("cohere").?.env_key);
    try std.testing.expectEqualStrings(env.ENV_VARS.discord.bot_token[0], ProviderRegistry.getByName("discord").?.env_key);
}

test {
    std.testing.refAllDecls(@This());
}
