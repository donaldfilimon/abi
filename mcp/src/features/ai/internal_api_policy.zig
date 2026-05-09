const std = @import("std");

pub const Provider = enum {
    ollama,
    abi_internal,
    llama_cpp,
    lm_studio,
    vllm,
    mlx,
    openai,
    anthropic,
    gemini,
    mistral,
    cohere,
    huggingface,
    unknown,
};

pub const Resolution = struct {
    selected_model: []const u8,
    provider: Provider,
    used_fallback: bool,
    reason: []const u8,
};

pub const PolicyError = error{
    InvalidModelId,
    DisallowedProvider,
    NoAllowedModels,
};

pub const Config = struct {
    allow_trusted_fallback: bool = true,
};

pub fn providerFromModelId(model_id: []const u8) Provider {
    const slash = std.mem.indexOfScalar(u8, model_id, '/') orelse return .unknown;
    const provider = model_id[0..slash];
    if (std.mem.eql(u8, provider, "ollama")) return .ollama;
    if (std.mem.eql(u8, provider, "abi")) return .abi_internal;
    if (std.mem.eql(u8, provider, "llama_cpp")) return .llama_cpp;
    if (std.mem.eql(u8, provider, "lm_studio")) return .lm_studio;
    if (std.mem.eql(u8, provider, "vllm")) return .vllm;
    if (std.mem.eql(u8, provider, "mlx")) return .mlx;
    if (std.mem.eql(u8, provider, "openai")) return .openai;
    if (std.mem.eql(u8, provider, "anthropic")) return .anthropic;
    if (std.mem.eql(u8, provider, "gemini")) return .gemini;
    if (std.mem.eql(u8, provider, "mistral")) return .mistral;
    if (std.mem.eql(u8, provider, "cohere")) return .cohere;
    if (std.mem.eql(u8, provider, "huggingface")) return .huggingface;
    return .unknown;
}

pub fn isInternalProvider(provider: Provider) bool {
    return switch (provider) {
        .ollama, .abi_internal => true,
        else => false,
    };
}

pub fn isTrustedFallbackProvider(provider: Provider) bool {
    return switch (provider) {
        .llama_cpp, .lm_studio, .vllm, .mlx => true,
        else => false,
    };
}

pub fn resolveModel(candidates: []const []const u8, config: Config) PolicyError!Resolution {
    if (candidates.len == 0) return error.NoAllowedModels;

    var first_trusted_fallback: ?[]const u8 = null;
    var first_trusted_provider: Provider = .unknown;

    for (candidates) |model_id| {
        const provider = providerFromModelId(model_id);
        if (isInternalProvider(provider)) {
            return .{
                .selected_model = model_id,
                .provider = provider,
                .used_fallback = false,
                .reason = "internal provider selected",
            };
        }
        if (config.allow_trusted_fallback and first_trusted_fallback == null and isTrustedFallbackProvider(provider)) {
            first_trusted_fallback = model_id;
            first_trusted_provider = provider;
        }
    }

    if (first_trusted_fallback) |model_id| {
        return .{
            .selected_model = model_id,
            .provider = first_trusted_provider,
            .used_fallback = true,
            .reason = "trusted fallback provider selected",
        };
    }

    return error.DisallowedProvider;
}

test "resolve selects internal first" {
    const resolution = try resolveModel(&.{ "openai/gpt-4o", "ollama/abbeycode" }, .{});
    try std.testing.expectEqualStrings("ollama/abbeycode", resolution.selected_model);
    try std.testing.expect(!resolution.used_fallback);
}

test "resolve allows trusted fallback when enabled" {
    const resolution = try resolveModel(&.{ "openai/gpt-4o", "lm_studio/qwen2.5" }, .{ .allow_trusted_fallback = true });
    try std.testing.expectEqualStrings("lm_studio/qwen2.5", resolution.selected_model);
    try std.testing.expect(resolution.used_fallback);
}

test "resolve rejects non trusted external providers" {
    try std.testing.expectError(error.DisallowedProvider, resolveModel(&.{ "openai/gpt-4o", "anthropic/claude-3-7" }, .{}));
}
