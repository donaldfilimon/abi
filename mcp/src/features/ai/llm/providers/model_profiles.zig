//! Built-in Model Profiles
//!
//! Pre-configured profiles for well-known models, providing context window sizes,
//! quantization defaults, VRAM requirements, and preferred provider chains.
//! Used by the router to select optimal providers for a given model name.

const std = @import("std");
const types = @import("types.zig");

pub const Quantization = enum {
    none,
    q4_k_m,
    q8_0,
    f16,

    pub fn label(self: Quantization) []const u8 {
        return switch (self) {
            .none => "none",
            .q4_k_m => "Q4_K_M",
            .q8_0 => "Q8_0",
            .f16 => "F16",
        };
    }
};

pub const ModelProfile = struct {
    /// Display name / identifier (matches config model_name).
    name: []const u8,
    /// Context window size in tokens.
    context_length: u32,
    /// Approximate parameter count in billions.
    params_billions: f32,
    /// Default quantization format.
    default_quant: Quantization,
    /// Estimated VRAM in GB at default quantization.
    vram_gb: f32,
    /// Preferred provider chain (tried in order).
    preferred_providers: []const types.ProviderId,
    /// Whether this model is suitable for local inference.
    local_capable: bool = true,
    /// Brief description.
    description: []const u8 = "",
};

/// Built-in profile: gpt-oss:20b — general-purpose open-source LLM.
pub const gpt_oss_20b = ModelProfile{
    .name = "gpt-oss:20b",
    .context_length = 8192,
    .params_billions = 20.0,
    .default_quant = .q4_k_m,
    .vram_gb = 12.0,
    .preferred_providers = &.{
        .ollama,
        .ollama_passthrough,
        .vllm,
        .llama_cpp,
    },
    .local_capable = true,
    .description = "General-purpose 20B open-source model, Q4_K_M quantized",
};

/// Built-in profile: gemma3:12b — Google's efficient instruction-tuned model.
pub const gemma3_12b = ModelProfile{
    .name = "gemma3:12b",
    .context_length = 32768,
    .params_billions = 12.0,
    .default_quant = .q4_k_m,
    .vram_gb = 8.0,
    .preferred_providers = &.{
        .ollama,
        .gemini,
        .ollama_passthrough,
        .vllm,
    },
    .local_capable = true,
    .description = "Gemma 3 12B instruction-tuned, 32K context",
};

/// All built-in profiles.
pub const builtin_profiles = [_]*const ModelProfile{
    &gpt_oss_20b,
    &gemma3_12b,
};

/// Look up a built-in model profile by name.
/// Returns null if no built-in profile matches.
pub fn getProfile(name: []const u8) ?*const ModelProfile {
    for (&builtin_profiles) |profile| {
        if (std.mem.eql(u8, profile.name, name)) return profile;
    }
    return null;
}

/// Get the preferred provider chain for a model name.
/// Returns an empty slice if no profile matches.
pub fn getProviderChain(name: []const u8) []const types.ProviderId {
    if (getProfile(name)) |profile| return profile.preferred_providers;
    return &.{};
}

/// Check if a model name has a known profile.
pub fn hasProfile(name: []const u8) bool {
    return getProfile(name) != null;
}

// ============================================================================
// Tests
// ============================================================================

test "getProfile returns gpt-oss:20b" {
    const profile = getProfile("gpt-oss:20b") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("gpt-oss:20b", profile.name);
    try std.testing.expectEqual(@as(u32, 8192), profile.context_length);
    try std.testing.expectEqual(@as(f32, 20.0), profile.params_billions);
    try std.testing.expectEqual(Quantization.q4_k_m, profile.default_quant);
}

test "getProfile returns gemma3:12b" {
    const profile = getProfile("gemma3:12b") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("gemma3:12b", profile.name);
    try std.testing.expectEqual(@as(u32, 32768), profile.context_length);
    try std.testing.expectEqual(@as(f32, 8.0), profile.vram_gb);
}

test "getProfile returns null for unknown" {
    try std.testing.expect(getProfile("nonexistent-model") == null);
}

test "getProviderChain for gpt-oss:20b" {
    const chain = getProviderChain("gpt-oss:20b");
    try std.testing.expect(chain.len == 4);
    try std.testing.expectEqual(types.ProviderId.ollama, chain[0]);
}

test "getProviderChain empty for unknown" {
    const chain = getProviderChain("unknown-model");
    try std.testing.expect(chain.len == 0);
}

test {
    std.testing.refAllDecls(@This());
}
