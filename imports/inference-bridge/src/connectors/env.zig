const std = @import("std");

const c = struct {
    pub extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;
};

/// Read an environment variable; returns an owned copy or null if unset.
pub fn getEnvOwned(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    const name_z = allocator.dupeZ(u8, name) catch return error.OutOfMemory;
    defer allocator.free(name_z);

    const value_ptr = c.getenv(name_z.ptr);
    if (value_ptr) |ptr| {
        const value = std.mem.span(ptr);
        return allocator.dupe(u8, value) catch return error.OutOfMemory;
    }
    return null;
}

/// Return the first owned env value found in priority order.
pub fn getFirstEnvOwned(allocator: std.mem.Allocator, names: []const []const u8) !?[]u8 {
    for (names) |name| {
        if (try getEnvOwned(allocator, name)) |value| {
            return value;
        }
    }
    return null;
}

/// Canonical env var names for each provider in priority order.
pub const ENV_VARS = struct {
    pub const openai = struct {
        pub const api_key = &[_][]const u8{ "ABI_OPENAI_API_KEY", "OPENAI_API_KEY" };
        pub const base_url = &[_][]const u8{ "ABI_OPENAI_BASE_URL", "OPENAI_BASE_URL" };
        pub const model = &[_][]const u8{ "ABI_OPENAI_MODEL", "OPENAI_MODEL" };
    };
    pub const anthropic = struct {
        pub const api_key = &[_][]const u8{ "ABI_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY" };
        pub const base_url = &[_][]const u8{ "ABI_ANTHROPIC_BASE_URL", "ANTHROPIC_BASE_URL" };
        pub const model = &[_][]const u8{ "ABI_ANTHROPIC_MODEL", "ANTHROPIC_MODEL" };
    };
    pub const gemini = struct {
        pub const api_key = &[_][]const u8{ "ABI_GEMINI_API_KEY", "GEMINI_API_KEY" };
        pub const base_url = &[_][]const u8{ "ABI_GEMINI_BASE_URL", "GEMINI_BASE_URL" };
        pub const model = &[_][]const u8{ "ABI_GEMINI_MODEL", "GEMINI_MODEL" };
    };
    pub const huggingface = struct {
        pub const api_key = &[_][]const u8{ "ABI_HF_API_TOKEN", "HF_API_TOKEN", "HUGGING_FACE_HUB_TOKEN" };
        pub const base_url = &[_][]const u8{ "ABI_HF_BASE_URL", "HF_BASE_URL" };
        pub const model = &[_][]const u8{ "ABI_HF_MODEL", "HF_MODEL" };
    };
    pub const ollama = struct {
        pub const host = &[_][]const u8{ "ABI_OLLAMA_HOST", "OLLAMA_HOST" };
        pub const model = &[_][]const u8{ "ABI_OLLAMA_MODEL", "OLLAMA_MODEL" };
    };
    pub const mistral = struct {
        pub const api_key = &[_][]const u8{ "ABI_MISTRAL_API_KEY", "MISTRAL_API_KEY" };
    };
    pub const cohere = struct {
        pub const api_key = &[_][]const u8{ "ABI_COHERE_API_KEY", "COHERE_API_KEY" };
    };
    pub const discord = struct {
        pub const bot_token = &[_][]const u8{ "ABI_DISCORD_BOT_TOKEN", "DISCORD_BOT_TOKEN" };
    };
};

test "getEnvOwned returns null for unset var" {
    const result = try getEnvOwned(std.testing.allocator, "ABI_TEST_NONEXISTENT_VAR_12345");
    try std.testing.expect(result == null);
}

test "getFirstEnvOwned returns null for empty list" {
    const result = try getFirstEnvOwned(std.testing.allocator, &.{});
    try std.testing.expect(result == null);
}

test "getFirstEnvOwned returns first match in priority order" {
    const result = try getFirstEnvOwned(std.testing.allocator, &.{
        "ABI_TEST_PRIORITY_FIRST_99999",
        "ABI_TEST_PRIORITY_SECOND_99999",
    });
    try std.testing.expect(result == null);
}

test "ENV_VARS documents ABI-prefixed primary for OpenAI" {
    try std.testing.expectEqualStrings("ABI_OPENAI_API_KEY", ENV_VARS.openai.api_key[0]);
    try std.testing.expectEqualStrings("OPENAI_API_KEY", ENV_VARS.openai.api_key[1]);
}

test "ENV_VARS documents ABI-prefixed primary for Anthropic" {
    try std.testing.expectEqualStrings("ABI_ANTHROPIC_API_KEY", ENV_VARS.anthropic.api_key[0]);
    try std.testing.expectEqualStrings("ANTHROPIC_API_KEY", ENV_VARS.anthropic.api_key[1]);
}

test "ENV_VARS documents ABI-prefixed primary for Gemini" {
    try std.testing.expectEqualStrings("ABI_GEMINI_API_KEY", ENV_VARS.gemini.api_key[0]);
    try std.testing.expectEqualStrings("GEMINI_API_KEY", ENV_VARS.gemini.api_key[1]);
}

test "ENV_VARS documents ABI-prefixed primary for HuggingFace" {
    try std.testing.expectEqualStrings("ABI_HF_API_TOKEN", ENV_VARS.huggingface.api_key[0]);
    try std.testing.expectEqualStrings("HF_API_TOKEN", ENV_VARS.huggingface.api_key[1]);
    try std.testing.expectEqualStrings("HUGGING_FACE_HUB_TOKEN", ENV_VARS.huggingface.api_key[2]);
}

test "ENV_VARS documents ABI-prefixed primary for Ollama" {
    try std.testing.expectEqualStrings("ABI_OLLAMA_HOST", ENV_VARS.ollama.host[0]);
    try std.testing.expectEqualStrings("OLLAMA_HOST", ENV_VARS.ollama.host[1]);
}

test {
    std.testing.refAllDecls(@This());
}
