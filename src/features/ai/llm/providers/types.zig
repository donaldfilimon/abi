const std = @import("std");

pub const ProviderId = enum {
    local_gguf,
    llama_cpp,
    mlx,
    ollama,
    lm_studio,
    vllm,
    anthropic,
    openai,
    plugin_http,
    plugin_native,
    codex,
    opencode,
    claude,
    gemini,
    ollama_passthrough,

    pub fn label(self: ProviderId) []const u8 {
        return switch (self) {
            .local_gguf => "local_gguf",
            .llama_cpp => "llama_cpp",
            .mlx => "mlx",
            .ollama => "ollama",
            .lm_studio => "lm_studio",
            .vllm => "vllm",
            .anthropic => "anthropic",
            .openai => "openai",
            .plugin_http => "plugin_http",
            .plugin_native => "plugin_native",
            .codex => "codex",
            .opencode => "opencode",
            .claude => "claude",
            .gemini => "gemini",
            .ollama_passthrough => "ollama_passthrough",
        };
    }

    pub fn fromString(value: []const u8) ?ProviderId {
        inline for (std.meta.fields(ProviderId)) |field| {
            if (std.mem.eql(u8, value, field.name)) {
                return @enumFromInt(field.value);
            }
        }
        return null;
    }
};

/// Structured chat message for multi-turn conversations.
/// Layout-compatible with connector shared.ChatMessage.
pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
};

pub const GenerateConfig = struct {
    model: []const u8,
    prompt: []const u8,
    backend: ?ProviderId = null,
    fallback: []const ProviderId = &.{},
    strict_backend: bool = false,
    plugin_id: ?[]const u8 = null,
    max_tokens: u32 = 256,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    repetition_penalty: f32 = 1.1,
    /// Optional structured messages for multi-turn conversations.
    /// When set, providers that support chat completions will use these
    /// instead of wrapping `prompt` in a single user message.
    messages: ?[]const ChatMessage = null,
    /// Optional system prompt. Used by providers that support a dedicated
    /// system message field (e.g., Anthropic).
    system_prompt: ?[]const u8 = null,
};

pub const GenerateResult = struct {
    provider: ProviderId,
    model_used: []u8,
    content: []u8,

    pub fn deinit(self: *GenerateResult, allocator: std.mem.Allocator) void {
        allocator.free(self.model_used);
        allocator.free(self.content);
        self.* = undefined;
    }
};

test "provider id includes new provider labels" {
    try std.testing.expectEqualStrings("codex", ProviderId.codex.label());
    try std.testing.expectEqualStrings("opencode", ProviderId.opencode.label());
    try std.testing.expectEqualStrings("claude", ProviderId.claude.label());
    try std.testing.expectEqualStrings("gemini", ProviderId.gemini.label());
    try std.testing.expectEqualStrings("ollama_passthrough", ProviderId.ollama_passthrough.label());
    try std.testing.expect(ProviderId.fromString("codex") != null);
    try std.testing.expect(ProviderId.fromString("gemini") != null);
}

test {
    std.testing.refAllDecls(@This());
}
