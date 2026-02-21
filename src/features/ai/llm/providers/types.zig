const std = @import("std");

pub const ProviderId = enum {
    local_gguf,
    llama_cpp,
    mlx,
    ollama,
    lm_studio,
    vllm,
    plugin_http,
    plugin_native,

    pub fn label(self: ProviderId) []const u8 {
        return switch (self) {
            .local_gguf => "local_gguf",
            .llama_cpp => "llama_cpp",
            .mlx => "mlx",
            .ollama => "ollama",
            .lm_studio => "lm_studio",
            .vllm => "vllm",
            .plugin_http => "plugin_http",
            .plugin_native => "plugin_native",
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
