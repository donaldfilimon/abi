const std = @import("std");

pub const Allocator = std.mem.Allocator;

pub const ProviderType = enum { ollama, openai };

pub const OllamaConfig = struct {
    host: []const u8 = "http://localhost:11434",
    model: []const u8 = "nomic-embed-text",
};

pub const OpenAIConfig = struct {
    base_url: []const u8 = "https://api.openai.com/v1",
    api_key: []const u8, // required
    model: []const u8 = "text-embedding-3-small",
};

pub const ProviderConfig = union(ProviderType) {
    ollama: OllamaConfig,
    openai: OpenAIConfig,
};

pub const plugin = @import("plugin.zig");

pub const ConnectorsError = error{
    NetworkError,
    InvalidResponse,
    ParseError,
    MissingApiKey,
};

pub fn embedText(allocator: Allocator, config: ProviderConfig, text: []const u8) ConnectorsError![]f32 {
    return switch (config) {
        .ollama => |cfg| ollama_embed(allocator, cfg, text),
        .openai => |cfg| openai_embed(allocator, cfg, text),
    };
}

fn ollama_embed(allocator: Allocator, cfg: OllamaConfig, text: []const u8) ConnectorsError![]f32 {
    const ollama = @import("ollama.zig");
    return ollama.embedText(allocator, cfg.host, cfg.model, text);
}

fn openai_embed(allocator: Allocator, cfg: OpenAIConfig, text: []const u8) ConnectorsError![]f32 {
    const openai = @import("openai.zig");
    return openai.embedText(allocator, cfg.base_url, cfg.api_key, cfg.model, text);
}
