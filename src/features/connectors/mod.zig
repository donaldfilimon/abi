//! Connectors Feature Module
//!
//! External service integrations and plugin system

const std = @import("std");

pub const openai = @import("openai.zig");
pub const ollama = @import("ollama.zig");
pub const plugin = @import("plugin.zig");

pub const ConnectorsError = error{
    NetworkError,
    InvalidResponse,
    ParseError,
    MissingApiKey,
};

pub const ProviderType = enum { ollama, openai };

pub const OllamaConfig = struct {
    host: []const u8 = "http://localhost:11434",
    model: []const u8 = "nomic-embed-text",
};

pub const OpenAIConfig = struct {
    base_url: []const u8 = "https://api.openai.com/v1",
    api_key: []const u8 = "",
    model: []const u8 = "text-embedding-3-small",
};

pub const ProviderConfig = union(ProviderType) {
    ollama: OllamaConfig,
    openai: OpenAIConfig,
};

pub fn embedText(allocator: std.mem.Allocator, config: ProviderConfig, text: []const u8) ![]f32 {
    return switch (config) {
        .ollama => |cfg| ollama.embedText(allocator, cfg.host, cfg.model, text),
        .openai => |cfg| openai.embedText(allocator, cfg.base_url, cfg.api_key, cfg.model, text),
    };
}

test {
    std.testing.refAllDecls(@This());
}
