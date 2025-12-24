const std = @import("std");

pub const CallRequest = struct {
    model: []const u8,
    prompt: []const u8,
    max_tokens: u16,
    temperature: f32 = 0.2,
};

pub const CallResult = struct {
    ok: bool,
    content: []const u8,
    tokens_in: u32 = 0,
    tokens_out: u32 = 0,
    status_code: u16 = 200,
    err_msg: ?[]const u8 = null,
};

pub const OllamaConfig = struct {
    host: []const u8 = "http://127.0.0.1:11434",
    model: []const u8 = "nomic-embed-text",
};

pub const OpenAIConfig = struct {
    base_url: []const u8 = "https://api.openai.com/v1",
    api_key: []const u8,
    model: []const u8 = "text-embedding-3-small",
};

pub const ProviderConfig = union(enum) {
    ollama: OllamaConfig,
    openai: OpenAIConfig,
};

pub fn embedText(allocator: std.mem.Allocator, config: ProviderConfig, text: []const u8) ![]f32 {
    return switch (config) {
        .ollama => |cfg| ollama.embedText(allocator, cfg.host, cfg.model, text),
        .openai => |cfg| openai.embedText(allocator, cfg, text),
    };
}

// Connector implementations
pub const hf_inference = @import("hf_inference.zig");
pub const local_scheduler = @import("local_scheduler.zig");
pub const mock = @import("mock.zig");
pub const ollama = @import("ollama.zig");
pub const openai = @import("openai.zig");
pub const plugin = @import("plugin.zig");

pub const Connector = struct {
    name: []const u8,
    init: *const fn (allocator: std.mem.Allocator) anyerror!void,
    call: *const fn (allocator: std.mem.Allocator, req: CallRequest) anyerror!CallResult,
    health: *const fn () bool,
};

pub fn getByName(name: []const u8) ?Connector {
    if (std.mem.eql(u8, name, "openai")) return openai.get();
    if (std.mem.eql(u8, name, "hf_inference")) return hf_inference.get();
    if (std.mem.eql(u8, name, "local_scheduler")) return local_scheduler.get();
    if (std.mem.eql(u8, name, "mock")) return mock.get();
    return null;
}

/// Initialize the connectors feature module
pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator; // Currently no global connector state to initialize
}

/// Deinitialize the connectors feature module
pub fn deinit() void {
    // Currently no global connector state to cleanup
}
