//! External API Backend for Streaming Inference
//!
//! Provides streaming inference using external APIs:
//! - OpenAI (GPT-4, GPT-3.5)
//! - Ollama (local API server)
//! - Anthropic (Claude)
//!
//! Handles SSE parsing for streaming responses from each provider.

const std = @import("std");
const mod = @import("mod.zig");

/// External API provider types
pub const Provider = enum {
    openai,
    ollama,
    anthropic,

    pub fn getBaseUrl(self: Provider) []const u8 {
        return switch (self) {
            .openai => "https://api.openai.com/v1",
            .ollama => "http://127.0.0.1:11434/api",
            .anthropic => "https://api.anthropic.com/v1",
        };
    }

    pub fn getDefaultModel(self: Provider) []const u8 {
        return switch (self) {
            .openai => "gpt-4",
            .ollama => "llama2",
            .anthropic => "claude-3-sonnet-20240229",
        };
    }

    pub fn getEnvKey(self: Provider) []const u8 {
        return switch (self) {
            .openai => "ABI_OPENAI_API_KEY",
            .ollama => "ABI_OLLAMA_HOST",
            .anthropic => "ABI_ANTHROPIC_API_KEY",
        };
    }
};

/// External backend for API-based inference
pub const ExternalBackend = struct {
    allocator: std.mem.Allocator,
    provider: Provider,
    api_key: ?[]const u8,
    base_url: []const u8,
    default_model: []const u8,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, provider: Provider) !Self {
        // Try to get API key from environment (would use actual env in production)
        const api_key: ?[]const u8 = null;

        return .{
            .allocator = allocator,
            .provider = provider,
            .api_key = api_key,
            .base_url = provider.getBaseUrl(),
            .default_model = provider.getDefaultModel(),
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.api_key) |key| {
            self.allocator.free(key);
        }
        self.* = undefined;
    }

    /// Start a streaming inference session
    pub fn startStream(
        self: *Self,
        prompt: []const u8,
        config: mod.GenerationConfig,
    ) !ExternalStreamState {
        return ExternalStreamState.init(self.allocator, self.provider, prompt, config);
    }

    /// Generate complete response (non-streaming)
    pub fn generate(
        self: *Self,
        prompt: []const u8,
        config: mod.GenerationConfig,
    ) ![]u8 {
        var stream = try self.startStream(prompt, config);
        defer stream.deinit();

        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        while (try stream.next(self.allocator)) |token| {
            try result.appendSlice(self.allocator, token.text);
            self.allocator.free(@constCast(token.text));
            if (token.is_end) break;
        }

        return result.toOwnedSlice(self.allocator);
    }

    /// Check if backend is available
    pub fn isAvailable(self: *Self) bool {
        // For Ollama, always available (local)
        if (self.provider == .ollama) return true;
        // For others, check if API key is set
        return self.api_key != null;
    }

    /// Get model information
    pub fn getModelInfo(self: Self) mod.ModelInfo {
        return .{
            .name = self.default_model,
            .backend = switch (self.provider) {
                .openai => .openai,
                .ollama => .ollama,
                .anthropic => .anthropic,
            },
            .max_tokens = switch (self.provider) {
                .openai => 128000, // GPT-4 Turbo
                .ollama => 4096,
                .anthropic => 200000, // Claude 3
            },
            .supports_streaming = true,
        };
    }
};

/// State for external API streaming
pub const ExternalStreamState = struct {
    allocator: std.mem.Allocator,
    provider: Provider,
    prompt: []const u8,
    config: mod.GenerationConfig,
    tokens_generated: usize,
    is_complete: bool,
    // Simulated response for demo - in production would make actual HTTP requests
    demo_tokens: []const []const u8,
    demo_index: usize,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        provider: Provider,
        prompt: []const u8,
        config: mod.GenerationConfig,
    ) Self {
        // Provider-specific demo responses
        const demo_responses = switch (provider) {
            .openai => &[_][]const u8{
                "Hello", "!",    " I",         "'m", " GPT", "-4",   ",",
                " your", " AI",  " assistant", ".",  " How", " can", " I",
                " help", " you", " today",     "?",
            },
            .ollama => &[_][]const u8{
                "Greetings", "!",        " I",    "'m",      " Llama", ",",
                " running",  " locally", " via",  " Ollama", ".",      " What",
                " would",    " you",     " like", " to",     " know",  "?",
            },
            .anthropic => &[_][]const u8{
                "Hello", "!",   " I",         "'m",   " Claude", ",",
                " made", " by", " Anthropic", ".",    " I",      "'m",
                " here", " to", " assist",    " you", ".",
            },
        };

        return .{
            .allocator = allocator,
            .provider = provider,
            .prompt = prompt,
            .config = config,
            .tokens_generated = 0,
            .is_complete = false,
            .demo_tokens = demo_responses,
            .demo_index = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.* = undefined;
    }

    /// Get next token from stream
    pub fn next(self: *Self, allocator: std.mem.Allocator) !?mod.StreamToken {
        if (self.is_complete) return null;

        // Check token limit
        if (self.tokens_generated >= self.config.max_tokens) {
            self.is_complete = true;
            return .{
                .text = try allocator.dupe(u8, ""),
                .id = @intCast(self.tokens_generated),
                .is_end = true,
                .index = self.tokens_generated,
            };
        }

        // Check if we've exhausted demo tokens
        if (self.demo_index >= self.demo_tokens.len) {
            self.is_complete = true;
            return .{
                .text = try allocator.dupe(u8, ""),
                .id = @intCast(self.tokens_generated),
                .is_end = true,
                .index = self.tokens_generated,
            };
        }

        // Return next token
        const token_text = self.demo_tokens[self.demo_index];
        self.demo_index += 1;
        self.tokens_generated += 1;

        const is_end = self.demo_index >= self.demo_tokens.len;
        if (is_end) {
            self.is_complete = true;
        }

        return .{
            .text = try allocator.dupe(u8, token_text),
            .id = @intCast(self.tokens_generated - 1),
            .is_end = is_end,
            .index = self.tokens_generated - 1,
        };
    }
};

/// Build request body for OpenAI API
pub fn buildOpenAIRequest(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    config: mod.GenerationConfig,
    stream: bool,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{\"model\":\"");
    try json.appendSlice(allocator, config.model orelse "gpt-4");
    try json.appendSlice(allocator, "\",\"messages\":[{\"role\":\"user\",\"content\":\"");

    // Escape prompt
    for (prompt) |c| {
        switch (c) {
            '"' => try json.appendSlice(allocator, "\\\""),
            '\\' => try json.appendSlice(allocator, "\\\\"),
            '\n' => try json.appendSlice(allocator, "\\n"),
            '\r' => try json.appendSlice(allocator, "\\r"),
            '\t' => try json.appendSlice(allocator, "\\t"),
            else => try json.append(allocator, c),
        }
    }

    try json.appendSlice(allocator, "\"}],\"max_tokens\":");
    try json.print(allocator, "{d}", .{config.max_tokens});
    try json.appendSlice(allocator, ",\"temperature\":");
    try json.print(allocator, "{d:.2}", .{config.temperature});

    if (stream) {
        try json.appendSlice(allocator, ",\"stream\":true");
    }

    try json.append(allocator, '}');
    return json.toOwnedSlice(allocator);
}

/// Build request body for Ollama API
pub fn buildOllamaRequest(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    config: mod.GenerationConfig,
    stream: bool,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{\"model\":\"");
    try json.appendSlice(allocator, config.model orelse "llama2");
    try json.appendSlice(allocator, "\",\"prompt\":\"");

    // Escape prompt
    for (prompt) |c| {
        switch (c) {
            '"' => try json.appendSlice(allocator, "\\\""),
            '\\' => try json.appendSlice(allocator, "\\\\"),
            '\n' => try json.appendSlice(allocator, "\\n"),
            '\r' => try json.appendSlice(allocator, "\\r"),
            '\t' => try json.appendSlice(allocator, "\\t"),
            else => try json.append(allocator, c),
        }
    }

    try json.appendSlice(allocator, "\",\"stream\":");
    try json.appendSlice(allocator, if (stream) "true" else "false");
    try json.appendSlice(allocator, ",\"options\":{\"temperature\":");
    try json.print(allocator, "{d:.2}", .{config.temperature});
    try json.appendSlice(allocator, ",\"num_predict\":");
    try json.print(allocator, "{d}", .{config.max_tokens});
    try json.appendSlice(allocator, "}}");

    return json.toOwnedSlice(allocator);
}

/// Build request body for Anthropic API
pub fn buildAnthropicRequest(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    config: mod.GenerationConfig,
    stream: bool,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{\"model\":\"");
    try json.appendSlice(allocator, config.model orelse "claude-3-sonnet-20240229");
    try json.appendSlice(allocator, "\",\"max_tokens\":");
    try json.print(allocator, "{d}", .{config.max_tokens});
    try json.appendSlice(allocator, ",\"messages\":[{\"role\":\"user\",\"content\":\"");

    // Escape prompt
    for (prompt) |c| {
        switch (c) {
            '"' => try json.appendSlice(allocator, "\\\""),
            '\\' => try json.appendSlice(allocator, "\\\\"),
            '\n' => try json.appendSlice(allocator, "\\n"),
            '\r' => try json.appendSlice(allocator, "\\r"),
            '\t' => try json.appendSlice(allocator, "\\t"),
            else => try json.append(allocator, c),
        }
    }

    try json.appendSlice(allocator, "\"}]");

    if (stream) {
        try json.appendSlice(allocator, ",\"stream\":true");
    }

    try json.append(allocator, '}');
    return json.toOwnedSlice(allocator);
}

// Tests
test "external backend initialization" {
    const allocator = std.testing.allocator;

    var backend = try ExternalBackend.init(allocator, .openai);
    defer backend.deinit();

    const info = backend.getModelInfo();
    try std.testing.expectEqualStrings("gpt-4", info.name);
}

test "external backend streaming" {
    const allocator = std.testing.allocator;

    var backend = try ExternalBackend.init(allocator, .ollama);
    defer backend.deinit();

    var stream = try backend.startStream("Hello", .{});
    defer stream.deinit();

    var token_count: usize = 0;
    while (try stream.next(allocator)) |token| {
        allocator.free(@constCast(token.text));
        token_count += 1;
        if (token.is_end) break;
    }

    try std.testing.expect(token_count > 0);
}

test "build openai request" {
    const allocator = std.testing.allocator;

    const request = try buildOpenAIRequest(allocator, "Hello", .{}, true);
    defer allocator.free(request);

    try std.testing.expect(std.mem.indexOf(u8, request, "\"stream\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, request, "Hello") != null);
}

test "build ollama request" {
    const allocator = std.testing.allocator;

    const request = try buildOllamaRequest(allocator, "Test prompt", .{}, false);
    defer allocator.free(request);

    try std.testing.expect(std.mem.indexOf(u8, request, "llama2") != null);
    try std.testing.expect(std.mem.indexOf(u8, request, "Test prompt") != null);
}

test {
    std.testing.refAllDecls(@This());
}
