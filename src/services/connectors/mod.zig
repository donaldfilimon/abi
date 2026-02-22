//! Connector configuration loaders and auth helpers.
//!
//! This module provides unified access to various AI service connectors including:
//!
//! - **OpenAI**: GPT models via the Chat Completions API
//! - **Anthropic**: Claude models via the Messages API
//! - **Ollama**: Local LLM inference server
//! - **HuggingFace**: Hosted inference API
//! - **Mistral**: Mistral AI models with OpenAI-compatible API
//! - **Cohere**: Chat, embeddings, and reranking
//! - **LM Studio**: Local LLM inference with OpenAI-compatible API
//! - **vLLM**: High-throughput local LLM serving with OpenAI-compatible API
//! - **MLX**: Apple Silicon-optimized inference via mlx-lm server
//! - **Discord**: Bot integration for Discord
//!
//! ## Usage
//!
//! Each connector can be loaded from environment variables:
//!
//! ```zig
//! const connectors = @import("abi").connectors;
//!
//! // Load and create clients
//! if (try connectors.tryLoadOpenAI(allocator)) |config| {
//!     var client = try connectors.openai.Client.init(allocator, config);
//!     defer client.deinit();
//!     // Use client...
//! }
//! ```
//!
//! ## Security
//!
//! All connectors securely wipe API keys from memory using `std.crypto.secureZero`
//! before freeing to prevent memory forensics attacks.

const std = @import("std");

pub const openai = @import("openai.zig");
pub const codex = @import("codex.zig");
pub const opencode = @import("opencode.zig");
pub const claude = @import("claude.zig");
pub const gemini = @import("gemini.zig");
pub const huggingface = @import("huggingface.zig");
pub const ollama = @import("ollama.zig");
pub const ollama_passthrough = @import("ollama_passthrough.zig");
pub const local_scheduler = @import("local_scheduler.zig");
pub const discord = @import("discord/mod.zig");
pub const anthropic = @import("anthropic.zig");
pub const mistral = @import("mistral.zig");
pub const cohere = @import("cohere.zig");
pub const lm_studio = @import("lm_studio.zig");
pub const vllm = @import("vllm.zig");
pub const mlx = @import("mlx.zig");
pub const llama_cpp = @import("llama_cpp.zig");
pub const shared = @import("shared.zig");

var initialized: bool = false;

/// Initialize the connectors subsystem (idempotent; no-op if already initialized).
pub fn init(_: std.mem.Allocator) !void {
    initialized = true;
}

/// Tear down the connectors subsystem; safe to call multiple times.
pub fn deinit() void {
    initialized = false;
}

/// Returns true; connectors are always available when this module is compiled in.
pub fn isEnabled() bool {
    return true;
}

/// Returns true after `init()` has been called.
pub fn isInitialized() bool {
    return initialized;
}

const builtin = @import("builtin");

// libc import for environment access - required for Zig 0.16
const c = @cImport(@cInclude("stdlib.h"));

/// Read environment variable by name; returns owned slice or null if unset. Caller must free.
pub fn getEnvOwned(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    // Zig 0.16: Environment access via libc getenv (build links libc)
    const name_z = allocator.dupeZ(u8, name) catch return error.OutOfMemory;
    defer allocator.free(name_z);

    const value_ptr = c.getenv(name_z.ptr);
    if (value_ptr) |ptr| {
        const value = std.mem.span(ptr);
        return allocator.dupe(u8, value) catch return error.OutOfMemory;
    }
    return null;
}

pub fn getFirstEnvOwned(allocator: std.mem.Allocator, names: []const []const u8) !?[]u8 {
    for (names) |name| {
        if (try getEnvOwned(allocator, name)) |value| {
            return value;
        }
    }
    return null;
}

pub const AuthHeader = struct {
    value: []u8,

    pub fn header(self: *const AuthHeader) std.http.Header {
        return .{ .name = "authorization", .value = self.value };
    }

    pub fn deinit(self: *AuthHeader, allocator: std.mem.Allocator) void {
        // Securely wipe auth token before freeing to prevent memory forensics
        std.crypto.secureZero(u8, self.value);
        allocator.free(self.value);
        self.* = undefined;
    }
};

pub fn buildBearerHeader(allocator: std.mem.Allocator, token: []const u8) !AuthHeader {
    const value = try std.fmt.allocPrint(allocator, "Bearer {s}", .{token});
    return .{ .value = value };
}

pub fn loadOpenAI(allocator: std.mem.Allocator) !openai.Config {
    return openai.loadFromEnv(allocator);
}

pub fn tryLoadOpenAI(allocator: std.mem.Allocator) !?openai.Config {
    return openai.loadFromEnv(allocator) catch |err| switch (err) {
        openai.OpenAIError.MissingApiKey => null,
        else => return err,
    };
}

pub fn loadCodex(allocator: std.mem.Allocator) !codex.Config {
    return codex.loadFromEnv(allocator);
}

pub fn tryLoadCodex(allocator: std.mem.Allocator) !?codex.Config {
    return codex.loadFromEnv(allocator) catch |err| switch (err) {
        codex.CodexError.MissingApiKey => null,
        else => return err,
    };
}

pub fn loadOpenCode(allocator: std.mem.Allocator) !opencode.Config {
    return opencode.loadFromEnv(allocator);
}

pub fn tryLoadOpenCode(allocator: std.mem.Allocator) !?opencode.Config {
    return opencode.loadFromEnv(allocator) catch |err| switch (err) {
        opencode.OpenCodeError.MissingApiKey => null,
        else => return err,
    };
}

pub fn loadClaude(allocator: std.mem.Allocator) !claude.Config {
    return claude.loadFromEnv(allocator);
}

pub fn tryLoadClaude(allocator: std.mem.Allocator) !?claude.Config {
    return claude.loadFromEnv(allocator) catch |err| switch (err) {
        claude.ClaudeError.MissingApiKey => null,
        else => return err,
    };
}

pub fn loadGemini(allocator: std.mem.Allocator) !gemini.Config {
    return gemini.loadFromEnv(allocator);
}

pub fn tryLoadGemini(allocator: std.mem.Allocator) !?gemini.Config {
    return gemini.loadFromEnv(allocator) catch |err| switch (err) {
        gemini.GeminiError.MissingApiKey => null,
        else => return err,
    };
}

pub fn loadHuggingFace(allocator: std.mem.Allocator) !huggingface.Config {
    return huggingface.loadFromEnv(allocator);
}

pub fn tryLoadHuggingFace(allocator: std.mem.Allocator) !?huggingface.Config {
    return huggingface.loadFromEnv(allocator) catch |err| switch (err) {
        huggingface.HuggingFaceError.MissingApiToken => null,
        else => return err,
    };
}

pub fn loadOllama(allocator: std.mem.Allocator) !ollama.Config {
    return ollama.loadFromEnv(allocator);
}

pub fn tryLoadOllama(allocator: std.mem.Allocator) !?ollama.Config {
    return ollama.loadFromEnv(allocator) catch null;
}

pub fn loadOllamaPassthrough(allocator: std.mem.Allocator) !ollama_passthrough.Config {
    return ollama_passthrough.loadFromEnv(allocator);
}

pub fn tryLoadOllamaPassthrough(allocator: std.mem.Allocator) !?ollama_passthrough.Config {
    return ollama_passthrough.loadFromEnv(allocator) catch null;
}

pub fn loadLocalScheduler(allocator: std.mem.Allocator) !local_scheduler.Config {
    return local_scheduler.loadFromEnv(allocator);
}

pub fn loadDiscord(allocator: std.mem.Allocator) !discord.Config {
    return discord.loadFromEnv(allocator);
}

pub fn tryLoadDiscord(allocator: std.mem.Allocator) !?discord.Config {
    return discord.loadFromEnv(allocator) catch |err| switch (err) {
        discord.DiscordError.MissingBotToken => null,
        else => return err,
    };
}

pub fn loadAnthropic(allocator: std.mem.Allocator) !anthropic.Config {
    return anthropic.loadFromEnv(allocator);
}

pub fn tryLoadAnthropic(allocator: std.mem.Allocator) !?anthropic.Config {
    return anthropic.loadFromEnv(allocator) catch |err| switch (err) {
        anthropic.AnthropicError.MissingApiKey => null,
        else => return err,
    };
}

pub fn loadMistral(allocator: std.mem.Allocator) !mistral.Config {
    return mistral.loadFromEnv(allocator);
}

pub fn tryLoadMistral(allocator: std.mem.Allocator) !?mistral.Config {
    return mistral.loadFromEnv(allocator) catch |err| switch (err) {
        mistral.MistralError.MissingApiKey => null,
        else => return err,
    };
}

pub fn loadCohere(allocator: std.mem.Allocator) !cohere.Config {
    return cohere.loadFromEnv(allocator);
}

pub fn tryLoadCohere(allocator: std.mem.Allocator) !?cohere.Config {
    return cohere.loadFromEnv(allocator) catch |err| switch (err) {
        cohere.CohereError.MissingApiKey => null,
        else => return err,
    };
}

pub fn loadLMStudio(allocator: std.mem.Allocator) !lm_studio.Config {
    return lm_studio.loadFromEnv(allocator);
}

pub fn tryLoadLMStudio(allocator: std.mem.Allocator) !?lm_studio.Config {
    return lm_studio.loadFromEnv(allocator) catch null;
}

pub fn loadVLLM(allocator: std.mem.Allocator) !vllm.Config {
    return vllm.loadFromEnv(allocator);
}

pub fn tryLoadVLLM(allocator: std.mem.Allocator) !?vllm.Config {
    return vllm.loadFromEnv(allocator) catch null;
}

pub fn loadMLX(allocator: std.mem.Allocator) !mlx.Config {
    return mlx.loadFromEnv(allocator);
}

pub fn tryLoadMLX(allocator: std.mem.Allocator) !?mlx.Config {
    return mlx.loadFromEnv(allocator) catch null;
}

pub fn loadLlamaCpp(allocator: std.mem.Allocator) !llama_cpp.Config {
    return llama_cpp.loadFromEnv(allocator);
}

pub fn tryLoadLlamaCpp(allocator: std.mem.Allocator) !?llama_cpp.Config {
    return llama_cpp.loadFromEnv(allocator) catch null;
}

test "connectors init toggles state" {
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}

test "getEnvOwned returns null for unset var" {
    const result = try getEnvOwned(std.testing.allocator, "ABI_TEST_NONEXISTENT_VAR_12345");
    try std.testing.expect(result == null);
}

test "getFirstEnvOwned returns null for empty list" {
    const result = try getFirstEnvOwned(std.testing.allocator, &.{});
    try std.testing.expect(result == null);
}

test "buildBearerHeader formats correctly" {
    var auth = try buildBearerHeader(std.testing.allocator, "test-token-123");
    defer auth.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("Bearer test-token-123", auth.value);
}

test "AuthHeader.header returns HTTP header" {
    var auth = try buildBearerHeader(std.testing.allocator, "sk-abc");
    defer auth.deinit(std.testing.allocator);
    const hdr = auth.header();
    try std.testing.expectEqualStrings("authorization", hdr.name);
    try std.testing.expectEqualStrings("Bearer sk-abc", hdr.value);
}

test "isEnabled always returns true" {
    try std.testing.expect(isEnabled());
}
