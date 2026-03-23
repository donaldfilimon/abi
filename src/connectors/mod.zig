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
const c = struct {
    pub extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;
};

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

pub const ProviderInfo = struct {
    name: []const u8,
    display_name: []const u8,
    env_key: []const u8,
    base_url: []const u8,
    is_alias: bool,
};

pub const ProviderRegistry = struct {
    pub const providers: [16]ProviderInfo = .{
        .{ .name = "openai", .display_name = "OpenAI", .env_key = "OPENAI_API_KEY", .base_url = "https://api.openai.com/v1", .is_alias = false },
        .{ .name = "anthropic", .display_name = "Anthropic", .env_key = "ANTHROPIC_API_KEY", .base_url = "https://api.anthropic.com/v1", .is_alias = false },
        .{ .name = "claude", .display_name = "Claude", .env_key = "ANTHROPIC_API_KEY", .base_url = "https://api.anthropic.com/v1", .is_alias = true },
        .{ .name = "codex", .display_name = "Codex", .env_key = "OPENAI_API_KEY", .base_url = "https://api.openai.com/v1", .is_alias = true },
        .{ .name = "opencode", .display_name = "OpenCode", .env_key = "ABI_OPENCODE_API_KEY", .base_url = "https://api.openai.com/v1", .is_alias = true },
        .{ .name = "gemini", .display_name = "Google Gemini", .env_key = "GEMINI_API_KEY", .base_url = "https://generativelanguage.googleapis.com/v1beta", .is_alias = false },
        .{ .name = "huggingface", .display_name = "HuggingFace", .env_key = "HF_API_TOKEN", .base_url = "https://api-inference.huggingface.co", .is_alias = false },
        .{ .name = "ollama", .display_name = "Ollama", .env_key = "OLLAMA_HOST", .base_url = "http://127.0.0.1:11434", .is_alias = false },
        .{ .name = "ollama_passthrough", .display_name = "Ollama Passthrough", .env_key = "OLLAMA_PASSTHROUGH_URL", .base_url = "http://127.0.0.1:11434", .is_alias = false },
        .{ .name = "mistral", .display_name = "Mistral AI", .env_key = "MISTRAL_API_KEY", .base_url = "https://api.mistral.ai/v1", .is_alias = false },
        .{ .name = "cohere", .display_name = "Cohere", .env_key = "COHERE_API_KEY", .base_url = "https://api.cohere.ai/v1", .is_alias = false },
        .{ .name = "lm_studio", .display_name = "LM Studio", .env_key = "ABI_LM_STUDIO_HOST", .base_url = "http://localhost:1234", .is_alias = false },
        .{ .name = "vllm", .display_name = "vLLM", .env_key = "VLLM_HOST", .base_url = "http://localhost:8000", .is_alias = false },
        .{ .name = "mlx", .display_name = "MLX", .env_key = "MLX_HOST", .base_url = "http://localhost:8080", .is_alias = false },
        .{ .name = "llama_cpp", .display_name = "llama.cpp", .env_key = "LLAMA_CPP_HOST", .base_url = "http://localhost:8080", .is_alias = false },
        .{ .name = "discord", .display_name = "Discord", .env_key = "DISCORD_BOT_TOKEN", .base_url = "https://discord.com/api/v10", .is_alias = false },
    };

    pub fn listAll() []const ProviderInfo {
        return &providers;
    }

    pub fn listAvailable() []const ProviderInfo {
        return &providers;
    }

    pub fn getByName(name: []const u8) ?ProviderInfo {
        for (providers) |p| {
            if (std.mem.eql(u8, p.name, name)) return p;
        }
        return null;
    }
};

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

test "ProviderRegistry.listAll returns 16 providers" {
    const all = ProviderRegistry.listAll();
    try std.testing.expectEqual(@as(usize, 16), all.len);
}

test "ProviderRegistry.getByName finds openai" {
    const info = ProviderRegistry.getByName("openai");
    try std.testing.expect(info != null);
    try std.testing.expectEqualStrings("OpenAI", info.?.display_name);
    try std.testing.expectEqualStrings("OPENAI_API_KEY", info.?.env_key);
    try std.testing.expect(!info.?.is_alias);
}

test "ProviderRegistry.getByName finds anthropic" {
    const info = ProviderRegistry.getByName("anthropic");
    try std.testing.expect(info != null);
    try std.testing.expectEqualStrings("Anthropic", info.?.display_name);
}

test "ProviderRegistry.getByName returns null for nonexistent" {
    const info = ProviderRegistry.getByName("nonexistent");
    try std.testing.expect(info == null);
}

test "ProviderRegistry.getByName identifies aliases" {
    const claude_info = ProviderRegistry.getByName("claude");
    try std.testing.expect(claude_info != null);
    try std.testing.expect(claude_info.?.is_alias);

    const codex_info = ProviderRegistry.getByName("codex");
    try std.testing.expect(codex_info != null);
    try std.testing.expect(codex_info.?.is_alias);
}

test "ProviderRegistry.listAvailable returns all providers" {
    const available = ProviderRegistry.listAvailable();
    try std.testing.expectEqual(@as(usize, 16), available.len);
}

test {
    std.testing.refAllDecls(@This());
}
