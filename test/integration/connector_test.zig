//! Integration Tests: Connectors
//!
//! Verifies connector module exports, client types, shared types,
//! and configuration loading (without requiring actual API keys).

const std = @import("std");
const abi = @import("abi");

const connectors = abi.connectors;

// ============================================================================
// Module Exports
// ============================================================================

test "connectors: module exports sub-connectors" {
    _ = connectors.openai;
    _ = connectors.anthropic;
    _ = connectors.claude;
    _ = connectors.gemini;
    _ = connectors.ollama;
    _ = connectors.huggingface;
    _ = connectors.mistral;
    _ = connectors.cohere;
    _ = connectors.lm_studio;
    _ = connectors.vllm;
    _ = connectors.mlx;
    _ = connectors.llama_cpp;
    _ = connectors.codex;
    _ = connectors.opencode;
    _ = connectors.discord;
    _ = connectors.shared;
    _ = connectors.ollama_passthrough;
    _ = connectors.local_scheduler;
}

// ============================================================================
// Shared Types
// ============================================================================

test "connectors: shared ChatMessage type" {
    const msg = connectors.shared.ChatMessage{
        .role = connectors.shared.Role.USER,
        .content = "Hello, world!",
    };

    try std.testing.expectEqualStrings("user", msg.role);
    try std.testing.expectEqualStrings("Hello, world!", msg.content);
}

test "connectors: shared Role constants" {
    try std.testing.expectEqualStrings("system", connectors.shared.Role.SYSTEM);
    try std.testing.expectEqualStrings("user", connectors.shared.Role.USER);
    try std.testing.expectEqualStrings("assistant", connectors.shared.Role.ASSISTANT);
    try std.testing.expectEqualStrings("function", connectors.shared.Role.FUNCTION);
    try std.testing.expectEqualStrings("tool", connectors.shared.Role.TOOL);
}

// ============================================================================
// Init / Deinit Lifecycle
// ============================================================================

test "connectors: init and deinit lifecycle" {
    try connectors.init(std.testing.allocator);
    try std.testing.expect(connectors.isInitialized());

    connectors.deinit();
    try std.testing.expect(!connectors.isInitialized());
}

test "connectors: isEnabled always true" {
    try std.testing.expect(connectors.isEnabled());
}

// ============================================================================
// Auth Helpers
// ============================================================================

test "connectors: buildBearerHeader" {
    var auth = try connectors.buildBearerHeader(std.testing.allocator, "test-key-xyz");
    defer auth.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("Bearer test-key-xyz", auth.value);

    const hdr = auth.header();
    try std.testing.expectEqualStrings("authorization", hdr.name);
}

test "connectors: AuthHeader type exists" {
    const AuthHeader = connectors.AuthHeader;
    try std.testing.expect(@sizeOf(AuthHeader) > 0);
}

// ============================================================================
// Environment-based Config Loading (no keys set)
// ============================================================================

test "connectors: tryLoadOpenAI returns null without key" {
    const result = try connectors.tryLoadOpenAI(std.testing.allocator);
    try std.testing.expect(result == null);
}

test "connectors: tryLoadClaude returns null without key" {
    const result = try connectors.tryLoadClaude(std.testing.allocator);
    try std.testing.expect(result == null);
}

test "connectors: tryLoadAnthropic returns null without key" {
    const result = try connectors.tryLoadAnthropic(std.testing.allocator);
    try std.testing.expect(result == null);
}

test "connectors: tryLoadGemini returns null without key" {
    const result = try connectors.tryLoadGemini(std.testing.allocator);
    try std.testing.expect(result == null);
}

test "connectors: tryLoadHuggingFace returns null without key" {
    const result = try connectors.tryLoadHuggingFace(std.testing.allocator);
    try std.testing.expect(result == null);
}

test "connectors: tryLoadMistral returns null without key" {
    const result = try connectors.tryLoadMistral(std.testing.allocator);
    try std.testing.expect(result == null);
}

test "connectors: tryLoadCohere returns null without key" {
    const result = try connectors.tryLoadCohere(std.testing.allocator);
    try std.testing.expect(result == null);
}

test "connectors: tryLoadDiscord returns null without key" {
    const result = try connectors.tryLoadDiscord(std.testing.allocator);
    try std.testing.expect(result == null);
}

test "connectors: tryLoadCodex returns null without key" {
    const result = try connectors.tryLoadCodex(std.testing.allocator);
    try std.testing.expect(result == null);
}

test "connectors: tryLoadOpenCode returns null without key" {
    const result = try connectors.tryLoadOpenCode(std.testing.allocator);
    try std.testing.expect(result == null);
}

// ============================================================================
// Environment Helpers
// ============================================================================

test "connectors: getEnvOwned returns null for unset var" {
    const result = try connectors.getEnvOwned(std.testing.allocator, "ABI_INTEGRATION_TEST_NONEXISTENT_42");
    try std.testing.expect(result == null);
}

test "connectors: getFirstEnvOwned returns null for empty list" {
    const result = try connectors.getFirstEnvOwned(std.testing.allocator, &.{});
    try std.testing.expect(result == null);
}

test "connectors: getFirstEnvOwned returns null for unset vars" {
    const result = try connectors.getFirstEnvOwned(std.testing.allocator, &.{
        "ABI_CONNECTOR_TEST_X_999",
        "ABI_CONNECTOR_TEST_Y_999",
    });
    try std.testing.expect(result == null);
}

// ============================================================================
// Local Connector tryLoad (no external service required)
// ============================================================================

test "connectors: tryLoadOllama returns non-null config with default host" {
    // Ollama doesn't require an API key — loadFromEnv uses a default host.
    // tryLoadOllama should return a config or null depending on env, but never error.
    const result = try connectors.tryLoadOllama(std.testing.allocator);
    if (result) |cfg| {
        var config = cfg;
        defer config.deinit(std.testing.allocator);
        // Default host should be non-empty
        try std.testing.expect(config.host.len > 0);
    }
}

test "connectors: tryLoadLMStudio returns config or null" {
    const result = try connectors.tryLoadLMStudio(std.testing.allocator);
    if (result) |cfg| {
        var config = cfg;
        defer config.deinit(std.testing.allocator);
        try std.testing.expect(config.host.len > 0);
    }
}

test "connectors: tryLoadVLLM returns config or null" {
    const result = try connectors.tryLoadVLLM(std.testing.allocator);
    if (result) |cfg| {
        var config = cfg;
        defer config.deinit(std.testing.allocator);
        try std.testing.expect(config.host.len > 0);
    }
}

test "connectors: tryLoadMLX returns config or null" {
    const result = try connectors.tryLoadMLX(std.testing.allocator);
    if (result) |cfg| {
        var config = cfg;
        defer config.deinit(std.testing.allocator);
        try std.testing.expect(config.host.len > 0);
    }
}

test "connectors: tryLoadLlamaCpp returns config or null" {
    const result = try connectors.tryLoadLlamaCpp(std.testing.allocator);
    if (result) |cfg| {
        var config = cfg;
        defer config.deinit(std.testing.allocator);
        try std.testing.expect(config.host.len > 0);
    }
}

test "connectors: tryLoadOllamaPassthrough returns config or null" {
    const result = try connectors.tryLoadOllamaPassthrough(std.testing.allocator);
    if (result) |cfg| {
        var config = cfg;
        defer config.deinit(std.testing.allocator);
        try std.testing.expect(config.host.len > 0);
    }
}

// ============================================================================
// ProviderRegistry: Discovery and Enumeration
// ============================================================================

test "connectors: ProviderRegistry lists exactly 16 providers" {
    const all = connectors.ProviderRegistry.listAll();
    try std.testing.expectEqual(@as(usize, 16), all.len);
}

test "connectors: ProviderRegistry getByName finds all 16 providers by name" {
    const expected_names = [_][]const u8{
        "openai",
        "anthropic",
        "claude",
        "codex",
        "opencode",
        "gemini",
        "huggingface",
        "ollama",
        "ollama_passthrough",
        "mistral",
        "cohere",
        "lm_studio",
        "vllm",
        "mlx",
        "llama_cpp",
        "discord",
    };
    for (expected_names) |name| {
        const info = connectors.ProviderRegistry.getByName(name);
        try std.testing.expect(info != null);
    }
}

test "connectors: ProviderRegistry getByName returns null for unknown provider" {
    try std.testing.expect(connectors.ProviderRegistry.getByName("grok") == null);
    try std.testing.expect(connectors.ProviderRegistry.getByName("") == null);
    try std.testing.expect(connectors.ProviderRegistry.getByName("OPENAI") == null); // case-sensitive
}

test "connectors: ProviderRegistry all providers have non-empty env_key" {
    for (connectors.ProviderRegistry.providers) |p| {
        try std.testing.expect(p.env_key.len > 0);
    }
}

test "connectors: ProviderRegistry all providers have non-empty base_url" {
    for (connectors.ProviderRegistry.providers) |p| {
        try std.testing.expect(p.base_url.len > 0);
    }
}

test "connectors: ProviderRegistry all providers have non-empty display_name" {
    for (connectors.ProviderRegistry.providers) |p| {
        try std.testing.expect(p.display_name.len > 0);
    }
}

test "connectors: ProviderRegistry all providers have non-empty name" {
    for (connectors.ProviderRegistry.providers) |p| {
        try std.testing.expect(p.name.len > 0);
    }
}

test "connectors: ProviderRegistry all env_keys use ABI_ prefix for non-aliases" {
    for (connectors.ProviderRegistry.providers) |p| {
        if (p.is_alias) continue;
        try std.testing.expect(std.mem.startsWith(u8, p.env_key, "ABI_"));
    }
}

test "connectors: ProviderRegistry identifies exactly 3 aliases" {
    var alias_count: usize = 0;
    for (connectors.ProviderRegistry.providers) |p| {
        if (p.is_alias) alias_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), alias_count);
}

test "connectors: ProviderRegistry alias providers are claude, codex, opencode" {
    const claude_info = connectors.ProviderRegistry.getByName("claude");
    try std.testing.expect(claude_info != null);
    try std.testing.expect(claude_info.?.is_alias);

    const codex_info = connectors.ProviderRegistry.getByName("codex");
    try std.testing.expect(codex_info != null);
    try std.testing.expect(codex_info.?.is_alias);

    const opencode_info = connectors.ProviderRegistry.getByName("opencode");
    try std.testing.expect(opencode_info != null);
    try std.testing.expect(opencode_info.?.is_alias);
}

test "connectors: ProviderRegistry non-alias providers are not marked as aliases" {
    const non_aliases = [_][]const u8{
        "openai",             "anthropic", "gemini",  "huggingface", "ollama",
        "ollama_passthrough", "mistral",   "cohere",  "lm_studio",   "vllm",
        "mlx",                "llama_cpp", "discord",
    };
    for (non_aliases) |name| {
        const info = connectors.ProviderRegistry.getByName(name);
        try std.testing.expect(info != null);
        try std.testing.expect(!info.?.is_alias);
    }
}

test "connectors: ProviderRegistry listAvailable returns provider list" {
    // In the live module, listAvailable returns all 16; in stub it returns 0.
    // Either way it should not crash and should return a valid slice.
    const available = connectors.ProviderRegistry.listAvailable();
    try std.testing.expect(available.len <= 16);
}

// ============================================================================
// Connector isAvailable: Sub-module Availability Checks
// ============================================================================

test "connectors: all sub-connectors expose isAvailable" {
    // Each connector has an isAvailable() function. Without API keys or
    // local servers, most should return false. Verify they are callable.
    _ = connectors.openai.isAvailable();
    _ = connectors.anthropic.isAvailable();
    _ = connectors.claude.isAvailable();
    _ = connectors.gemini.isAvailable();
    _ = connectors.ollama.isAvailable();
    _ = connectors.huggingface.isAvailable();
    _ = connectors.mistral.isAvailable();
    _ = connectors.cohere.isAvailable();
    _ = connectors.lm_studio.isAvailable();
    _ = connectors.vllm.isAvailable();
    _ = connectors.mlx.isAvailable();
    _ = connectors.llama_cpp.isAvailable();
    _ = connectors.codex.isAvailable();
    _ = connectors.opencode.isAvailable();
    _ = connectors.discord.isAvailable();
    _ = connectors.ollama_passthrough.isAvailable();
}

// ============================================================================
// Connector Config Types: Existence and Defaults
// ============================================================================

test "connectors: Config types are accessible for all connectors" {
    // Verify all connectors export a Config type that can be referenced
    try std.testing.expect(@sizeOf(connectors.openai.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.anthropic.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.claude.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.gemini.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.ollama.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.huggingface.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.mistral.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.cohere.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.lm_studio.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.vllm.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.mlx.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.llama_cpp.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.codex.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.opencode.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.discord.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.local_scheduler.Config) > 0);
    try std.testing.expect(@sizeOf(connectors.ollama_passthrough.Config) > 0);
}

test "connectors: Client types are accessible for key connectors" {
    // Verify connectors export a Client type
    try std.testing.expect(@sizeOf(connectors.openai.Client) > 0);
    try std.testing.expect(@sizeOf(connectors.anthropic.Client) > 0);
    try std.testing.expect(@sizeOf(connectors.ollama.Client) > 0);
    try std.testing.expect(@sizeOf(connectors.lm_studio.Client) > 0);
    try std.testing.expect(@sizeOf(connectors.vllm.Client) > 0);
    try std.testing.expect(@sizeOf(connectors.mlx.Client) > 0);
    try std.testing.expect(@sizeOf(connectors.llama_cpp.Client) > 0);
    try std.testing.expect(@sizeOf(connectors.discord.Client) > 0);
}

test "connectors: Message types alias shared.ChatMessage" {
    // Most connectors re-export shared.ChatMessage as their Message type
    try std.testing.expectEqual(@sizeOf(connectors.shared.ChatMessage), @sizeOf(connectors.openai.Message));
    try std.testing.expectEqual(@sizeOf(connectors.shared.ChatMessage), @sizeOf(connectors.anthropic.Message));
    try std.testing.expectEqual(@sizeOf(connectors.shared.ChatMessage), @sizeOf(connectors.ollama.Message));
    try std.testing.expectEqual(@sizeOf(connectors.shared.ChatMessage), @sizeOf(connectors.mistral.Message));
    try std.testing.expectEqual(@sizeOf(connectors.shared.ChatMessage), @sizeOf(connectors.gemini.Message));
    try std.testing.expectEqual(@sizeOf(connectors.shared.ChatMessage), @sizeOf(connectors.lm_studio.Message));
    try std.testing.expectEqual(@sizeOf(connectors.shared.ChatMessage), @sizeOf(connectors.vllm.Message));
    try std.testing.expectEqual(@sizeOf(connectors.shared.ChatMessage), @sizeOf(connectors.llama_cpp.Message));
}

// ============================================================================
// ENV_VARS: Env Var Name Consistency
// ============================================================================

test "connectors: ENV_VARS all entries are non-empty strings" {
    // OpenAI
    for (connectors.ENV_VARS.openai.api_key) |v| try std.testing.expect(v.len > 0);
    for (connectors.ENV_VARS.openai.base_url) |v| try std.testing.expect(v.len > 0);
    for (connectors.ENV_VARS.openai.model) |v| try std.testing.expect(v.len > 0);

    // Anthropic
    for (connectors.ENV_VARS.anthropic.api_key) |v| try std.testing.expect(v.len > 0);
    for (connectors.ENV_VARS.anthropic.base_url) |v| try std.testing.expect(v.len > 0);
    for (connectors.ENV_VARS.anthropic.model) |v| try std.testing.expect(v.len > 0);

    // Gemini
    for (connectors.ENV_VARS.gemini.api_key) |v| try std.testing.expect(v.len > 0);
    for (connectors.ENV_VARS.gemini.base_url) |v| try std.testing.expect(v.len > 0);
    for (connectors.ENV_VARS.gemini.model) |v| try std.testing.expect(v.len > 0);

    // HuggingFace
    for (connectors.ENV_VARS.huggingface.api_key) |v| try std.testing.expect(v.len > 0);
    for (connectors.ENV_VARS.huggingface.base_url) |v| try std.testing.expect(v.len > 0);
    for (connectors.ENV_VARS.huggingface.model) |v| try std.testing.expect(v.len > 0);

    // Ollama
    for (connectors.ENV_VARS.ollama.host) |v| try std.testing.expect(v.len > 0);
    for (connectors.ENV_VARS.ollama.model) |v| try std.testing.expect(v.len > 0);

    // Mistral
    for (connectors.ENV_VARS.mistral.api_key) |v| try std.testing.expect(v.len > 0);

    // Cohere
    for (connectors.ENV_VARS.cohere.api_key) |v| try std.testing.expect(v.len > 0);

    // Discord
    for (connectors.ENV_VARS.discord.bot_token) |v| try std.testing.expect(v.len > 0);
}

test "connectors: ENV_VARS primary entries all start with ABI_ prefix" {
    try std.testing.expect(std.mem.startsWith(u8, connectors.ENV_VARS.openai.api_key[0], "ABI_"));
    try std.testing.expect(std.mem.startsWith(u8, connectors.ENV_VARS.openai.base_url[0], "ABI_"));
    try std.testing.expect(std.mem.startsWith(u8, connectors.ENV_VARS.openai.model[0], "ABI_"));
    try std.testing.expect(std.mem.startsWith(u8, connectors.ENV_VARS.anthropic.api_key[0], "ABI_"));
    try std.testing.expect(std.mem.startsWith(u8, connectors.ENV_VARS.anthropic.base_url[0], "ABI_"));
    try std.testing.expect(std.mem.startsWith(u8, connectors.ENV_VARS.anthropic.model[0], "ABI_"));
    try std.testing.expect(std.mem.startsWith(u8, connectors.ENV_VARS.gemini.api_key[0], "ABI_"));
    try std.testing.expect(std.mem.startsWith(u8, connectors.ENV_VARS.huggingface.api_key[0], "ABI_"));
    try std.testing.expect(std.mem.startsWith(u8, connectors.ENV_VARS.ollama.host[0], "ABI_"));
    try std.testing.expect(std.mem.startsWith(u8, connectors.ENV_VARS.mistral.api_key[0], "ABI_"));
    try std.testing.expect(std.mem.startsWith(u8, connectors.ENV_VARS.cohere.api_key[0], "ABI_"));
    try std.testing.expect(std.mem.startsWith(u8, connectors.ENV_VARS.discord.bot_token[0], "ABI_"));
}

// ============================================================================
// Shared Error and Type Exports
// ============================================================================

test "connectors: ConnectorError type is accessible via shared" {
    // Verify the shared error type exists and has expected variants
    const err: connectors.shared.ConnectorError = connectors.shared.ConnectorError.MissingApiKey;
    try std.testing.expect(err == connectors.shared.ConnectorError.MissingApiKey);
    _ = connectors.shared.ConnectorError.ApiRequestFailed;
    _ = connectors.shared.ConnectorError.InvalidResponse;
    _ = connectors.shared.ConnectorError.RateLimitExceeded;
    _ = connectors.shared.ConnectorError.Timeout;
    _ = connectors.shared.ConnectorError.OutOfMemory;
}

test "connectors: ProviderError type is accessible via shared" {
    _ = connectors.shared.ProviderError.MissingAuth;
    _ = connectors.shared.ProviderError.ApiRequestFailed;
    _ = connectors.shared.ProviderError.RateLimitExceeded;
    _ = connectors.shared.ProviderError.InvalidResponse;
    _ = connectors.shared.ProviderError.Timeout;
    _ = connectors.shared.ProviderError.NetworkError;
    _ = connectors.shared.ProviderError.ModelNotFound;
    _ = connectors.shared.ProviderError.FeatureDisabled;
}

test "connectors: shared ProviderInfo struct can be constructed" {
    const info = connectors.shared.ProviderInfo{
        .name = "test-provider",
        .is_available = false,
        .env_key = "ABI_TEST_KEY",
    };
    try std.testing.expectEqualStrings("test-provider", info.name);
    try std.testing.expect(!info.is_available);
    try std.testing.expectEqualStrings("ABI_TEST_KEY", info.env_key);
}

// ============================================================================
// Shared OpenAI-Compatible Types
// ============================================================================

test "connectors: OpenAICompatConfig has expected defaults" {
    const allocator = std.testing.allocator;
    var config = connectors.shared.OpenAICompatConfig{
        .host = try allocator.dupe(u8, "http://localhost:9999"),
    };
    defer config.deinit(allocator);

    try std.testing.expect(config.api_key == null);
    try std.testing.expectEqualStrings("default", config.model);
    try std.testing.expect(!config.model_owned);
    try std.testing.expectEqual(@as(u32, 120_000), config.timeout_ms);
}

test "connectors: OpenAICompatChatRequest has expected defaults" {
    const msgs = [_]connectors.shared.ChatMessage{
        .{ .role = "user", .content = "test" },
    };
    const req = connectors.shared.OpenAICompatChatRequest{
        .model = "test-model",
        .messages = &msgs,
    };
    try std.testing.expectEqual(@as(f32, 0.7), req.temperature);
    try std.testing.expectEqual(@as(f32, 1.0), req.top_p);
    try std.testing.expect(req.max_tokens == null);
    try std.testing.expect(!req.stream);
}

// ============================================================================
// Init/Deinit Robustness
// ============================================================================

test "connectors: double init is safe" {
    try connectors.init(std.testing.allocator);
    try std.testing.expect(connectors.isInitialized());

    // Second init should not crash
    try connectors.init(std.testing.allocator);
    try std.testing.expect(connectors.isInitialized());

    connectors.deinit();
    try std.testing.expect(!connectors.isInitialized());
}

test "connectors: deinit without init is safe" {
    // Ensure deinit on an uninitialized module does not crash
    connectors.deinit();
    try std.testing.expect(!connectors.isInitialized());
}

test "connectors: init-deinit-init cycle works" {
    try connectors.init(std.testing.allocator);
    try std.testing.expect(connectors.isInitialized());

    connectors.deinit();
    try std.testing.expect(!connectors.isInitialized());

    try connectors.init(std.testing.allocator);
    try std.testing.expect(connectors.isInitialized());

    connectors.deinit();
    try std.testing.expect(!connectors.isInitialized());
}

// ============================================================================
// ProviderRegistry: Cross-referencing with ENV_VARS
// ============================================================================

test "connectors: ProviderRegistry env_key matches ENV_VARS for openai" {
    const reg_info = connectors.ProviderRegistry.getByName("openai").?;
    try std.testing.expectEqualStrings(connectors.ENV_VARS.openai.api_key[0], reg_info.env_key);
}

test "connectors: ProviderRegistry env_key matches ENV_VARS for anthropic" {
    const reg_info = connectors.ProviderRegistry.getByName("anthropic").?;
    try std.testing.expectEqualStrings(connectors.ENV_VARS.anthropic.api_key[0], reg_info.env_key);
}

test "connectors: ProviderRegistry env_key matches ENV_VARS for gemini" {
    const reg_info = connectors.ProviderRegistry.getByName("gemini").?;
    try std.testing.expectEqualStrings(connectors.ENV_VARS.gemini.api_key[0], reg_info.env_key);
}

test "connectors: ProviderRegistry env_key matches ENV_VARS for huggingface" {
    const reg_info = connectors.ProviderRegistry.getByName("huggingface").?;
    try std.testing.expectEqualStrings(connectors.ENV_VARS.huggingface.api_key[0], reg_info.env_key);
}

test "connectors: ProviderRegistry env_key matches ENV_VARS for ollama" {
    const reg_info = connectors.ProviderRegistry.getByName("ollama").?;
    try std.testing.expectEqualStrings(connectors.ENV_VARS.ollama.host[0], reg_info.env_key);
}

test "connectors: ProviderRegistry env_key matches ENV_VARS for mistral" {
    const reg_info = connectors.ProviderRegistry.getByName("mistral").?;
    try std.testing.expectEqualStrings(connectors.ENV_VARS.mistral.api_key[0], reg_info.env_key);
}

test "connectors: ProviderRegistry env_key matches ENV_VARS for cohere" {
    const reg_info = connectors.ProviderRegistry.getByName("cohere").?;
    try std.testing.expectEqualStrings(connectors.ENV_VARS.cohere.api_key[0], reg_info.env_key);
}

test "connectors: ProviderRegistry env_key matches ENV_VARS for discord" {
    const reg_info = connectors.ProviderRegistry.getByName("discord").?;
    try std.testing.expectEqualStrings(connectors.ENV_VARS.discord.bot_token[0], reg_info.env_key);
}

// ============================================================================
// ProviderRegistry: Base URL Verification
// ============================================================================

test "connectors: ProviderRegistry base_url for cloud providers uses HTTPS" {
    const cloud_providers = [_][]const u8{
        "openai", "anthropic", "gemini", "huggingface", "mistral", "cohere", "discord",
    };
    for (cloud_providers) |name| {
        const info = connectors.ProviderRegistry.getByName(name).?;
        try std.testing.expect(std.mem.startsWith(u8, info.base_url, "https://"));
    }
}

test "connectors: ProviderRegistry base_url for local providers uses HTTP" {
    const local_providers = [_][]const u8{
        "ollama", "ollama_passthrough", "lm_studio", "vllm", "mlx", "llama_cpp",
    };
    for (local_providers) |name| {
        const info = connectors.ProviderRegistry.getByName(name).?;
        try std.testing.expect(std.mem.startsWith(u8, info.base_url, "http://"));
    }
}

// ============================================================================
// Shared HTTP Status Helpers (via abi.connectors.shared)
// ============================================================================

test "connectors: shared isRateLimitStatus detects 429" {
    try std.testing.expect(connectors.shared.isRateLimitStatus(429));
    try std.testing.expect(!connectors.shared.isRateLimitStatus(200));
    try std.testing.expect(!connectors.shared.isRateLimitStatus(500));
}

test "connectors: shared mapHttpStatus maps correctly" {
    try std.testing.expectEqual(connectors.shared.ConnectorError.RateLimitExceeded, connectors.shared.mapHttpStatus(429));
    try std.testing.expectEqual(connectors.shared.ConnectorError.ApiRequestFailed, connectors.shared.mapHttpStatus(500));
    try std.testing.expectEqual(connectors.shared.ConnectorError.ApiRequestFailed, connectors.shared.mapHttpStatus(400));
}

test {
    std.testing.refAllDecls(@This());
}
