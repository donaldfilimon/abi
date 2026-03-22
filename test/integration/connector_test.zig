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

test {
    std.testing.refAllDecls(@This());
}
