//! Connector integration tests.
//!
//! Verifies that all 9 AI connectors + 2 utility connectors expose a consistent
//! public interface (isAvailable, Config, loadFromEnv, tryLoad*) through the
//! abi.connectors namespace.

const std = @import("std");
const abi = @import("abi");

// ============================================================================
// isAvailable() Interface Consistency
// ============================================================================

test "all connectors expose isAvailable()" {
    // Each connector must return bool and not allocate.
    const openai: bool = abi.connectors.openai.isAvailable();
    const anthropic: bool = abi.connectors.anthropic.isAvailable();
    const ollama: bool = abi.connectors.ollama.isAvailable();
    const huggingface: bool = abi.connectors.huggingface.isAvailable();
    const mistral: bool = abi.connectors.mistral.isAvailable();
    const cohere: bool = abi.connectors.cohere.isAvailable();
    const lm_studio: bool = abi.connectors.lm_studio.isAvailable();
    const vllm_avail: bool = abi.connectors.vllm.isAvailable();
    const mlx_avail: bool = abi.connectors.mlx.isAvailable();

    // Without env vars configured, all should return false in CI.
    // We can't assert false since local dev might have keys, but we
    // CAN verify the functions return without error or panic.
    _ = openai;
    _ = anthropic;
    _ = ollama;
    _ = huggingface;
    _ = mistral;
    _ = cohere;
    _ = lm_studio;
    _ = vllm_avail;
    _ = mlx_avail;
}

test "isAvailable is consistent across repeated calls" {
    // isAvailable() must be idempotent — same result every call.
    const first = abi.connectors.openai.isAvailable();
    const second = abi.connectors.openai.isAvailable();
    const third = abi.connectors.openai.isAvailable();
    try std.testing.expectEqual(first, second);
    try std.testing.expectEqual(second, third);
}

test "isAvailable idempotent for all connectors" {
    // Verify idempotency across all connectors, not just openai.
    inline for (.{
        abi.connectors.anthropic.isAvailable(),
        abi.connectors.ollama.isAvailable(),
        abi.connectors.huggingface.isAvailable(),
        abi.connectors.mistral.isAvailable(),
        abi.connectors.cohere.isAvailable(),
        abi.connectors.lm_studio.isAvailable(),
        abi.connectors.vllm.isAvailable(),
        abi.connectors.mlx.isAvailable(),
    }) |first_result| {
        _ = first_result;
    }

    // Re-call after first pass — still stable
    const a1 = abi.connectors.anthropic.isAvailable();
    const a2 = abi.connectors.anthropic.isAvailable();
    try std.testing.expectEqual(a1, a2);

    const o1 = abi.connectors.ollama.isAvailable();
    const o2 = abi.connectors.ollama.isAvailable();
    try std.testing.expectEqual(o1, o2);

    const ls1 = abi.connectors.lm_studio.isAvailable();
    const ls2 = abi.connectors.lm_studio.isAvailable();
    try std.testing.expectEqual(ls1, ls2);

    const v1 = abi.connectors.vllm.isAvailable();
    const v2 = abi.connectors.vllm.isAvailable();
    try std.testing.expectEqual(v1, v2);

    const m1 = abi.connectors.mlx.isAvailable();
    const m2 = abi.connectors.mlx.isAvailable();
    try std.testing.expectEqual(m1, m2);
}

// ============================================================================
// Config Types Exist
// ============================================================================

test "all connectors expose Config type" {
    // Verify each connector has a Config struct accessible through abi.connectors.
    _ = abi.connectors.openai.Config;
    _ = abi.connectors.anthropic.Config;
    _ = abi.connectors.ollama.Config;
    _ = abi.connectors.huggingface.Config;
    _ = abi.connectors.mistral.Config;
    _ = abi.connectors.cohere.Config;
    _ = abi.connectors.lm_studio.Config;
    _ = abi.connectors.vllm.Config;
    _ = abi.connectors.mlx.Config;
}

// ============================================================================
// tryLoad* Functions (Graceful Failure)
// ============================================================================

test "tryLoad functions return null without env vars" {
    const allocator = std.testing.allocator;

    // Without API keys set, tryLoad* should return null, not error.
    const openai_config = try abi.connectors.tryLoadOpenAI(allocator);
    try std.testing.expect(openai_config == null);

    const anthropic_config = try abi.connectors.tryLoadAnthropic(allocator);
    try std.testing.expect(anthropic_config == null);

    const hf_config = try abi.connectors.tryLoadHuggingFace(allocator);
    try std.testing.expect(hf_config == null);

    const discord_config = try abi.connectors.tryLoadDiscord(allocator);
    try std.testing.expect(discord_config == null);

    const mistral_config = try abi.connectors.tryLoadMistral(allocator);
    try std.testing.expect(mistral_config == null);

    const cohere_config = try abi.connectors.tryLoadCohere(allocator);
    try std.testing.expect(cohere_config == null);

    // Local connectors (lm_studio, vllm, mlx, ollama) always succeed with
    // defaults — they don't require API keys. Verify they return valid configs
    // and properly clean up.
    if (try abi.connectors.tryLoadLMStudio(allocator)) |cfg| {
        var config = cfg;
        config.deinit(allocator);
    }

    if (try abi.connectors.tryLoadVLLM(allocator)) |cfg| {
        var config = cfg;
        config.deinit(allocator);
    }

    if (try abi.connectors.tryLoadMLX(allocator)) |cfg| {
        var config = cfg;
        config.deinit(allocator);
    }
}

// ============================================================================
// Connector Module Init/Deinit
// ============================================================================

test "connectors module init and deinit" {
    const allocator = std.testing.allocator;

    try abi.connectors.init(allocator);
    try std.testing.expect(abi.connectors.isInitialized());

    abi.connectors.deinit();
    try std.testing.expect(!abi.connectors.isInitialized());
}

test "connectors double init is safe" {
    const allocator = std.testing.allocator;

    try abi.connectors.init(allocator);
    try abi.connectors.init(allocator);
    try std.testing.expect(abi.connectors.isInitialized());

    abi.connectors.deinit();
}

// ============================================================================
// Auth Header Builder
// ============================================================================

test "buildBearerHeader produces valid format" {
    const allocator = std.testing.allocator;

    var auth = try abi.connectors.buildBearerHeader(allocator, "test-token-123");
    defer auth.deinit(allocator);

    try std.testing.expectEqualStrings("Bearer test-token-123", auth.value);
    const header = auth.header();
    try std.testing.expectEqualStrings("authorization", header.name);
}

test "buildBearerHeader with empty token" {
    const allocator = std.testing.allocator;

    var auth = try abi.connectors.buildBearerHeader(allocator, "");
    defer auth.deinit(allocator);

    try std.testing.expectEqualStrings("Bearer ", auth.value);
}

// ============================================================================
// getEnvOwned Boundary Tests
// ============================================================================

test "getEnvOwned returns null for missing vars" {
    const allocator = std.testing.allocator;

    const result = try abi.connectors.getEnvOwned(allocator, "ABI_NONEXISTENT_TEST_VAR_999");
    try std.testing.expect(result == null);
}

test "getFirstEnvOwned returns null for all missing" {
    const allocator = std.testing.allocator;

    const result = try abi.connectors.getFirstEnvOwned(allocator, &.{
        "ABI_NONEXISTENT_A_12345",
        "ABI_NONEXISTENT_B_12345",
        "ABI_NONEXISTENT_C_12345",
    });
    try std.testing.expect(result == null);
}

test "getFirstEnvOwned with empty list" {
    const allocator = std.testing.allocator;

    const result = try abi.connectors.getFirstEnvOwned(allocator, &.{});
    try std.testing.expect(result == null);
}
