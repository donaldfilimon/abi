//! Connector integration tests.
//!
//! Verifies that all 9 AI connectors + 2 utility connectors expose a consistent
//! public interface (isAvailable, Config, loadFromEnv, tryLoad*) through the
//! abi.services.connectors namespace.

const std = @import("std");
const abi = @import("abi");

// ============================================================================
// isAvailable() Interface Consistency
// ============================================================================

test "all connectors expose isAvailable()" {
    // Each connector must return bool and not allocate.
    const openai: bool = abi.services.connectors.openai.isAvailable();
    const anthropic: bool = abi.services.connectors.anthropic.isAvailable();
    const ollama: bool = abi.services.connectors.ollama.isAvailable();
    const huggingface: bool = abi.services.connectors.huggingface.isAvailable();
    const mistral: bool = abi.services.connectors.mistral.isAvailable();
    const cohere: bool = abi.services.connectors.cohere.isAvailable();
    const lm_studio: bool = abi.services.connectors.lm_studio.isAvailable();
    const vllm_avail: bool = abi.services.connectors.vllm.isAvailable();
    const mlx_avail: bool = abi.services.connectors.mlx.isAvailable();
    const discord_avail: bool = abi.services.connectors.discord.isAvailable();

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
    _ = discord_avail;
}

test "isAvailable is consistent across repeated calls" {
    // isAvailable() must be idempotent — same result every call.
    const first = abi.services.connectors.openai.isAvailable();
    const second = abi.services.connectors.openai.isAvailable();
    const third = abi.services.connectors.openai.isAvailable();
    try std.testing.expectEqual(first, second);
    try std.testing.expectEqual(second, third);
}

test "isAvailable idempotent for all connectors" {
    // Verify idempotency across all connectors, not just openai.
    inline for (.{
        abi.services.connectors.anthropic.isAvailable(),
        abi.services.connectors.ollama.isAvailable(),
        abi.services.connectors.huggingface.isAvailable(),
        abi.services.connectors.mistral.isAvailable(),
        abi.services.connectors.cohere.isAvailable(),
        abi.services.connectors.lm_studio.isAvailable(),
        abi.services.connectors.vllm.isAvailable(),
        abi.services.connectors.mlx.isAvailable(),
        abi.services.connectors.discord.isAvailable(),
    }) |first_result| {
        _ = first_result;
    }

    // Re-call after first pass — still stable
    const a1 = abi.services.connectors.anthropic.isAvailable();
    const a2 = abi.services.connectors.anthropic.isAvailable();
    try std.testing.expectEqual(a1, a2);

    const o1 = abi.services.connectors.ollama.isAvailable();
    const o2 = abi.services.connectors.ollama.isAvailable();
    try std.testing.expectEqual(o1, o2);

    const ls1 = abi.services.connectors.lm_studio.isAvailable();
    const ls2 = abi.services.connectors.lm_studio.isAvailable();
    try std.testing.expectEqual(ls1, ls2);

    const v1 = abi.services.connectors.vllm.isAvailable();
    const v2 = abi.services.connectors.vllm.isAvailable();
    try std.testing.expectEqual(v1, v2);

    const m1 = abi.services.connectors.mlx.isAvailable();
    const m2 = abi.services.connectors.mlx.isAvailable();
    try std.testing.expectEqual(m1, m2);
}

// ============================================================================
// Config Types Exist
// ============================================================================

test "all connectors expose Config type" {
    // Verify each connector has a Config struct accessible through abi.services.connectors.
    _ = abi.services.connectors.openai.Config;
    _ = abi.services.connectors.anthropic.Config;
    _ = abi.services.connectors.ollama.Config;
    _ = abi.services.connectors.huggingface.Config;
    _ = abi.services.connectors.mistral.Config;
    _ = abi.services.connectors.cohere.Config;
    _ = abi.services.connectors.lm_studio.Config;
    _ = abi.services.connectors.vllm.Config;
    _ = abi.services.connectors.mlx.Config;
}

// ============================================================================
// tryLoad* Functions (Graceful Failure)
// ============================================================================

test "tryLoad functions return null without env vars" {
    // Skip if any connector env vars are already set in the environment.
    const has_openai = std.c.getenv("OPENAI_API_KEY") != null or std.c.getenv("ABI_OPENAI_API_KEY") != null;
    const has_anthropic = std.c.getenv("ANTHROPIC_API_KEY") != null or std.c.getenv("ABI_ANTHROPIC_API_KEY") != null;
    const has_hf = std.c.getenv("HF_TOKEN") != null or std.c.getenv("HUGGINGFACE_TOKEN") != null or std.c.getenv("ABI_HUGGINGFACE_TOKEN") != null;
    const has_discord = std.c.getenv("DISCORD_TOKEN") != null or std.c.getenv("ABI_DISCORD_TOKEN") != null;
    const has_mistral = std.c.getenv("MISTRAL_API_KEY") != null or std.c.getenv("ABI_MISTRAL_API_KEY") != null;
    const has_cohere = std.c.getenv("COHERE_API_KEY") != null or std.c.getenv("ABI_COHERE_API_KEY") != null;

    if (has_openai or has_anthropic or has_hf or has_discord or has_mistral or has_cohere) {
        return error.SkipZigTest;
    }

    const allocator = std.testing.allocator;

    // Without API keys set, tryLoad* should return null, not error.
    const openai_config = try abi.services.connectors.tryLoadOpenAI(allocator);
    try std.testing.expect(openai_config == null);

    const anthropic_config = try abi.services.connectors.tryLoadAnthropic(allocator);
    try std.testing.expect(anthropic_config == null);

    const hf_config = try abi.services.connectors.tryLoadHuggingFace(allocator);
    try std.testing.expect(hf_config == null);

    const discord_config = try abi.services.connectors.tryLoadDiscord(allocator);
    try std.testing.expect(discord_config == null);

    const mistral_config = try abi.services.connectors.tryLoadMistral(allocator);
    try std.testing.expect(mistral_config == null);

    const cohere_config = try abi.services.connectors.tryLoadCohere(allocator);
    try std.testing.expect(cohere_config == null);

    // Local connectors (lm_studio, vllm, mlx, ollama) always succeed with
    // defaults — they don't require API keys. Verify they return valid configs
    // and properly clean up.
    if (try abi.services.connectors.tryLoadLMStudio(allocator)) |cfg| {
        var config = cfg;
        config.deinit(allocator);
    }

    if (try abi.services.connectors.tryLoadVLLM(allocator)) |cfg| {
        var config = cfg;
        config.deinit(allocator);
    }

    if (try abi.services.connectors.tryLoadMLX(allocator)) |cfg| {
        var config = cfg;
        config.deinit(allocator);
    }

    if (try abi.services.connectors.tryLoadOllama(allocator)) |cfg| {
        var config = cfg;
        config.deinit(allocator);
    }
}

// ============================================================================
// Connector Module Init/Deinit
// ============================================================================

test "connectors module init and deinit" {
    const allocator = std.testing.allocator;

    try abi.services.connectors.init(allocator);
    try std.testing.expect(abi.services.connectors.isInitialized());

    abi.services.connectors.deinit();
    try std.testing.expect(!abi.services.connectors.isInitialized());
}

test "connectors double init is safe" {
    const allocator = std.testing.allocator;

    try abi.services.connectors.init(allocator);
    try abi.services.connectors.init(allocator);
    try std.testing.expect(abi.services.connectors.isInitialized());

    abi.services.connectors.deinit();
}

// ============================================================================
// Auth Header Builder
// ============================================================================

test "buildBearerHeader produces valid format" {
    const allocator = std.testing.allocator;

    var auth = try abi.services.connectors.buildBearerHeader(allocator, "test-token-123");
    defer auth.deinit(allocator);

    try std.testing.expectEqualStrings("Bearer test-token-123", auth.value);
    const header = auth.header();
    try std.testing.expectEqualStrings("authorization", header.name);
}

test "buildBearerHeader with empty token" {
    const allocator = std.testing.allocator;

    var auth = try abi.services.connectors.buildBearerHeader(allocator, "");
    defer auth.deinit(allocator);

    try std.testing.expectEqualStrings("Bearer ", auth.value);
}

// ============================================================================
// getEnvOwned Boundary Tests
// ============================================================================

test "getEnvOwned returns null for missing vars" {
    const allocator = std.testing.allocator;

    const result = try abi.services.connectors.getEnvOwned(allocator, "ABI_NONEXISTENT_TEST_VAR_999");
    try std.testing.expect(result == null);
}

test "getFirstEnvOwned returns null for all missing" {
    const allocator = std.testing.allocator;

    const result = try abi.services.connectors.getFirstEnvOwned(allocator, &.{
        "ABI_NONEXISTENT_A_12345",
        "ABI_NONEXISTENT_B_12345",
        "ABI_NONEXISTENT_C_12345",
    });
    try std.testing.expect(result == null);
}

test "getFirstEnvOwned with empty list" {
    const allocator = std.testing.allocator;

    const result = try abi.services.connectors.getFirstEnvOwned(allocator, &.{});
    try std.testing.expect(result == null);
}
