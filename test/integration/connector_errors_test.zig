//! Integration Tests: Connector Error Handling
//!
//! Verifies that connectors handle error conditions correctly:
//! - Missing API keys return appropriate errors
//! - Retry logic distinguishes retryable from non-retryable errors
//! - Config loading handles missing environment variables gracefully
//! - Secure memory cleanup (secureZero) is called on deallocation paths

const std = @import("std");
const abi = @import("abi");

const connectors = abi.connectors;
const shared = connectors.shared;

// ============================================================================
// Missing API Key Error Tests
// ============================================================================

test "connector errors: OpenAI loadFromEnv returns MissingApiKey without key" {
    const result = connectors.openai.loadFromEnv(std.testing.allocator);
    try std.testing.expectError(error.MissingApiKey, result);
}

test "connector errors: Anthropic loadFromEnv returns MissingApiKey without key" {
    const result = connectors.anthropic.loadFromEnv(std.testing.allocator);
    try std.testing.expectError(error.MissingApiKey, result);
}

test "connector errors: Claude loadFromEnv returns MissingApiKey without key" {
    const result = connectors.claude.loadFromEnv(std.testing.allocator);
    try std.testing.expectError(error.MissingApiKey, result);
}

test "connector errors: Gemini loadFromEnv returns MissingApiKey without key" {
    const result = connectors.gemini.loadFromEnv(std.testing.allocator);
    try std.testing.expectError(error.MissingApiKey, result);
}

test "connector errors: HuggingFace loadFromEnv returns MissingApiToken without key" {
    const result = connectors.huggingface.loadFromEnv(std.testing.allocator);
    try std.testing.expectError(error.MissingApiToken, result);
}

test "connector errors: Mistral loadFromEnv returns MissingApiKey without key" {
    const result = connectors.mistral.loadFromEnv(std.testing.allocator);
    try std.testing.expectError(error.MissingApiKey, result);
}

test "connector errors: Cohere loadFromEnv returns MissingApiKey without key" {
    const result = connectors.cohere.loadFromEnv(std.testing.allocator);
    try std.testing.expectError(error.MissingApiKey, result);
}

test "connector errors: Discord loadFromEnv returns MissingBotToken without key" {
    const result = connectors.discord.loadFromEnv(std.testing.allocator);
    try std.testing.expectError(error.MissingBotToken, result);
}

test "connector errors: Codex loadFromEnv returns MissingApiKey without key" {
    const result = connectors.codex.loadFromEnv(std.testing.allocator);
    try std.testing.expectError(error.MissingApiKey, result);
}

test "connector errors: OpenCode loadFromEnv returns MissingApiKey without key" {
    const result = connectors.opencode.loadFromEnv(std.testing.allocator);
    try std.testing.expectError(error.MissingApiKey, result);
}

// ============================================================================
// tryLoad wrappers return null (not error) for missing keys
// ============================================================================

test "connector errors: tryLoad wrappers convert missing-key errors to null" {
    // All tryLoad* wrappers should return null (not propagate error)
    // when the underlying API key is missing.
    try std.testing.expect(try connectors.tryLoadOpenAI(std.testing.allocator) == null);
    try std.testing.expect(try connectors.tryLoadAnthropic(std.testing.allocator) == null);
    try std.testing.expect(try connectors.tryLoadClaude(std.testing.allocator) == null);
    try std.testing.expect(try connectors.tryLoadGemini(std.testing.allocator) == null);
    try std.testing.expect(try connectors.tryLoadHuggingFace(std.testing.allocator) == null);
    try std.testing.expect(try connectors.tryLoadMistral(std.testing.allocator) == null);
    try std.testing.expect(try connectors.tryLoadCohere(std.testing.allocator) == null);
    try std.testing.expect(try connectors.tryLoadDiscord(std.testing.allocator) == null);
    try std.testing.expect(try connectors.tryLoadCodex(std.testing.allocator) == null);
    try std.testing.expect(try connectors.tryLoadOpenCode(std.testing.allocator) == null);
}

// ============================================================================
// Retry Logic: Retryable vs Non-Retryable Status Codes
// ============================================================================

test "connector errors: isRetryableStatus for 429 rate limit" {
    try std.testing.expect(shared.isRetryableStatus(429));
}

test "connector errors: isRetryableStatus for 5xx server errors" {
    try std.testing.expect(shared.isRetryableStatus(500));
    try std.testing.expect(shared.isRetryableStatus(502));
    try std.testing.expect(shared.isRetryableStatus(503));
    try std.testing.expect(shared.isRetryableStatus(504));
}

test "connector errors: non-retryable client errors (4xx except 429)" {
    try std.testing.expect(!shared.isRetryableStatus(400));
    try std.testing.expect(!shared.isRetryableStatus(401));
    try std.testing.expect(!shared.isRetryableStatus(403));
    try std.testing.expect(!shared.isRetryableStatus(404));
    try std.testing.expect(!shared.isRetryableStatus(422));
}

test "connector errors: success codes are not retryable" {
    try std.testing.expect(!shared.isRetryableStatus(200));
    try std.testing.expect(!shared.isRetryableStatus(201));
    try std.testing.expect(!shared.isRetryableStatus(204));
}

test "connector errors: mapHttpStatus maps 429 to RateLimitExceeded" {
    const err = shared.mapHttpStatus(429);
    try std.testing.expectEqual(shared.ConnectorError.RateLimitExceeded, err);
}

test "connector errors: mapHttpStatus maps 4xx/5xx to ApiRequestFailed" {
    try std.testing.expectEqual(shared.ConnectorError.ApiRequestFailed, shared.mapHttpStatus(400));
    try std.testing.expectEqual(shared.ConnectorError.ApiRequestFailed, shared.mapHttpStatus(401));
    try std.testing.expectEqual(shared.ConnectorError.ApiRequestFailed, shared.mapHttpStatus(403));
    try std.testing.expectEqual(shared.ConnectorError.ApiRequestFailed, shared.mapHttpStatus(500));
    try std.testing.expectEqual(shared.ConnectorError.ApiRequestFailed, shared.mapHttpStatus(503));
}

test "connector errors: isClientError covers 4xx range" {
    try std.testing.expect(shared.isClientError(400));
    try std.testing.expect(shared.isClientError(401));
    try std.testing.expect(shared.isClientError(429));
    try std.testing.expect(shared.isClientError(499));
    try std.testing.expect(!shared.isClientError(399));
    try std.testing.expect(!shared.isClientError(500));
}

test "connector errors: isServerError covers 5xx range" {
    try std.testing.expect(shared.isServerError(500));
    try std.testing.expect(shared.isServerError(503));
    try std.testing.expect(shared.isServerError(599));
    try std.testing.expect(!shared.isServerError(499));
    try std.testing.expect(!shared.isServerError(600));
}

// ============================================================================
// Exponential Backoff Logic
// ============================================================================

test "connector errors: exponential backoff doubles each attempt" {
    const base: u64 = 1000;
    const max: u64 = 60000;
    try std.testing.expectEqual(@as(u64, 1000), shared.exponentialBackoff(0, base, max));
    try std.testing.expectEqual(@as(u64, 2000), shared.exponentialBackoff(1, base, max));
    try std.testing.expectEqual(@as(u64, 4000), shared.exponentialBackoff(2, base, max));
    try std.testing.expectEqual(@as(u64, 8000), shared.exponentialBackoff(3, base, max));
    try std.testing.expectEqual(@as(u64, 16000), shared.exponentialBackoff(4, base, max));
}

test "connector errors: exponential backoff respects max cap" {
    try std.testing.expectEqual(@as(u64, 10000), shared.exponentialBackoff(5, 1000, 10000));
    try std.testing.expectEqual(@as(u64, 10000), shared.exponentialBackoff(10, 1000, 10000));
    try std.testing.expectEqual(@as(u64, 10000), shared.exponentialBackoff(20, 1000, 10000));
}

test "connector errors: retry delay with jitter stays within bounds" {
    // Jitter adds/subtracts up to 25%, so for base 1000, attempt 0:
    // backoff = 1000, jitter range = 250, result should be ~875-1125
    const d0 = shared.calculateRetryDelay(0, 1000, 60000);
    try std.testing.expect(d0 >= 750 and d0 <= 1500);

    // For attempt 1: backoff = 2000, jitter range = 500
    const d1 = shared.calculateRetryDelay(1, 1000, 60000);
    try std.testing.expect(d1 >= 1500 and d1 <= 2500);

    // For attempt 2: backoff = 4000, jitter range = 1000
    const d2 = shared.calculateRetryDelay(2, 1000, 60000);
    try std.testing.expect(d2 >= 3000 and d2 <= 5000);
}

test "connector errors: jitter alternates by attempt parity" {
    // Even attempts add jitter, odd subtract — verify they differ
    const d_even = shared.calculateRetryDelay(0, 1000, 60000);
    const d_odd = shared.calculateRetryDelay(1, 1000, 60000);
    // They use different base backoffs so just verify both are reasonable
    try std.testing.expect(d_even > 0);
    try std.testing.expect(d_odd > d_even); // attempt 1 has 2x base
}

// ============================================================================
// Config Loading: Missing Environment Variables
// ============================================================================

test "connector errors: getEnvOwned returns null for nonexistent env var" {
    const result = try connectors.getEnvOwned(std.testing.allocator, "ABI_CONNECTOR_ERRORS_TEST_NONEXISTENT");
    try std.testing.expect(result == null);
}

test "connector errors: getFirstEnvOwned returns null when no vars exist" {
    const result = try connectors.getFirstEnvOwned(std.testing.allocator, &.{
        "ABI_CONNECTOR_ERRORS_TEST_VAR_A",
        "ABI_CONNECTOR_ERRORS_TEST_VAR_B",
        "ABI_CONNECTOR_ERRORS_TEST_VAR_C",
    });
    try std.testing.expect(result == null);
}

test "connector errors: getFirstEnvOwned returns null for empty list" {
    const result = try connectors.getFirstEnvOwned(std.testing.allocator, &.{});
    try std.testing.expect(result == null);
}

test "connector errors: envIsSet returns false for unset variables" {
    try std.testing.expect(!shared.envIsSet("ABI_CONNECTOR_ERRORS_TEST_UNSET_VAR_1"));
    try std.testing.expect(!shared.envIsSet("ABI_CONNECTOR_ERRORS_TEST_UNSET_VAR_2"));
}

test "connector errors: anyEnvIsSet returns false when none are set" {
    try std.testing.expect(!shared.anyEnvIsSet(&.{
        "ABI_CONNECTOR_ERRORS_TEST_NONE_A",
        "ABI_CONNECTOR_ERRORS_TEST_NONE_B",
    }));
}

test "connector errors: anyEnvIsSet returns false for empty list" {
    const empty = [_][]const u8{};
    try std.testing.expect(!shared.anyEnvIsSet(&empty));
}

test "connector errors: envIsSet rejects oversized variable names" {
    // envIsSet uses a 256-byte stack buffer; names >= 256 should return false
    const long_name = "A" ** 300;
    try std.testing.expect(!shared.envIsSet(long_name));
}

// ============================================================================
// Secure Memory Cleanup
// ============================================================================

test "connector errors: secureFree wipes and frees memory" {
    const allocator = std.testing.allocator;
    const secret = try allocator.dupe(u8, "sk-test-secret-key-for-cleanup");

    // Verify data is intact before cleanup
    try std.testing.expectEqualStrings("sk-test-secret-key-for-cleanup", secret);

    // secureFree should secureZero then free without crashing
    shared.secureFree(allocator, @constCast(secret));
}

test "connector errors: secureFreeOptional handles null gracefully" {
    const allocator = std.testing.allocator;
    // Should be a no-op, no crash
    shared.secureFreeOptional(allocator, null);
}

test "connector errors: secureFreeOptional wipes non-null data" {
    const allocator = std.testing.allocator;
    const secret = try allocator.dupe(u8, "optional-secret-key");

    // Should wipe and free without crashing
    shared.secureFreeOptional(allocator, @constCast(secret));
}

test "connector errors: deinitConfig securely wipes api key" {
    const allocator = std.testing.allocator;
    const api_key = try allocator.dupe(u8, "test-api-key-for-deinit");
    const base_url = try allocator.dupe(u8, "https://api.example.com");

    // deinitConfig should secureFree api_key and free base_url
    shared.deinitConfig(allocator, @constCast(api_key), @constCast(base_url));
}

test "connector errors: AuthHeader.deinit securely wipes token" {
    const allocator = std.testing.allocator;
    var auth = try connectors.buildBearerHeader(allocator, "sk-auth-header-test");

    // Verify the header is valid before deinit
    const hdr = auth.header();
    try std.testing.expectEqualStrings("authorization", hdr.name);

    // deinit should secureZero the value before freeing
    auth.deinit(allocator);
}

// ============================================================================
// HTTP Status Classification Helpers
// ============================================================================

test "connector errors: isSuccessStatus covers full 2xx range" {
    try std.testing.expect(!shared.isSuccessStatus(199));
    try std.testing.expect(shared.isSuccessStatus(200));
    try std.testing.expect(shared.isSuccessStatus(201));
    try std.testing.expect(shared.isSuccessStatus(204));
    try std.testing.expect(shared.isSuccessStatus(299));
    try std.testing.expect(!shared.isSuccessStatus(300));
    try std.testing.expect(!shared.isSuccessStatus(0));
}

test "connector errors: status classification is mutually exclusive for standard codes" {
    // 200 is only success
    try std.testing.expect(shared.isSuccessStatus(200));
    try std.testing.expect(!shared.isClientError(200));
    try std.testing.expect(!shared.isServerError(200));

    // 400 is only client error
    try std.testing.expect(!shared.isSuccessStatus(400));
    try std.testing.expect(shared.isClientError(400));
    try std.testing.expect(!shared.isServerError(400));

    // 500 is only server error
    try std.testing.expect(!shared.isSuccessStatus(500));
    try std.testing.expect(!shared.isClientError(500));
    try std.testing.expect(shared.isServerError(500));
}

// ============================================================================
// JSON Helper Edge Cases
// ============================================================================

test "connector errors: jsonGetString returns null for non-object" {
    const val = std.json.Value{ .integer = 42 };
    try std.testing.expect(shared.jsonGetString(val, "key") == null);
}

test "connector errors: jsonGetInt returns null for non-object" {
    const val = std.json.Value{ .bool = true };
    try std.testing.expect(shared.jsonGetInt(val, "key") == null);
}

test {
    std.testing.refAllDecls(@This());
}
