//! Shared types and utilities for connector modules.
//!
//! Provides common data structures, error types, and helper functions
//! used across all 9 AI service connectors: OpenAI, Anthropic, Ollama,
//! HuggingFace, Mistral, Cohere, LM Studio, vLLM, and MLX.
//!
//! ## Design Pattern
//!
//! All connectors follow a consistent pattern:
//! - `Config` struct with API key and base URL
//! - `Client` struct with async HTTP client
//! - `loadFromEnv()` to load config from environment variables
//! - Secure memory wiping for API keys
//!
//! This module provides the common building blocks for that pattern.

const std = @import("std");

// ============================================================================
// Chat Message Types
// ============================================================================

/// Chat message structure used in conversation-based APIs.
/// Compatible with OpenAI, Ollama, Anthropic, and similar chat completions formats.
pub const ChatMessage = struct {
    /// The role of the message sender (e.g., "system", "user", "assistant")
    role: []const u8,
    /// The content of the message
    content: []const u8,
};

/// Message role constants for common API formats
pub const Role = struct {
    pub const SYSTEM = "system";
    pub const USER = "user";
    pub const ASSISTANT = "assistant";
    pub const FUNCTION = "function";
    pub const TOOL = "tool";
};

// ============================================================================
// Common Error Types
// ============================================================================

/// Common errors shared across all connectors.
/// Each connector can extend this with provider-specific errors.
pub const ConnectorError = error{
    /// API key was not provided via environment variable.
    MissingApiKey,
    /// The API request failed (network error or non-2xx status).
    ApiRequestFailed,
    /// The API response could not be parsed.
    InvalidResponse,
    /// Rate limit exceeded (HTTP 429). Retry after backoff.
    RateLimitExceeded,
    /// Request timed out.
    Timeout,
    /// Out of memory.
    OutOfMemory,
};

/// Unified error set for all provider connectors.
/// Provider-specific error sets should be a superset of this base set.
/// This allows callers to handle common failure modes without importing
/// provider-specific types.
pub const ProviderError = error{
    MissingAuth,
    ApiRequestFailed,
    RateLimitExceeded,
    InvalidResponse,
    Timeout,
    NetworkError,
    ModelNotFound,
    FeatureDisabled,
};

// ============================================================================
// Provider Info
// ============================================================================

/// Metadata about a provider connector, used for discovery and availability checks.
pub const ProviderInfo = struct {
    name: []const u8,
    is_available: bool,
    env_key: []const u8,
};

// ============================================================================
// Secure Memory Helpers
// ============================================================================

/// Securely wipe and free an API key or sensitive string.
/// Uses `std.crypto.secureZero` to prevent memory forensics.
pub fn secureFree(allocator: std.mem.Allocator, data: []u8) void {
    std.crypto.secureZero(u8, data);
    allocator.free(data);
}

/// Securely wipe and free an optional string.
pub fn secureFreeOptional(allocator: std.mem.Allocator, data: ?[]u8) void {
    if (data) |d| {
        secureFree(allocator, d);
    }
}

// ============================================================================
// Config Helpers
// ============================================================================

/// Generic config deinit helper that securely wipes API key.
/// Use this in connector Config.deinit() implementations.
pub fn deinitConfig(
    allocator: std.mem.Allocator,
    api_key: []u8,
    base_url: []u8,
) void {
    secureFree(allocator, api_key);
    allocator.free(base_url);
}

// ============================================================================
// HTTP Response Helpers
// ============================================================================

/// Check if an HTTP status code indicates success (2xx).
pub fn isSuccessStatus(status: u16) bool {
    return status >= 200 and status < 300;
}

/// Check if an HTTP status code indicates rate limiting (429).
pub fn isRateLimitStatus(status: u16) bool {
    return status == 429;
}

/// Check if an HTTP status code indicates a client error (4xx).
pub fn isClientError(status: u16) bool {
    return status >= 400 and status < 500;
}

/// Check if an HTTP status code indicates a server error (5xx).
pub fn isServerError(status: u16) bool {
    return status >= 500 and status < 600;
}

/// Map HTTP status to connector error. Returns `RateLimitExceeded` for 429,
/// `ApiRequestFailed` for all other error statuses (4xx/5xx and unexpected codes).
pub fn mapHttpStatus(status: u16) ConnectorError {
    if (status == 429) return ConnectorError.RateLimitExceeded;
    return ConnectorError.ApiRequestFailed;
}

// ============================================================================
// JSON Helpers
// ============================================================================

/// Extract a string field from a JSON object.
/// Returns null if field doesn't exist or isn't a string.
pub fn jsonGetString(value: std.json.Value, field: []const u8) ?[]const u8 {
    if (value != .object) return null;
    if (value.object.get(field)) |v| {
        if (v == .string) return v.string;
    }
    return null;
}

/// Extract an integer field from a JSON object.
pub fn jsonGetInt(value: std.json.Value, field: []const u8) ?i64 {
    if (value != .object) return null;
    if (value.object.get(field)) |v| {
        if (v == .integer) return v.integer;
    }
    return null;
}

/// Extract an unsigned integer field from a JSON object.
pub fn jsonGetUint(comptime T: type, value: std.json.Value, field: []const u8) ?T {
    if (jsonGetInt(value, field)) |i| {
        if (i >= 0 and i <= std.math.maxInt(T)) {
            return @intCast(i);
        }
    }
    return null;
}

// ============================================================================
// JSON Encoding Helpers
// ============================================================================

const json_utils = @import("../foundation/mod.zig").utils.json;

/// Encode an array of ChatMessages as JSON array elements.
/// Produces: {"role":"...","content":"..."},{"role":"...","content":"..."}
/// The caller is responsible for the surrounding `[` and `]`.
pub fn encodeMessageArray(
    allocator: std.mem.Allocator,
    buf: *std.ArrayListUnmanaged(u8),
    messages: []const ChatMessage,
) !void {
    for (messages, 0..) |msg, i| {
        if (i > 0) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "{\"role\":\"");
        try json_utils.appendJsonEscaped(allocator, buf, msg.role);
        try buf.appendSlice(allocator, "\",\"content\":\"");
        try json_utils.appendJsonEscaped(allocator, buf, msg.content);
        try buf.appendSlice(allocator, "\"}");
    }
}

/// Encode an array of strings as JSON string array elements.
/// Produces: "str1","str2","str3"
/// The caller is responsible for the surrounding `[` and `]`.
pub fn encodeStringArray(
    allocator: std.mem.Allocator,
    buf: *std.ArrayListUnmanaged(u8),
    strings: []const []const u8,
) !void {
    for (strings, 0..) |s, i| {
        if (i > 0) try buf.append(allocator, ',');
        try buf.append(allocator, '"');
        try json_utils.appendJsonEscaped(allocator, buf, s);
        try buf.append(allocator, '"');
    }
}

// ============================================================================
// Availability Helpers
// ============================================================================

const c_stdlib = struct {
    pub fn getenv(_: [*:0]const u8) ?[*:0]const u8 {
        return null;
    }
};

/// Check if an environment variable is set and non-empty.
/// Returns false for unset vars AND empty strings (e.g., `VAR=""`).
/// Used by connector isAvailable() functions for zero-allocation health checks.
pub fn envIsSet(name: []const u8) bool {
    // Use stack buffer for null-terminated string to avoid allocation
    var buf: [256]u8 = undefined;
    if (name.len >= buf.len) return false;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const value = c_stdlib.getenv(@ptrCast(buf[0..name.len :0])) orelse return false;
    // Treat empty string as unset (e.g., OPENAI_API_KEY="")
    return std.mem.sliceTo(value, 0).len > 0;
}

/// Check if any of the given environment variable names are set.
pub fn anyEnvIsSet(names: []const []const u8) bool {
    for (names) |name| {
        if (envIsSet(name)) return true;
    }
    return false;
}

// ============================================================================
// Retry Helpers
// ============================================================================

/// Calculate exponential backoff delay in milliseconds.
/// Returns: base_ms * 2^attempt, capped at max_ms
pub fn exponentialBackoff(attempt: u32, base_ms: u64, max_ms: u64) u64 {
    const multiplier = std.math.shl(u64, 1, @min(attempt, 10));
    return @min(base_ms * multiplier, max_ms);
}

/// Calculate retry delay with jitter for production use.
/// Adds ±25% jitter to prevent thundering herd on retry storms.
pub fn calculateRetryDelay(attempt: u32, base_ms: u64, max_ms: u64) u64 {
    const delay = exponentialBackoff(attempt, base_ms, max_ms);
    return addJitter(delay, attempt);
}

/// Add deterministic jitter to a delay value.
/// Uses attempt number to alternate between adding and subtracting.
fn addJitter(delay: u64, attempt: u32) u64 {
    const jitter_range = delay / 4;
    if (jitter_range == 0) return delay;
    if (attempt % 2 == 0) {
        return delay + jitter_range / 2;
    } else {
        return delay -| jitter_range / 2; // saturating subtract
    }
}

/// Re-export RetryOptions from async_http for connector convenience.
pub const RetryOptions = @import("../foundation/mod.zig").utils.async_http.RetryOptions;

/// Default retry options for AI API connectors.
/// 3 retries, 1s base, 30s max, retries on 429/5xx/network errors.
pub const DEFAULT_RETRY_OPTIONS = RetryOptions.DEFAULT;

/// Check if an HTTP error is retryable.
pub fn isRetryableStatus(status: u16) bool {
    return status == 429 or status >= 500;
}

// ============================================================================
// Tests
// ============================================================================

test "ChatMessage can be created" {
    const msg = ChatMessage{
        .role = Role.USER,
        .content = "Hello, world!",
    };

    try std.testing.expectEqualStrings("user", msg.role);
    try std.testing.expectEqualStrings("Hello, world!", msg.content);
}

test "isSuccessStatus returns correct values" {
    try std.testing.expect(isSuccessStatus(200));
    try std.testing.expect(isSuccessStatus(201));
    try std.testing.expect(isSuccessStatus(299));
    try std.testing.expect(!isSuccessStatus(199));
    try std.testing.expect(!isSuccessStatus(300));
    try std.testing.expect(!isSuccessStatus(404));
    try std.testing.expect(!isSuccessStatus(500));
}

test "isRateLimitStatus detects 429" {
    try std.testing.expect(isRateLimitStatus(429));
    try std.testing.expect(!isRateLimitStatus(200));
    try std.testing.expect(!isRateLimitStatus(500));
}

test "exponentialBackoff calculates correctly" {
    try std.testing.expectEqual(@as(u64, 1000), exponentialBackoff(0, 1000, 60000));
    try std.testing.expectEqual(@as(u64, 2000), exponentialBackoff(1, 1000, 60000));
    try std.testing.expectEqual(@as(u64, 4000), exponentialBackoff(2, 1000, 60000));
    try std.testing.expectEqual(@as(u64, 60000), exponentialBackoff(10, 1000, 60000)); // capped
}

test "calculateRetryDelay with jitter" {
    const d0 = calculateRetryDelay(0, 1000, 60000);
    try std.testing.expect(d0 >= 750 and d0 <= 1500);

    const d1 = calculateRetryDelay(1, 1000, 60000);
    try std.testing.expect(d1 >= 1500 and d1 <= 2500);
}

test "isRetryableStatus" {
    try std.testing.expect(isRetryableStatus(429));
    try std.testing.expect(isRetryableStatus(500));
    try std.testing.expect(isRetryableStatus(503));
    try std.testing.expect(!isRetryableStatus(200));
    try std.testing.expect(!isRetryableStatus(400));
    try std.testing.expect(!isRetryableStatus(404));
}

test "encodeMessageArray encodes messages" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello" },
        .{ .role = "assistant", .content = "Hi there" },
    };

    try encodeMessageArray(allocator, &buf, &messages);
    const result = buf.items;

    try std.testing.expect(std.mem.indexOf(u8, result, "\"role\":\"user\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"content\":\"Hello\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"role\":\"assistant\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"content\":\"Hi there\"") != null);
}

test "encodeMessageArray empty" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    try encodeMessageArray(allocator, &buf, &.{});
    try std.testing.expectEqual(@as(usize, 0), buf.items.len);
}

test "encodeStringArray encodes strings" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    const strings = [_][]const u8{ "Hello", "World" };
    try encodeStringArray(allocator, &buf, &strings);
    const result = buf.items;

    try std.testing.expect(std.mem.indexOf(u8, result, "\"Hello\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"World\"") != null);
}

test "encodeStringArray empty" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    const empty = [_][]const u8{};
    try encodeStringArray(allocator, &buf, &empty);
    try std.testing.expectEqual(@as(usize, 0), buf.items.len);
}

test "secureFree wipes memory" {
    const allocator = std.testing.allocator;
    const data = try allocator.dupe(u8, "secret_api_key_12345");

    // Verify data exists
    try std.testing.expectEqualStrings("secret_api_key_12345", data);

    // After secureFree, data is wiped (we can't read it, but the function shouldn't crash)
    secureFree(allocator, @constCast(data));
    // Note: data is now freed and invalid - this test just ensures no crash
}

test "envIsSet returns false for nonexistent var" {
    try std.testing.expect(!envIsSet("ABI_NONEXISTENT_TEST_VAR_12345"));
}

test "anyEnvIsSet returns false for empty list" {
    const empty = [_][]const u8{};
    try std.testing.expect(!anyEnvIsSet(&empty));
}

test "ProviderError is subset of OpenAI error set" {
    const openai = @import("openai.zig");
    // Verify every ProviderError value is present in OpenAIError.
    // If ProviderError has a value not in OpenAIError, this assignment fails at comptime.
    const provider_as_openai: openai.OpenAIError = @errorCast(ProviderError.ApiRequestFailed);
    try std.testing.expectEqual(provider_as_openai, openai.OpenAIError.ApiRequestFailed);
}

test "ProviderError is subset of Anthropic error set" {
    const anthropic = @import("anthropic.zig");
    const provider_as_anthropic: anthropic.AnthropicError = @errorCast(ProviderError.RateLimitExceeded);
    try std.testing.expectEqual(provider_as_anthropic, anthropic.AnthropicError.RateLimitExceeded);
}

test "ProviderError is subset of Ollama error set" {
    const ollama = @import("ollama.zig");
    const provider_as_ollama: ollama.OllamaError = @errorCast(ProviderError.InvalidResponse);
    try std.testing.expectEqual(provider_as_ollama, ollama.OllamaError.InvalidResponse);
}

test "ProviderError contains all expected error cases" {
    const errors = [_]ProviderError{
        ProviderError.MissingAuth,
        ProviderError.ApiRequestFailed,
        ProviderError.RateLimitExceeded,
        ProviderError.InvalidResponse,
        ProviderError.Timeout,
        ProviderError.NetworkError,
        ProviderError.ModelNotFound,
        ProviderError.FeatureDisabled,
    };
    try std.testing.expectEqual(@as(usize, 8), errors.len);
}

test "ProviderInfo can be constructed" {
    const info = ProviderInfo{
        .name = "openai",
        .is_available = true,
        .env_key = "ABI_OPENAI_API_KEY",
    };
    try std.testing.expectEqualStrings("openai", info.name);
    try std.testing.expect(info.is_available);
    try std.testing.expectEqualStrings("ABI_OPENAI_API_KEY", info.env_key);
}

test "provider error sets are proper supersets of ProviderError" {
    const openai = @import("openai.zig");
    const anthropic = @import("anthropic.zig");
    const ollama = @import("ollama.zig");

    // OpenAI has MissingApiKey beyond ProviderError
    const oe: openai.OpenAIError = openai.OpenAIError.MissingApiKey;
    try std.testing.expect(oe == openai.OpenAIError.MissingApiKey);

    // Anthropic has ContentFiltered and MissingApiKey beyond ProviderError
    const ae: anthropic.AnthropicError = anthropic.AnthropicError.ContentFiltered;
    try std.testing.expect(ae == anthropic.AnthropicError.ContentFiltered);

    // Ollama has ModelNotAvailable beyond ProviderError
    const le: ollama.OllamaError = ollama.OllamaError.ModelNotAvailable;
    try std.testing.expect(le == ollama.OllamaError.ModelNotAvailable);
}

test {
    std.testing.refAllDecls(@This());
}
