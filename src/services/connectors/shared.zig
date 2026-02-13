//! Shared types and utilities for connector modules.
//!
//! Provides common data structures, error types, and helper functions
//! used across different AI service connectors such as OpenAI, Anthropic,
//! Ollama, Mistral, Cohere, and HuggingFace.
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

/// Map HTTP status to connector error.
pub fn mapHttpStatus(status: u16) ConnectorError {
    if (status == 429) return ConnectorError.RateLimitExceeded;
    if (status >= 400) return ConnectorError.ApiRequestFailed;
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

const json_utils = @import("../shared/utils.zig").json;

/// Encode an array of ChatMessages as JSON array elements into a writer.
/// Produces: {"role":"...","content":"..."},{"role":"...","content":"..."}
/// The caller is responsible for the surrounding `[` and `]`.
pub fn encodeMessageArray(
    allocator: std.mem.Allocator,
    buf: *std.ArrayListUnmanaged(u8),
    messages: []const ChatMessage,
) !void {
    for (messages, 0..) |msg, i| {
        if (i > 0) try buf.append(allocator, ',');
        try buf.print(
            allocator,
            "{{\"role\":\"{s}\",\"content\":\"{}\"}}",
            .{ msg.role, json_utils.jsonEscape(msg.content) },
        );
    }
}

/// Encode an array of strings as JSON string array elements into a writer.
/// Produces: "str1","str2","str3"
/// The caller is responsible for the surrounding `[` and `]`.
pub fn encodeStringArray(
    allocator: std.mem.Allocator,
    buf: *std.ArrayListUnmanaged(u8),
    strings: []const []const u8,
) !void {
    for (strings, 0..) |s, i| {
        if (i > 0) try buf.append(allocator, ',');
        try buf.print(allocator, "\"{}\"", .{json_utils.jsonEscape(s)});
    }
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

test "encodeMessageArray encodes messages" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8){};
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
    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try encodeMessageArray(allocator, &buf, &.{});
    try std.testing.expectEqual(@as(usize, 0), buf.items.len);
}

test "encodeStringArray encodes strings" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    const strings = [_][]const u8{ "Hello", "World" };
    try encodeStringArray(allocator, &buf, &strings);
    const result = buf.items;

    try std.testing.expect(std.mem.indexOf(u8, result, "\"Hello\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"World\"") != null);
}

test "encodeStringArray empty" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8){};
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
