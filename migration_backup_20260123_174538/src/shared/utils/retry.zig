//! Retry utilities for shared use.
//!
//! Provides retry configuration and helpers for retryable operations.

const std = @import("std");

/// Retry configuration.
pub const RetryConfig = struct {
    /// Maximum number of retry attempts.
    max_retries: u32 = 3,
    /// Initial delay between retries (milliseconds).
    initial_delay_ms: u64 = 100,
    /// Maximum delay between retries (milliseconds).
    max_delay_ms: u64 = 30_000,
    /// Backoff multiplier.
    multiplier: f64 = 2.0,
    /// Add random jitter to delays.
    jitter: bool = true,
    /// Jitter factor (0.0 to 1.0).
    jitter_factor: f64 = 0.25,
};

/// Check if an HTTP status code indicates a retryable error.
pub fn isStatusRetryable(status_code: u16) bool {
    return switch (status_code) {
        // Server errors
        500, 502, 503, 504 => true,
        // Rate limiting
        429 => true,
        // Request timeout
        408 => true,
        else => false,
    };
}

/// Calculate delay for a given retry attempt using exponential backoff.
pub fn calculateDelay(config: RetryConfig, attempt: u32) u64 {
    const base_delay = config.initial_delay_ms;
    const multiplier = std.math.pow(f64, config.multiplier, @as(f64, @floatFromInt(attempt)));
    var delay: u64 = @intFromFloat(@as(f64, @floatFromInt(base_delay)) * multiplier);

    // Cap at max delay
    if (delay > config.max_delay_ms) {
        delay = config.max_delay_ms;
    }

    // Add jitter if enabled
    if (config.jitter) {
        const jitter_range = @as(f64, @floatFromInt(delay)) * config.jitter_factor;
        const jitter: i64 = @as(i64, @intFromFloat(jitter_range * 2.0)) - @as(i64, @intFromFloat(jitter_range));
        const signed_delay: i64 = @intCast(delay);
        const adjusted = signed_delay + jitter;
        delay = if (adjusted > 0) @intCast(adjusted) else 1;
    }

    return delay;
}

/// Retry error result.
pub const RetryError = struct {
    attempts: u32,
    last_status: u16,

    pub fn format(
        self: RetryError,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("RetryError: failed after {} attempts (last status: {})", .{
            self.attempts,
            self.last_status,
        });
    }
};

test "isStatusRetryable" {
    try std.testing.expect(isStatusRetryable(500));
    try std.testing.expect(isStatusRetryable(502));
    try std.testing.expect(isStatusRetryable(503));
    try std.testing.expect(isStatusRetryable(429));
    try std.testing.expect(!isStatusRetryable(200));
    try std.testing.expect(!isStatusRetryable(404));
    try std.testing.expect(!isStatusRetryable(401));
}

test "calculateDelay" {
    const config = RetryConfig{ .jitter = false };
    try std.testing.expectEqual(@as(u64, 100), calculateDelay(config, 0));
    try std.testing.expectEqual(@as(u64, 200), calculateDelay(config, 1));
    try std.testing.expectEqual(@as(u64, 400), calculateDelay(config, 2));
}
