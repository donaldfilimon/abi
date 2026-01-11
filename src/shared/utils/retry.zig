//! Retry logic with exponential backoff for handling transient failures.
//!
//! Provides configurable retry strategies with exponential backoff,
//! jitter, and maximum attempt limits.

const std = @import("std");
const time = @import("time.zig");

pub const RetryConfig = struct {
    /// Maximum number of retry attempts (0 = no retries)
    max_attempts: u32 = 3,
    /// Initial backoff duration in milliseconds
    initial_backoff_ms: u64 = 100,
    /// Maximum backoff duration in milliseconds
    max_backoff_ms: u64 = 30_000,
    /// Backoff multiplier for exponential growth
    backoff_multiplier: f32 = 2.0,
    /// Add random jitter to prevent thundering herd
    enable_jitter: bool = true,
};

pub const RetryError = error{
    MaxAttemptsExceeded,
    NonRetryableError,
};

/// Execute a function with retry logic and exponential backoff
pub fn retryWithBackoff(
    comptime T: type,
    comptime func: anytype,
    args: anytype,
    config: RetryConfig,
    comptime is_retryable: ?fn (anyerror) bool,
) !T {
    var attempt: u32 = 0;
    var backoff_ms = config.initial_backoff_ms;

    // Initialize random for jitter
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        std.posix.getrandom(std.mem.asBytes(&seed)) catch {
            // Fallback to timestamp if getrandom fails
            seed = @as(u64, @intCast(time.unixMilliseconds()));
        };
        break :blk seed;
    });
    const random = prng.random();

    while (attempt <= config.max_attempts) : (attempt += 1) {
        // Try the function
        const result = @call(.auto, func, args) catch |err| {
            // Check if error is retryable
            if (is_retryable) |check_fn| {
                if (!check_fn(err)) {
                    return RetryError.NonRetryableError;
                }
            }

            // If this was the last attempt, return the error
            if (attempt >= config.max_attempts) {
                return err;
            }

            // Calculate backoff with optional jitter
            const actual_backoff = if (config.enable_jitter) blk: {
                // Add jitter: random value between 50% and 100% of backoff
                const jitter_min = backoff_ms / 2;
                const jitter_range = backoff_ms - jitter_min;
                const jitter = random.intRangeAtMost(u64, 0, jitter_range);
                break :blk jitter_min + jitter;
            } else backoff_ms;

            // Sleep for backoff duration
            time.sleepMs(actual_backoff);

            // Increase backoff for next attempt (exponential)
            backoff_ms = @min(
                @as(u64, @intFromFloat(@as(f64, @floatFromInt(backoff_ms)) * config.backoff_multiplier)),
                config.max_backoff_ms,
            );

            continue;
        };

        // Success!
        return result;
    }

    return RetryError.MaxAttemptsExceeded;
}

/// Common retryable error checker for HTTP operations
pub fn isHttpRetryable(err: anyerror) bool {
    return switch (err) {
        error.ConnectionRefused,
        error.ConnectionResetByPeer,
        error.ConnectionTimedOut,
        error.Timeout,
        error.TemporaryNameServerFailure,
        error.UnexpectedConnectFailure,
        => true,
        else => false,
    };
}

/// Retryable status codes (5xx server errors, 429 rate limit)
pub fn isStatusRetryable(status_code: u16) bool {
    return status_code == 429 or // Rate limit
        status_code == 500 or // Internal server error
        status_code == 502 or // Bad gateway
        status_code == 503 or // Service unavailable
        status_code == 504; // Gateway timeout
}

test "retry with backoff succeeds on first attempt" {
    const config = RetryConfig{
        .max_attempts = 3,
        .initial_backoff_ms = 10,
    };

    var call_count: u32 = 0;
    const result = try retryWithBackoff(
        u32,
        struct {
            fn testFunc(count: *u32) !u32 {
                count.* += 1;
                return 42;
            }
        }.testFunc,
        .{&call_count},
        config,
        null,
    );

    try std.testing.expectEqual(@as(u32, 42), result);
    try std.testing.expectEqual(@as(u32, 1), call_count);
}

test "retry with backoff succeeds after retries" {
    const config = RetryConfig{
        .max_attempts = 3,
        .initial_backoff_ms = 1,
        .enable_jitter = false,
    };

    var call_count: u32 = 0;
    const result = try retryWithBackoff(
        u32,
        struct {
            fn testFunc(count: *u32) !u32 {
                count.* += 1;
                if (count.* < 3) return error.TemporaryFailure;
                return 42;
            }
        }.testFunc,
        .{&call_count},
        config,
        struct {
            fn isRetryable(err: anyerror) bool {
                return err == error.TemporaryFailure;
            }
        }.isRetryable,
    );

    try std.testing.expectEqual(@as(u32, 42), result);
    try std.testing.expectEqual(@as(u32, 3), call_count);
}

test "retry with backoff fails after max attempts" {
    const config = RetryConfig{
        .max_attempts = 2,
        .initial_backoff_ms = 1,
        .enable_jitter = false,
    };

    var call_count: u32 = 0;
    const result = retryWithBackoff(
        u32,
        struct {
            fn testFunc(count: *u32) !u32 {
                count.* += 1;
                return error.AlwaysFails;
            }
        }.testFunc,
        .{&call_count},
        config,
        struct {
            fn isRetryable(err: anyerror) bool {
                return err == error.AlwaysFails;
            }
        }.isRetryable,
    );

    try std.testing.expectError(error.AlwaysFails, result);
    try std.testing.expectEqual(@as(u32, 3), call_count); // Initial + 2 retries
}

test "http retryable error detection" {
    try std.testing.expect(isHttpRetryable(error.ConnectionRefused));
    try std.testing.expect(isHttpRetryable(error.Timeout));
    try std.testing.expect(!isHttpRetryable(error.OutOfMemory));
}

test "status code retryable detection" {
    try std.testing.expect(isStatusRetryable(429)); // Rate limit
    try std.testing.expect(isStatusRetryable(500)); // Server error
    try std.testing.expect(isStatusRetryable(503)); // Service unavailable
    try std.testing.expect(!isStatusRetryable(200)); // Success
    try std.testing.expect(!isStatusRetryable(404)); // Not found
    try std.testing.expect(!isStatusRetryable(401)); // Unauthorized
}

