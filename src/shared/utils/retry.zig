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

/// Extract the error type from a function's return type.
/// Used to infer error types for generic retry operations.
fn FuncErrorType(comptime func: anytype) type {
    const FuncType = @TypeOf(func);
    const func_info = @typeInfo(FuncType);
    if (func_info != .@"fn") {
        @compileError("Expected a function type");
    }
    const ReturnType = func_info.@"fn".return_type orelse @compileError("Function must have a return type");
    const return_info = @typeInfo(ReturnType);
    if (return_info != .error_union) {
        @compileError("Function must return an error union");
    }
    return return_info.error_union.error_set;
}

/// Execute a function with retry logic and exponential backoff.
/// The `func` parameter should be a function that returns `!T` (an error union).
/// The `is_retryable` callback determines which errors should trigger a retry.
/// If `is_retryable` is null, all errors are considered retryable.
pub fn retryWithBackoff(
    comptime T: type,
    comptime func: anytype,
    args: anytype,
    config: RetryConfig,
    comptime is_retryable: ?*const fn (FuncErrorType(func)) bool,
) FuncErrorType(func)!T {
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

/// HTTP-related errors that are commonly retryable.
pub const HttpRetryableError = error{
    ConnectionRefused,
    ConnectionResetByPeer,
    ConnectionTimedOut,
    Timeout,
    TemporaryNameServerFailure,
    UnexpectedConnectFailure,
};

/// Common retryable error checker for HTTP operations.
/// This function checks if an error is in the set of known HTTP retryable errors.
/// For use with functions that return errors compatible with HttpRetryableError.
pub fn isHttpRetryable(comptime E: type) *const fn (E) bool {
    return struct {
        fn check(err: E) bool {
            // Check against known retryable error values
            inline for (@typeInfo(HttpRetryableError).error_set.?) |retryable_err| {
                if (@typeInfo(E) == .error_set) {
                    inline for (@typeInfo(E).error_set.?) |e| {
                        if (std.mem.eql(u8, e.name, retryable_err.name) and @intFromError(err) == @intFromError(@field(E, e.name))) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }
    }.check;
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

    const TestError = error{TemporaryFailure};

    var call_count: u32 = 0;
    const result = try retryWithBackoff(
        u32,
        struct {
            fn testFunc(count: *u32) TestError!u32 {
                count.* += 1;
                if (count.* < 3) return error.TemporaryFailure;
                return 42;
            }
        }.testFunc,
        .{&call_count},
        config,
        struct {
            fn isRetryable(err: TestError) bool {
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

    const TestError = error{AlwaysFails};

    var call_count: u32 = 0;
    const result = retryWithBackoff(
        u32,
        struct {
            fn testFunc(count: *u32) TestError!u32 {
                count.* += 1;
                return error.AlwaysFails;
            }
        }.testFunc,
        .{&call_count},
        config,
        struct {
            fn isRetryable(err: TestError) bool {
                return err == error.AlwaysFails;
            }
        }.isRetryable,
    );

    try std.testing.expectError(error.AlwaysFails, result);
    try std.testing.expectEqual(@as(u32, 3), call_count); // Initial + 2 retries
}

test "http retryable error detection" {
    // Test with a mixed error set containing both retryable and non-retryable errors
    const MixedError = error{ ConnectionRefused, Timeout, OutOfMemory };
    const checker = isHttpRetryable(MixedError);
    try std.testing.expect(checker(error.ConnectionRefused));
    try std.testing.expect(checker(error.Timeout));
    try std.testing.expect(!checker(error.OutOfMemory));
}

test "status code retryable detection" {
    try std.testing.expect(isStatusRetryable(429)); // Rate limit
    try std.testing.expect(isStatusRetryable(500)); // Server error
    try std.testing.expect(isStatusRetryable(503)); // Service unavailable
    try std.testing.expect(!isStatusRetryable(200)); // Success
    try std.testing.expect(!isStatusRetryable(404)); // Not found
    try std.testing.expect(!isStatusRetryable(401)); // Unauthorized
}
