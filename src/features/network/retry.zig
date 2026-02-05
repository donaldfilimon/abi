//! Retry mechanism with exponential backoff.
//!
//! Provides configurable retry logic for network operations with
//! exponential backoff, jitter, and retry condition filtering.

const std = @import("std");
const time = @import("../../services/shared/time.zig");

/// Comprehensive error set for retry operations
pub const RetryableError = error{
    // Connection errors
    ConnectionRefused,
    ConnectionResetByPeer,
    NetworkUnreachable,
    HostUnreachable,
    // Timeout errors
    Timeout,
    TimedOut,
    TimerFailed,
    // Server errors
    ServerError,
    InternalServerError,
    BadGateway,
    ServiceUnavailable,
    // Rate limiting
    TooManyRequests,
    RateLimited,
    // Temporary failures
    TemporaryFailure,
    TryAgain,
    // Other
    Unknown,
};

/// Retry configuration.
pub const RetryConfig = struct {
    /// Maximum number of retry attempts.
    max_retries: u32 = 3,
    /// Initial delay between retries (nanoseconds).
    initial_delay_ns: u64 = 100_000_000, // 100ms
    /// Maximum delay between retries (nanoseconds).
    max_delay_ns: u64 = 30_000_000_000, // 30s
    /// Backoff multiplier.
    multiplier: f64 = 2.0,
    /// Add random jitter to delays.
    jitter: bool = true,
    /// Jitter factor (0.0 to 1.0).
    jitter_factor: f64 = 0.25,
    /// Timeout for entire retry sequence (nanoseconds, 0 = no limit).
    total_timeout_ns: u64 = 0,
};

/// Retry result.
pub fn RetryResult(comptime T: type) type {
    return union(enum) {
        success: T,
        failure: RetryError,

        pub fn isSuccess(self: @This()) bool {
            return self == .success;
        }

        pub fn unwrap(self: @This()) !T {
            return switch (self) {
                .success => |v| v,
                .failure => |e| e.last_error,
            };
        }
    };
}

/// Retry error details.
pub const RetryError = struct {
    attempts: u32,
    last_error: RetryableError,
    elapsed_ns: u64,
    errors: [16]?RetryableError = [_]?RetryableError{null} ** 16,
};

/// Retry strategy.
pub const RetryStrategy = enum {
    /// Exponential backoff.
    exponential,
    /// Linear backoff.
    linear,
    /// Fixed delay.
    fixed,
    /// Immediate retry (no delay).
    immediate,
    /// Fibonacci backoff.
    fibonacci,
};

/// Retryable error categories.
pub const RetryableErrors = struct {
    /// Network connection errors.
    connection: bool = true,
    /// Timeout errors.
    timeout: bool = true,
    /// Server errors (5xx).
    server_error: bool = true,
    /// Rate limiting (429).
    rate_limited: bool = true,
    /// Temporary failures.
    temporary: bool = true,

    /// Check if an error should be retried.
    pub fn shouldRetry(self: RetryableErrors, err: RetryableError) bool {
        // Map errors to categories
        return switch (err) {
            error.ConnectionRefused,
            error.ConnectionResetByPeer,
            error.NetworkUnreachable,
            error.HostUnreachable,
            => self.connection,

            error.Timeout,
            error.TimedOut,
            => self.timeout,

            error.ServerError,
            error.InternalServerError,
            error.BadGateway,
            error.ServiceUnavailable,
            => self.server_error,

            error.TooManyRequests,
            error.RateLimited,
            => self.rate_limited,

            error.TemporaryFailure,
            error.TryAgain,
            => self.temporary,

            else => false,
        };
    }
};

fn applyJitter(prng: *std.Random.DefaultPrng, base_delay: u64, jitter: bool, jitter_factor: f64) u64 {
    if (!jitter or base_delay == 0) return base_delay;

    const jitter_range = @as(u64, @intFromFloat(@as(f64, @floatFromInt(base_delay)) * jitter_factor));
    if (jitter_range == 0) return base_delay;

    var adjusted = base_delay;
    const jitter_value = prng.random().uintLessThan(u64, jitter_range * 2);
    if (jitter_value < jitter_range) {
        adjusted -|= jitter_value;
    } else {
        adjusted +|= jitter_value - jitter_range;
    }

    return adjusted;
}

/// Retry executor with configurable strategy.
pub fn RetryExecutor(comptime T: type) type {
    return struct {
        const Self = @This();

        config: RetryConfig,
        strategy: RetryStrategy,
        retryable: RetryableErrors,
        prng: std.Random.DefaultPrng,

        /// Initialize retry executor.
        pub fn init(config: RetryConfig, strategy: RetryStrategy) Self {
            return .{
                .config = config,
                .strategy = strategy,
                .retryable = .{},
                .prng = std.Random.DefaultPrng.init(time.getSeed()),
            };
        }

        /// Initialize with custom retryable errors.
        pub fn initWithErrors(config: RetryConfig, strategy: RetryStrategy, retryable: RetryableErrors) Self {
            return .{
                .config = config,
                .strategy = strategy,
                .retryable = retryable,
                .prng = std.Random.DefaultPrng.init(time.getSeed()),
            };
        }

        /// Execute operation with retry logic.
        pub fn execute(self: *Self, operation: *const fn () RetryableError!T) RetryResult(T) {
            var timer = std.time.Timer.start() catch {
                return .{ .failure = .{
                    .attempts = 0,
                    .last_error = error.TimerFailed,
                    .elapsed_ns = 0,
                } };
            };

            var attempts: u32 = 0;
            var last_error: RetryableError = error.Unknown;
            var errors: [16]?RetryableError = [_]?RetryableError{null} ** 16;

            while (attempts <= self.config.max_retries) {
                // Check total timeout
                const elapsed = timer.read();
                if (self.config.total_timeout_ns > 0 and elapsed >= self.config.total_timeout_ns) {
                    return .{ .failure = .{
                        .attempts = attempts,
                        .last_error = error.Timeout,
                        .elapsed_ns = elapsed,
                        .errors = errors,
                    } };
                }

                // Execute operation
                if (operation()) |result| {
                    return .{ .success = result };
                } else |err| {
                    last_error = err;
                    if (attempts < 16) {
                        errors[attempts] = err;
                    }

                    // Check if error is retryable
                    if (!self.retryable.shouldRetry(err)) {
                        return .{ .failure = .{
                            .attempts = attempts + 1,
                            .last_error = err,
                            .elapsed_ns = timer.read(),
                            .errors = errors,
                        } };
                    }

                    attempts += 1;

                    // Calculate delay
                    if (attempts <= self.config.max_retries) {
                        const delay = self.calculateDelay(attempts);
                        time.sleepNs(delay);
                    }
                }
            }

            return .{ .failure = .{
                .attempts = attempts,
                .last_error = last_error,
                .elapsed_ns = timer.read(),
                .errors = errors,
            } };
        }

        /// Execute with context parameter.
        pub fn executeWithContext(
            self: *Self,
            comptime Context: type,
            ctx: Context,
            operation: *const fn (Context) RetryableError!T,
        ) RetryResult(T) {
            var timer = std.time.Timer.start() catch {
                return .{ .failure = .{
                    .attempts = 0,
                    .last_error = error.TimerFailed,
                    .elapsed_ns = 0,
                } };
            };

            var attempts: u32 = 0;
            var last_error: RetryableError = error.Unknown;
            var errors: [16]?RetryableError = [_]?RetryableError{null} ** 16;

            while (attempts <= self.config.max_retries) {
                const elapsed = timer.read();
                if (self.config.total_timeout_ns > 0 and elapsed >= self.config.total_timeout_ns) {
                    return .{ .failure = .{
                        .attempts = attempts,
                        .last_error = error.Timeout,
                        .elapsed_ns = elapsed,
                        .errors = errors,
                    } };
                }

                if (operation(ctx)) |result| {
                    return .{ .success = result };
                } else |err| {
                    last_error = err;
                    if (attempts < 16) {
                        errors[attempts] = err;
                    }

                    if (!self.retryable.shouldRetry(err)) {
                        return .{ .failure = .{
                            .attempts = attempts + 1,
                            .last_error = err,
                            .elapsed_ns = timer.read(),
                            .errors = errors,
                        } };
                    }

                    attempts += 1;

                    if (attempts <= self.config.max_retries) {
                        const delay = self.calculateDelay(attempts);
                        time.sleepNs(delay);
                    }
                }
            }

            return .{ .failure = .{
                .attempts = attempts,
                .last_error = last_error,
                .elapsed_ns = timer.read(),
                .errors = errors,
            } };
        }

        fn calculateDelay(self: *Self, attempt: u32) u64 {
            var base_delay: u64 = switch (self.strategy) {
                .exponential => blk: {
                    const exp: u6 = @intCast(@min(attempt - 1, 62));
                    const factor = std.math.pow(f64, self.config.multiplier, @floatFromInt(exp));
                    break :blk @intFromFloat(@as(f64, @floatFromInt(self.config.initial_delay_ns)) * factor);
                },
                .linear => self.config.initial_delay_ns * attempt,
                .fixed => self.config.initial_delay_ns,
                .immediate => 0,
                .fibonacci => blk: {
                    var a: u64 = self.config.initial_delay_ns;
                    var b: u64 = self.config.initial_delay_ns;
                    var i: u32 = 1;
                    while (i < attempt) : (i += 1) {
                        const next = a +| b;
                        a = b;
                        b = next;
                    }
                    break :blk b;
                },
            };

            // Apply max delay cap
            base_delay = @min(base_delay, self.config.max_delay_ns);

            base_delay = applyJitter(&self.prng, base_delay, self.config.jitter, self.config.jitter_factor);

            return base_delay;
        }
    };
}

/// Simple retry function for one-off retries.
/// The operation must return `RetryableError!T` for proper error categorization.
pub fn retry(
    comptime T: type,
    operation: *const fn () RetryableError!T,
    config: RetryConfig,
) RetryResult(T) {
    var executor = RetryExecutor(T).init(config, .exponential);
    return executor.execute(operation);
}

/// Retry with custom strategy.
/// The operation must return `RetryableError!T` for proper error categorization.
pub fn retryWithStrategy(
    comptime T: type,
    operation: *const fn () RetryableError!T,
    config: RetryConfig,
    strategy: RetryStrategy,
) RetryResult(T) {
    var executor = RetryExecutor(T).init(config, strategy);
    return executor.execute(operation);
}

/// Backoff calculator for external use.
pub const BackoffCalculator = struct {
    config: RetryConfig,
    strategy: RetryStrategy,
    attempt: u32,
    prng: std.Random.DefaultPrng,

    pub fn init(config: RetryConfig, strategy: RetryStrategy) BackoffCalculator {
        return .{
            .config = config,
            .strategy = strategy,
            .attempt = 0,
            .prng = std.Random.DefaultPrng.init(time.getSeed()),
        };
    }

    /// Get next delay and increment attempt counter.
    pub fn nextDelay(self: *BackoffCalculator) u64 {
        self.attempt += 1;
        return self.getDelay(self.attempt);
    }

    /// Get delay for specific attempt.
    pub fn getDelay(self: *BackoffCalculator, attempt: u32) u64 {
        if (attempt == 0) return 0;

        var base_delay: u64 = switch (self.strategy) {
            .exponential => blk: {
                const exp: u6 = @intCast(@min(attempt - 1, 62));
                const factor = std.math.pow(f64, self.config.multiplier, @floatFromInt(exp));
                break :blk @intFromFloat(@as(f64, @floatFromInt(self.config.initial_delay_ns)) * factor);
            },
            .linear => self.config.initial_delay_ns * attempt,
            .fixed => self.config.initial_delay_ns,
            .immediate => 0,
            .fibonacci => blk: {
                var a: u64 = self.config.initial_delay_ns;
                var b: u64 = self.config.initial_delay_ns;
                var i: u32 = 1;
                while (i < attempt) : (i += 1) {
                    const next = a +| b;
                    a = b;
                    b = next;
                }
                break :blk b;
            },
        };

        base_delay = @min(base_delay, self.config.max_delay_ns);

        base_delay = applyJitter(&self.prng, base_delay, self.config.jitter, self.config.jitter_factor);

        return base_delay;
    }

    /// Reset attempt counter.
    pub fn reset(self: *BackoffCalculator) void {
        self.attempt = 0;
    }

    /// Check if max retries reached.
    pub fn isExhausted(self: *const BackoffCalculator) bool {
        return self.attempt >= self.config.max_retries;
    }
};

test "retry success on first attempt" {
    const op = struct {
        fn call() RetryableError!u32 {
            return 42;
        }
    };

    const result = retry(u32, op.call, .{ .max_retries = 3 });
    try std.testing.expect(result.isSuccess());
    try std.testing.expectEqual(@as(u32, 42), try result.unwrap());
}

test "backoff calculator exponential" {
    var calc = BackoffCalculator.init(.{
        .initial_delay_ns = 1_000_000,
        .multiplier = 2.0,
        .jitter = false,
    }, .exponential);

    const d1 = calc.getDelay(1);
    const d2 = calc.getDelay(2);
    const d3 = calc.getDelay(3);

    try std.testing.expectEqual(@as(u64, 1_000_000), d1);
    try std.testing.expectEqual(@as(u64, 2_000_000), d2);
    try std.testing.expectEqual(@as(u64, 4_000_000), d3);
}

test "backoff calculator linear" {
    var calc = BackoffCalculator.init(.{
        .initial_delay_ns = 1_000_000,
        .jitter = false,
    }, .linear);

    const d1 = calc.getDelay(1);
    const d2 = calc.getDelay(2);
    const d3 = calc.getDelay(3);

    try std.testing.expectEqual(@as(u64, 1_000_000), d1);
    try std.testing.expectEqual(@as(u64, 2_000_000), d2);
    try std.testing.expectEqual(@as(u64, 3_000_000), d3);
}

test "backoff max delay cap" {
    var calc = BackoffCalculator.init(.{
        .initial_delay_ns = 1_000_000,
        .max_delay_ns = 5_000_000,
        .multiplier = 10.0,
        .jitter = false,
    }, .exponential);

    const d1 = calc.getDelay(1);
    const d10 = calc.getDelay(10);

    try std.testing.expectEqual(@as(u64, 1_000_000), d1);
    try std.testing.expectEqual(@as(u64, 5_000_000), d10); // Capped
}

test "retryable errors" {
    const retryable = RetryableErrors{};

    // These errors should be retryable by default
    try std.testing.expect(retryable.shouldRetry(error.ConnectionRefused));
    try std.testing.expect(retryable.shouldRetry(error.Timeout));

    // Unknown errors are not retryable by default (falls through to else branch)
    try std.testing.expect(!retryable.shouldRetry(error.Unknown));
    try std.testing.expect(!retryable.shouldRetry(error.TimerFailed));
}
