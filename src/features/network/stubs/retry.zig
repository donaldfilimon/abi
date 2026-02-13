const std = @import("std");

pub const RetryConfig = struct {
    max_retries: u32 = 3,
    initial_delay_ms: u64 = 100,
    max_delay_ms: u64 = 30_000,
    backoff_multiplier: f64 = 2.0,
};

pub const RetryResult = struct {
    success: bool = false,
    attempts: u32 = 0,
    total_delay_ms: u64 = 0,
};

pub const RetryError = error{
    NetworkDisabled,
    MaxRetriesExceeded,
    NonRetryableError,
};

pub const RetryStrategy = enum { fixed, exponential, linear, jittered };

pub const RetryExecutor = struct {
    pub fn init(_: std.mem.Allocator, _: RetryConfig) @This() {
        return .{};
    }
    pub fn deinit(_: *@This()) void {}
};

pub const RetryableErrors = struct {
    pub fn isRetryable(_: anyerror) bool {
        return false;
    }
};

pub const BackoffCalculator = struct {
    pub fn calculate(_: RetryConfig, _: u32) u64 {
        return 0;
    }
};

pub fn retry(_: anytype) !void {
    return error.NetworkDisabled;
}

pub fn retryWithStrategy(_: RetryStrategy, _: anytype) !void {
    return error.NetworkDisabled;
}
