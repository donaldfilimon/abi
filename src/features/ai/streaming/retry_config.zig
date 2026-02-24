//! Streaming-specific retry configuration.
//!
//! Provides retry settings optimized for LLM streaming operations,
//! including token-level timeouts and backend-specific retry policies.

const std = @import("std");
const retry = @import("../../network/retry.zig");

/// Streaming-specific retry configuration.
///
/// Extends the base network retry config with streaming-specific
/// settings like token timeouts and total stream duration limits.
pub const StreamingRetryConfig = struct {
    /// Base retry config for connection/backend operations.
    /// Defaults optimized for streaming: fewer retries, shorter delays.
    base: retry.RetryConfig = .{
        .max_retries = 3,
        .initial_delay_ns = 100_000_000, // 100ms
        .max_delay_ns = 5_000_000_000, // 5s (shorter than default 30s)
        .multiplier = 2.0,
        .jitter = true,
        .jitter_factor = 0.25,
        .total_timeout_ns = 0,
    },

    /// Enable retry for streaming operations.
    enabled: bool = true,

    /// Timeout for single token generation (milliseconds).
    /// If a token takes longer than this, the stream is considered stalled.
    token_timeout_ms: u64 = 30_000, // 30s per token

    /// Timeout for entire stream (milliseconds, 0 = unlimited).
    /// Maximum wall-clock time for a complete streaming response.
    total_timeout_ms: u64 = 300_000, // 5 minutes total

    /// Backend-specific timeout for initial connection (milliseconds).
    /// Time to wait for backend to start generating tokens.
    backend_timeout_ms: u64 = 60_000, // 60s for first token

    /// WebSocket ping/pong timeout (milliseconds).
    /// Used for WebSocket health checking.
    websocket_timeout_ms: u64 = 30_000, // 30s

    /// Create a config optimized for local backends (faster, lower latency).
    pub fn forLocalBackend() StreamingRetryConfig {
        return .{
            .base = .{
                .max_retries = 2,
                .initial_delay_ns = 50_000_000, // 50ms
                .max_delay_ns = 1_000_000_000, // 1s
                .multiplier = 2.0,
                .jitter = true,
                .jitter_factor = 0.1,
                .total_timeout_ns = 0,
            },
            .token_timeout_ms = 10_000, // 10s per token
            .backend_timeout_ms = 10_000, // 10s for first token
        };
    }

    /// Create a config optimized for external API backends (more tolerant).
    pub fn forExternalBackend() StreamingRetryConfig {
        return .{
            .base = .{
                .max_retries = 3,
                .initial_delay_ns = 200_000_000, // 200ms
                .max_delay_ns = 10_000_000_000, // 10s
                .multiplier = 2.0,
                .jitter = true,
                .jitter_factor = 0.25,
                .total_timeout_ns = 0,
            },
            .token_timeout_ms = 60_000, // 60s per token (APIs can be slow)
            .backend_timeout_ms = 120_000, // 120s for first token
        };
    }

    /// Convert token timeout to nanoseconds.
    pub fn tokenTimeoutNs(self: StreamingRetryConfig) u64 {
        return self.token_timeout_ms * std.time.ns_per_ms;
    }

    /// Convert total timeout to nanoseconds.
    pub fn totalTimeoutNs(self: StreamingRetryConfig) u64 {
        return self.total_timeout_ms * std.time.ns_per_ms;
    }

    /// Convert backend timeout to nanoseconds.
    pub fn backendTimeoutNs(self: StreamingRetryConfig) u64 {
        return self.backend_timeout_ms * std.time.ns_per_ms;
    }
};

/// Errors that are considered retryable for streaming operations.
pub const StreamingRetryableErrors = struct {
    /// Retry on connection errors (connection refused, reset, unreachable).
    connection: bool = true,

    /// Retry on timeout errors.
    timeout: bool = true,

    /// Retry on server errors (5xx).
    server_error: bool = true,

    /// Retry on rate limiting (429).
    rate_limited: bool = true,

    /// Retry on temporary failures.
    temporary: bool = true,

    /// Check if an error should be retried based on configuration.
    pub fn shouldRetry(self: StreamingRetryableErrors, err: StreamingError) bool {
        return switch (err) {
            error.ConnectionRefused,
            error.ConnectionReset,
            error.NetworkUnreachable,
            error.HostUnreachable,
            => self.connection,

            error.Timeout,
            error.TokenTimeout,
            error.BackendTimeout,
            => self.timeout,

            error.ServerError,
            error.BackendUnavailable,
            => self.server_error,

            error.RateLimited,
            error.TooManyRequests,
            => self.rate_limited,

            error.TemporaryFailure,
            => self.temporary,

            else => false,
        };
    }
};

/// Streaming-specific error types.
pub const StreamingError = error{
    // Connection errors
    ConnectionRefused,
    ConnectionReset,
    NetworkUnreachable,
    HostUnreachable,

    // Timeout errors
    Timeout,
    TokenTimeout,
    BackendTimeout,

    // Server errors
    ServerError,
    BackendUnavailable,

    // Rate limiting
    RateLimited,
    TooManyRequests,

    // Temporary failures
    TemporaryFailure,

    // Circuit breaker
    CircuitBreakerOpen,

    // Stream-specific
    StreamInterrupted,
    InvalidSessionId,
};

// ============================================================================
// Tests
// ============================================================================

test "StreamingRetryConfig defaults" {
    const config = StreamingRetryConfig{};

    try std.testing.expectEqual(@as(u32, 3), config.base.max_retries);
    try std.testing.expectEqual(@as(u64, 100_000_000), config.base.initial_delay_ns);
    try std.testing.expectEqual(@as(u64, 30_000), config.token_timeout_ms);
    try std.testing.expectEqual(@as(u64, 300_000), config.total_timeout_ms);
    try std.testing.expect(config.enabled);
}

test "StreamingRetryConfig forLocalBackend" {
    const config = StreamingRetryConfig.forLocalBackend();

    try std.testing.expectEqual(@as(u32, 2), config.base.max_retries);
    try std.testing.expectEqual(@as(u64, 10_000), config.token_timeout_ms);
    try std.testing.expectEqual(@as(u64, 10_000), config.backend_timeout_ms);
}

test "StreamingRetryConfig forExternalBackend" {
    const config = StreamingRetryConfig.forExternalBackend();

    try std.testing.expectEqual(@as(u32, 3), config.base.max_retries);
    try std.testing.expectEqual(@as(u64, 60_000), config.token_timeout_ms);
    try std.testing.expectEqual(@as(u64, 120_000), config.backend_timeout_ms);
}

test "StreamingRetryConfig timeout conversions" {
    const config = StreamingRetryConfig{};

    try std.testing.expectEqual(@as(u64, 30_000_000_000), config.tokenTimeoutNs());
    try std.testing.expectEqual(@as(u64, 300_000_000_000), config.totalTimeoutNs());
    try std.testing.expectEqual(@as(u64, 60_000_000_000), config.backendTimeoutNs());
}

test "StreamingRetryableErrors shouldRetry" {
    const errors = StreamingRetryableErrors{};

    try std.testing.expect(errors.shouldRetry(error.ConnectionRefused));
    try std.testing.expect(errors.shouldRetry(error.Timeout));
    try std.testing.expect(errors.shouldRetry(error.ServerError));
    try std.testing.expect(errors.shouldRetry(error.RateLimited));
    try std.testing.expect(!errors.shouldRetry(error.CircuitBreakerOpen));
    try std.testing.expect(!errors.shouldRetry(error.InvalidSessionId));
}

test {
    std.testing.refAllDecls(@This());
}
