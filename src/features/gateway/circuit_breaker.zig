//! Circuit breaker for gateway — thin wrapper around shared resilience module.

const types = @import("types.zig");
const resilience = @import("../../services/shared/resilience/circuit_breaker.zig");
const std = @import("std");

pub const CircuitBreakerState = types.CircuitBreakerState;
pub const CircuitBreakerConfig = types.CircuitBreakerConfig;

/// Gateway circuit breaker — single-threaded, no allocator needed.
pub const CircuitBreaker = struct {
    inner: resilience.SimpleCircuitBreaker,

    pub fn init(config: CircuitBreakerConfig) CircuitBreaker {
        return .{ .inner = resilience.SimpleCircuitBreaker.init(.{
            .failure_threshold = config.failure_threshold,
            .success_threshold = config.half_open_max_requests,
            .half_open_max_requests = config.half_open_max_requests,
            .timeout_ms = config.reset_timeout_ms,
            .name = "gateway",
        }) };
    }

    pub fn recordSuccess(self: *CircuitBreaker) void {
        self.inner.recordSuccess();
    }

    pub fn recordFailure(self: *CircuitBreaker, _: u128) void {
        self.inner.recordFailure();
    }

    pub fn isAllowed(self: *CircuitBreaker, _: u128) bool {
        return self.inner.canAttempt();
    }

    pub fn reset(self: *CircuitBreaker) void {
        self.inner.reset();
    }

    /// Force to half-open state (for testing).
    pub fn forceHalfOpen(self: *CircuitBreaker) void {
        self.inner.forceState(.half_open);
    }
};

test {
    std.testing.refAllDecls(@This());
}
