//! Shared Failover Types for GPU Backend Failover
//!
//! Canonical type definitions shared between the primary failover manager
//! (`failover.zig`) and the mega orchestrator failover (`mega/failover.zig`).
//! Both modules import from here to avoid type duplication.

const std = @import("std");
const backend_mod = @import("backend.zig");

/// Circuit breaker state.
pub const CircuitState = enum {
    /// Normal operation, requests pass through.
    closed,
    /// Backend failing, requests immediately fail over.
    open,
    /// After timeout, allow test requests.
    half_open,
};

/// Health status for a backend.
pub const BackendHealth = struct {
    backend: backend_mod.Backend,
    state: CircuitState = .closed,
    failure_count: u32 = 0,
    success_count: u32 = 0,
    last_failure_time: i64 = 0,
    last_success_time: i64 = 0,
    consecutive_failures: u32 = 0,
    consecutive_successes: u32 = 0,
    total_requests: u64 = 0,
    total_failures: u64 = 0,

    /// Calculate failure rate (0.0 to 1.0).
    pub fn failureRate(self: BackendHealth) f32 {
        if (self.total_requests == 0) return 0;
        return @as(f32, @floatFromInt(self.total_failures)) / @as(f32, @floatFromInt(self.total_requests));
    }

    /// Check if backend is healthy (closed circuit, low consecutive failures).
    pub fn isHealthy(self: BackendHealth) bool {
        return self.state == .closed and self.consecutive_failures < 3;
    }
};

/// Reason for a failover.
///
/// Superset of reasons used by both the primary failover manager and
/// the mega orchestrator failover.
pub const FailoverReason = enum {
    // Mega orchestrator reasons
    circuit_open,
    timeout,
    error_threshold,
    manual,
    health_check,
    // Primary failover manager reasons
    device_lost,
    repeated_failures,
    performance_degradation,
    memory_exhausted,
};

/// A failover event for logging/metrics (mega-style simple record).
pub const FailoverEvent = struct {
    timestamp: i64,
    from_backend: backend_mod.Backend,
    to_backend: backend_mod.Backend,
    reason: FailoverReason,
    success: bool,
};

/// Aggregate failover statistics (mega-style).
pub const FailoverStats = struct {
    total_failovers: u64 = 0,
    successful_failovers: u64 = 0,
    failed_failovers: u64 = 0,
    current_primary: ?backend_mod.Backend = null,
    backends_available: u32 = 0,
    backends_unavailable: u32 = 0,
};

test "circuit state transitions" {
    const state: CircuitState = .closed;
    try std.testing.expectEqual(CircuitState.closed, state);
}

test "backend health defaults" {
    const health = BackendHealth{ .backend = .cuda };
    try std.testing.expect(health.isHealthy());
    try std.testing.expectEqual(@as(f32, 0), health.failureRate());
}

test "failover reason coverage" {
    // Verify all variants are accessible
    const reasons = [_]FailoverReason{
        .circuit_open,
        .timeout,
        .error_threshold,
        .manual,
        .health_check,
        .device_lost,
        .repeated_failures,
        .performance_degradation,
        .memory_exhausted,
    };
    try std.testing.expectEqual(@as(usize, 9), reasons.len);
}
