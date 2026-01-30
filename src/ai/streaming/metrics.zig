//! Streaming-specific metrics collection.
//!
//! Provides comprehensive observability for streaming operations including:
//! - Per-backend request counters and latency histograms
//! - Circuit breaker state tracking
//! - Session cache hit/miss rates
//! - Recovery success metrics
//!
//! ## Usage
//!
//! ```zig
//! var metrics = try StreamingMetrics.init(allocator);
//! defer metrics.deinit();
//!
//! // Record a successful stream
//! metrics.recordStreamStart(.openai);
//! metrics.recordTokenLatency(.openai, 150);
//! metrics.recordStreamComplete(.openai, 5000);
//!
//! // Record failures and recovery
//! metrics.recordStreamFailure(.openai);
//! metrics.recordRetryAttempt(.openai);
//! metrics.recordSessionCacheHit();
//! ```

const std = @import("std");
const observability = @import("../../observability/mod.zig");
const Counter = observability.Counter;
const Gauge = observability.Gauge;
const Histogram = observability.Histogram;
const backends = @import("backends/mod.zig");

/// Backend types for per-backend metrics (re-exported from backends).
pub const BackendType = backends.BackendType;

/// Per-backend metrics collection.
pub const BackendMetrics = struct {
    /// Total requests to this backend.
    requests_total: Counter,
    /// Successful requests.
    requests_success: Counter,
    /// Failed requests (before all retries exhausted).
    requests_failed: Counter,
    /// Retry attempts.
    requests_retried: Counter,
    /// Times circuit breaker opened for this backend.
    circuit_breaker_opens: Counter,
    /// Token generation latency in milliseconds.
    token_latency_ms: Histogram,
    /// Stream duration in milliseconds.
    stream_duration_ms: Histogram,
    /// Currently active streams.
    active_streams: Gauge,

    /// Initialize backend metrics with the given allocator.
    pub fn init(allocator: std.mem.Allocator, backend_name: []const u8) !BackendMetrics {
        // Latency buckets: 10ms, 50ms, 100ms, 250ms, 500ms, 1s, 2s, 5s, 10s
        const latency_bounds = [_]u64{ 10, 50, 100, 250, 500, 1000, 2000, 5000, 10000 };
        // Duration buckets: 100ms, 500ms, 1s, 5s, 10s, 30s, 60s, 120s, 300s
        const duration_bounds = [_]u64{ 100, 500, 1000, 5000, 10000, 30000, 60000, 120000, 300000 };

        return BackendMetrics{
            .requests_total = .{ .name = backend_name },
            .requests_success = .{ .name = backend_name },
            .requests_failed = .{ .name = backend_name },
            .requests_retried = .{ .name = backend_name },
            .circuit_breaker_opens = .{ .name = backend_name },
            .token_latency_ms = try Histogram.init(allocator, backend_name, @constCast(&latency_bounds)),
            .stream_duration_ms = try Histogram.init(allocator, backend_name, @constCast(&duration_bounds)),
            .active_streams = .{ .name = backend_name },
        };
    }

    pub fn deinit(self: *BackendMetrics, allocator: std.mem.Allocator) void {
        self.token_latency_ms.deinit(allocator);
        self.stream_duration_ms.deinit(allocator);
    }
};

/// Streaming metrics configuration.
pub const StreamingMetricsConfig = struct {
    /// Enable per-backend metrics.
    enable_backend_metrics: bool = true,
    /// Enable session cache metrics.
    enable_cache_metrics: bool = true,
    /// Enable recovery metrics.
    enable_recovery_metrics: bool = true,
};

/// Comprehensive streaming metrics collector.
pub const StreamingMetrics = struct {
    allocator: std.mem.Allocator,
    config: StreamingMetricsConfig,

    // Per-backend metrics
    backend_metrics: [4]BackendMetrics,

    // Global counters
    total_streams: Counter,
    total_tokens: Counter,
    total_errors: Counter,

    // Session cache metrics
    session_cache_entries: Gauge,
    session_cache_hits: Counter,
    session_cache_misses: Counter,
    session_cache_evictions: Counter,

    // Recovery metrics
    recovery_attempts: Counter,
    recovery_success: Counter,
    recovery_failed: Counter,
    partial_recovery_success: Counter,

    // Circuit breaker global stats
    circuit_breaker_total_opens: Counter,
    circuit_breaker_total_closes: Counter,

    const Self = @This();

    /// Initialize streaming metrics.
    pub fn init(allocator: std.mem.Allocator, config: StreamingMetricsConfig) !Self {
        var backend_metrics: [4]BackendMetrics = undefined;
        var initialized_count: usize = 0;
        errdefer {
            for (0..initialized_count) |i| {
                backend_metrics[i].deinit(allocator);
            }
        }

        const backend_names = [_][]const u8{ "local", "openai", "ollama", "anthropic" };
        for (backend_names, 0..) |name, i| {
            backend_metrics[i] = try BackendMetrics.init(allocator, name);
            initialized_count += 1;
        }

        return Self{
            .allocator = allocator,
            .config = config,
            .backend_metrics = backend_metrics,
            .total_streams = .{ .name = "streaming_total_streams" },
            .total_tokens = .{ .name = "streaming_total_tokens" },
            .total_errors = .{ .name = "streaming_total_errors" },
            .session_cache_entries = .{ .name = "streaming_session_cache_entries" },
            .session_cache_hits = .{ .name = "streaming_session_cache_hits" },
            .session_cache_misses = .{ .name = "streaming_session_cache_misses" },
            .session_cache_evictions = .{ .name = "streaming_session_cache_evictions" },
            .recovery_attempts = .{ .name = "streaming_recovery_attempts" },
            .recovery_success = .{ .name = "streaming_recovery_success" },
            .recovery_failed = .{ .name = "streaming_recovery_failed" },
            .partial_recovery_success = .{ .name = "streaming_partial_recovery_success" },
            .circuit_breaker_total_opens = .{ .name = "streaming_circuit_breaker_opens" },
            .circuit_breaker_total_closes = .{ .name = "streaming_circuit_breaker_closes" },
        };
    }

    /// Clean up resources.
    pub fn deinit(self: *Self) void {
        for (&self.backend_metrics) |*bm| {
            bm.deinit(self.allocator);
        }
    }

    /// Get metrics for a specific backend.
    fn getBackendMetrics(self: *Self, backend: BackendType) *BackendMetrics {
        return &self.backend_metrics[@intFromEnum(backend)];
    }

    // =========================================================================
    // Stream lifecycle metrics
    // =========================================================================

    /// Record that a stream has started.
    pub fn recordStreamStart(self: *Self, backend: BackendType) void {
        if (!self.config.enable_backend_metrics) return;

        const bm = self.getBackendMetrics(backend);
        bm.requests_total.inc(1);
        bm.active_streams.inc();
        self.total_streams.inc(1);
    }

    /// Record that a stream completed successfully.
    pub fn recordStreamComplete(self: *Self, backend: BackendType, duration_ms: u64) void {
        if (!self.config.enable_backend_metrics) return;

        const bm = self.getBackendMetrics(backend);
        bm.requests_success.inc(1);
        bm.stream_duration_ms.record(duration_ms);
        bm.active_streams.dec();
    }

    /// Record that a stream failed.
    pub fn recordStreamFailure(self: *Self, backend: BackendType) void {
        if (!self.config.enable_backend_metrics) return;

        const bm = self.getBackendMetrics(backend);
        bm.requests_failed.inc(1);
        bm.active_streams.dec();
        self.total_errors.inc(1);
    }

    /// Record token generation latency.
    pub fn recordTokenLatency(self: *Self, backend: BackendType, latency_ms: u64) void {
        if (!self.config.enable_backend_metrics) return;

        const bm = self.getBackendMetrics(backend);
        bm.token_latency_ms.record(latency_ms);
        self.total_tokens.inc(1);
    }

    // =========================================================================
    // Retry metrics
    // =========================================================================

    /// Record a retry attempt.
    pub fn recordRetryAttempt(self: *Self, backend: BackendType) void {
        if (!self.config.enable_backend_metrics) return;

        const bm = self.getBackendMetrics(backend);
        bm.requests_retried.inc(1);
    }

    // =========================================================================
    // Circuit breaker metrics
    // =========================================================================

    /// Record circuit breaker opening.
    pub fn recordCircuitBreakerOpen(self: *Self, backend: BackendType) void {
        if (!self.config.enable_backend_metrics) return;

        const bm = self.getBackendMetrics(backend);
        bm.circuit_breaker_opens.inc(1);
        self.circuit_breaker_total_opens.inc(1);
    }

    /// Record circuit breaker closing (recovery).
    pub fn recordCircuitBreakerClose(self: *Self, _: BackendType) void {
        if (!self.config.enable_backend_metrics) return;

        self.circuit_breaker_total_closes.inc(1);
    }

    // =========================================================================
    // Session cache metrics
    // =========================================================================

    /// Record a session cache hit.
    pub fn recordSessionCacheHit(self: *Self) void {
        if (!self.config.enable_cache_metrics) return;

        self.session_cache_hits.inc(1);
    }

    /// Record a session cache miss.
    pub fn recordSessionCacheMiss(self: *Self) void {
        if (!self.config.enable_cache_metrics) return;

        self.session_cache_misses.inc(1);
    }

    /// Record a session cache eviction.
    pub fn recordSessionCacheEviction(self: *Self) void {
        if (!self.config.enable_cache_metrics) return;

        self.session_cache_evictions.inc(1);
    }

    /// Update the current session cache entry count.
    pub fn setSessionCacheEntries(self: *Self, count: i64) void {
        if (!self.config.enable_cache_metrics) return;

        self.session_cache_entries.set(count);
    }

    // =========================================================================
    // Recovery metrics
    // =========================================================================

    /// Record a recovery attempt.
    pub fn recordRecoveryAttempt(self: *Self) void {
        if (!self.config.enable_recovery_metrics) return;

        self.recovery_attempts.inc(1);
    }

    /// Record a successful recovery.
    pub fn recordRecoverySuccess(self: *Self) void {
        if (!self.config.enable_recovery_metrics) return;

        self.recovery_success.inc(1);
    }

    /// Record a failed recovery.
    pub fn recordRecoveryFailure(self: *Self) void {
        if (!self.config.enable_recovery_metrics) return;

        self.recovery_failed.inc(1);
    }

    /// Record a partial recovery success (some tokens recovered).
    pub fn recordPartialRecovery(self: *Self) void {
        if (!self.config.enable_recovery_metrics) return;

        self.partial_recovery_success.inc(1);
    }

    // =========================================================================
    // Snapshot and export
    // =========================================================================

    /// Snapshot of streaming metrics at a point in time.
    pub const Snapshot = struct {
        // Global stats
        total_streams: u64,
        total_tokens: u64,
        total_errors: u64,

        // Per-backend stats (indexed by BackendType)
        backend_requests: [4]u64,
        backend_successes: [4]u64,
        backend_failures: [4]u64,
        backend_retries: [4]u64,
        backend_circuit_opens: [4]u64,
        backend_active_streams: [4]i64,

        // Cache stats
        cache_entries: i64,
        cache_hits: u64,
        cache_misses: u64,
        cache_evictions: u64,

        // Recovery stats
        recovery_attempts: u64,
        recovery_success: u64,
        recovery_failed: u64,
        partial_recoveries: u64,

        // Circuit breaker global stats
        circuit_opens: u64,
        circuit_closes: u64,

        /// Get cache hit rate (0.0 - 1.0).
        pub fn cacheHitRate(self: Snapshot) f64 {
            const total = self.cache_hits + self.cache_misses;
            if (total == 0) return 0.0;
            return @as(f64, @floatFromInt(self.cache_hits)) / @as(f64, @floatFromInt(total));
        }

        /// Get recovery success rate (0.0 - 1.0).
        pub fn recoverySuccessRate(self: Snapshot) f64 {
            if (self.recovery_attempts == 0) return 0.0;
            return @as(f64, @floatFromInt(self.recovery_success)) / @as(f64, @floatFromInt(self.recovery_attempts));
        }

        /// Get success rate for a specific backend (0.0 - 1.0).
        pub fn backendSuccessRate(self: Snapshot, backend: BackendType) f64 {
            const idx = @intFromEnum(backend);
            const total = self.backend_requests[idx];
            if (total == 0) return 0.0;
            return @as(f64, @floatFromInt(self.backend_successes[idx])) / @as(f64, @floatFromInt(total));
        }
    };

    /// Get a point-in-time snapshot of all metrics.
    pub fn snapshot(self: *const Self) Snapshot {
        var snap = Snapshot{
            .total_streams = self.total_streams.get(),
            .total_tokens = self.total_tokens.get(),
            .total_errors = self.total_errors.get(),
            .backend_requests = undefined,
            .backend_successes = undefined,
            .backend_failures = undefined,
            .backend_retries = undefined,
            .backend_circuit_opens = undefined,
            .backend_active_streams = undefined,
            .cache_entries = self.session_cache_entries.get(),
            .cache_hits = self.session_cache_hits.get(),
            .cache_misses = self.session_cache_misses.get(),
            .cache_evictions = self.session_cache_evictions.get(),
            .recovery_attempts = self.recovery_attempts.get(),
            .recovery_success = self.recovery_success.get(),
            .recovery_failed = self.recovery_failed.get(),
            .partial_recoveries = self.partial_recovery_success.get(),
            .circuit_opens = self.circuit_breaker_total_opens.get(),
            .circuit_closes = self.circuit_breaker_total_closes.get(),
        };

        for (0..4) |i| {
            snap.backend_requests[i] = self.backend_metrics[i].requests_total.get();
            snap.backend_successes[i] = self.backend_metrics[i].requests_success.get();
            snap.backend_failures[i] = self.backend_metrics[i].requests_failed.get();
            snap.backend_retries[i] = self.backend_metrics[i].requests_retried.get();
            snap.backend_circuit_opens[i] = self.backend_metrics[i].circuit_breaker_opens.get();
            snap.backend_active_streams[i] = self.backend_metrics[i].active_streams.get();
        }

        return snap;
    }

    /// Reset all metrics to zero.
    pub fn reset(self: *Self) void {
        self.total_streams.reset();
        self.total_tokens.reset();
        self.total_errors.reset();

        for (&self.backend_metrics) |*bm| {
            bm.requests_total.reset();
            bm.requests_success.reset();
            bm.requests_failed.reset();
            bm.requests_retried.reset();
            bm.circuit_breaker_opens.reset();
            bm.active_streams.set(0);
        }

        self.session_cache_entries.set(0);
        self.session_cache_hits.reset();
        self.session_cache_misses.reset();
        self.session_cache_evictions.reset();

        self.recovery_attempts.reset();
        self.recovery_success.reset();
        self.recovery_failed.reset();
        self.partial_recovery_success.reset();

        self.circuit_breaker_total_opens.reset();
        self.circuit_breaker_total_closes.reset();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "StreamingMetrics initialization" {
    const allocator = std.testing.allocator;
    var metrics = try StreamingMetrics.init(allocator, .{});
    defer metrics.deinit();

    const snap = metrics.snapshot();
    try std.testing.expectEqual(@as(u64, 0), snap.total_streams);
    try std.testing.expectEqual(@as(u64, 0), snap.total_tokens);
    try std.testing.expectEqual(@as(u64, 0), snap.total_errors);
}

test "StreamingMetrics stream lifecycle" {
    const allocator = std.testing.allocator;
    var metrics = try StreamingMetrics.init(allocator, .{});
    defer metrics.deinit();

    // Start a stream
    metrics.recordStreamStart(.openai);
    var snap = metrics.snapshot();
    try std.testing.expectEqual(@as(u64, 1), snap.total_streams);
    try std.testing.expectEqual(@as(u64, 1), snap.backend_requests[@intFromEnum(BackendType.openai)]);
    try std.testing.expectEqual(@as(i64, 1), snap.backend_active_streams[@intFromEnum(BackendType.openai)]);

    // Record tokens
    metrics.recordTokenLatency(.openai, 100);
    metrics.recordTokenLatency(.openai, 150);
    snap = metrics.snapshot();
    try std.testing.expectEqual(@as(u64, 2), snap.total_tokens);

    // Complete the stream
    metrics.recordStreamComplete(.openai, 5000);
    snap = metrics.snapshot();
    try std.testing.expectEqual(@as(u64, 1), snap.backend_successes[@intFromEnum(BackendType.openai)]);
    try std.testing.expectEqual(@as(i64, 0), snap.backend_active_streams[@intFromEnum(BackendType.openai)]);
}

test "StreamingMetrics failure and retry" {
    const allocator = std.testing.allocator;
    var metrics = try StreamingMetrics.init(allocator, .{});
    defer metrics.deinit();

    // Start a stream that will fail
    metrics.recordStreamStart(.ollama);
    metrics.recordRetryAttempt(.ollama);
    metrics.recordRetryAttempt(.ollama);
    metrics.recordStreamFailure(.ollama);

    const snap = metrics.snapshot();
    try std.testing.expectEqual(@as(u64, 1), snap.backend_failures[@intFromEnum(BackendType.ollama)]);
    try std.testing.expectEqual(@as(u64, 2), snap.backend_retries[@intFromEnum(BackendType.ollama)]);
    try std.testing.expectEqual(@as(u64, 1), snap.total_errors);
}

test "StreamingMetrics cache metrics" {
    const allocator = std.testing.allocator;
    var metrics = try StreamingMetrics.init(allocator, .{});
    defer metrics.deinit();

    metrics.recordSessionCacheHit();
    metrics.recordSessionCacheHit();
    metrics.recordSessionCacheMiss();
    metrics.setSessionCacheEntries(10);

    const snap = metrics.snapshot();
    try std.testing.expectEqual(@as(u64, 2), snap.cache_hits);
    try std.testing.expectEqual(@as(u64, 1), snap.cache_misses);
    try std.testing.expectEqual(@as(i64, 10), snap.cache_entries);

    // Hit rate should be 2/3 = 0.666...
    const hit_rate = snap.cacheHitRate();
    try std.testing.expect(hit_rate > 0.66 and hit_rate < 0.67);
}

test "StreamingMetrics recovery metrics" {
    const allocator = std.testing.allocator;
    var metrics = try StreamingMetrics.init(allocator, .{});
    defer metrics.deinit();

    metrics.recordRecoveryAttempt();
    metrics.recordRecoverySuccess();
    metrics.recordRecoveryAttempt();
    metrics.recordRecoveryFailure();
    metrics.recordRecoveryAttempt();
    metrics.recordPartialRecovery();

    const snap = metrics.snapshot();
    try std.testing.expectEqual(@as(u64, 3), snap.recovery_attempts);
    try std.testing.expectEqual(@as(u64, 1), snap.recovery_success);
    try std.testing.expectEqual(@as(u64, 1), snap.recovery_failed);
    try std.testing.expectEqual(@as(u64, 1), snap.partial_recoveries);

    // Success rate should be 1/3 = 0.333...
    const success_rate = snap.recoverySuccessRate();
    try std.testing.expect(success_rate > 0.33 and success_rate < 0.34);
}

test "StreamingMetrics circuit breaker metrics" {
    const allocator = std.testing.allocator;
    var metrics = try StreamingMetrics.init(allocator, .{});
    defer metrics.deinit();

    metrics.recordCircuitBreakerOpen(.local);
    metrics.recordCircuitBreakerOpen(.local);
    metrics.recordCircuitBreakerClose(.local);

    const snap = metrics.snapshot();
    try std.testing.expectEqual(@as(u64, 2), snap.backend_circuit_opens[@intFromEnum(BackendType.local)]);
    try std.testing.expectEqual(@as(u64, 2), snap.circuit_opens);
    try std.testing.expectEqual(@as(u64, 1), snap.circuit_closes);
}

test "StreamingMetrics reset" {
    const allocator = std.testing.allocator;
    var metrics = try StreamingMetrics.init(allocator, .{});
    defer metrics.deinit();

    // Record some metrics
    metrics.recordStreamStart(.anthropic);
    metrics.recordTokenLatency(.anthropic, 100);
    metrics.recordSessionCacheHit();

    // Verify they're recorded
    var snap = metrics.snapshot();
    try std.testing.expect(snap.total_streams > 0);

    // Reset
    metrics.reset();

    // Verify all zeros
    snap = metrics.snapshot();
    try std.testing.expectEqual(@as(u64, 0), snap.total_streams);
    try std.testing.expectEqual(@as(u64, 0), snap.cache_hits);
}

test "StreamingMetrics disabled config" {
    const allocator = std.testing.allocator;
    var metrics = try StreamingMetrics.init(allocator, .{
        .enable_backend_metrics = false,
        .enable_cache_metrics = false,
        .enable_recovery_metrics = false,
    });
    defer metrics.deinit();

    // These should be no-ops
    metrics.recordStreamStart(.openai);
    metrics.recordSessionCacheHit();
    metrics.recordRecoveryAttempt();

    // All should still be zero
    const snap = metrics.snapshot();
    try std.testing.expectEqual(@as(u64, 0), snap.total_streams);
    try std.testing.expectEqual(@as(u64, 0), snap.cache_hits);
    try std.testing.expectEqual(@as(u64, 0), snap.recovery_attempts);
}

test "Snapshot backendSuccessRate" {
    const allocator = std.testing.allocator;
    var metrics = try StreamingMetrics.init(allocator, .{});
    defer metrics.deinit();

    // 3 requests, 2 successes for openai
    metrics.recordStreamStart(.openai);
    metrics.recordStreamComplete(.openai, 100);
    metrics.recordStreamStart(.openai);
    metrics.recordStreamComplete(.openai, 200);
    metrics.recordStreamStart(.openai);
    metrics.recordStreamFailure(.openai);

    const snap = metrics.snapshot();
    const rate = snap.backendSuccessRate(.openai);
    // 2/3 = 0.666...
    try std.testing.expect(rate > 0.66 and rate < 0.67);

    // Unused backend should have 0 rate
    try std.testing.expectEqual(@as(f64, 0.0), snap.backendSuccessRate(.local));
}
