//! Streaming Recovery Integration Tests
//!
//! Tests for stream error recovery including:
//! - Circuit breaker behavior under failures
//! - Retry logic with exponential backoff
//! - Session cache for reconnection
//! - Recovery event propagation
//! - Fault injection scenarios
//!
//! ## Running
//!
//! These tests are included in the main integration test suite.
//! Run with: `zig build test --summary all`

const std = @import("std");
const testing = std.testing;
const abi = @import("abi");
const time = abi.services.shared.time;
const sync = abi.services.shared.sync;
const streaming = abi.features.ai.streaming;

const CircuitBreaker = streaming.CircuitBreaker;
const CircuitBreakerConfig = streaming.CircuitBreakerConfig;
const CircuitState = streaming.CircuitState;
const StreamRecovery = streaming.StreamRecovery;
const RecoveryConfig = streaming.RecoveryConfig;
const SessionCache = streaming.SessionCache;
const SessionCacheConfig = streaming.SessionCacheConfig;
const StreamingMetrics = streaming.StreamingMetrics;
const BackendType = streaming.BackendType;

// ============================================================================
// Circuit Breaker Integration Tests
// ============================================================================

test "circuit breaker: state transitions under failures" {
    var cb = CircuitBreaker.init(.{
        .failure_threshold = 3,
        .success_threshold = 2,
        .timeout_ms = 10, // Short timeout for testing
        .half_open_max_requests = 2,
    });

    // Initial state: closed
    try testing.expectEqual(CircuitState.closed, cb.getState());
    try testing.expect(cb.canAttempt());

    // Fail 3 times to open circuit
    for (0..3) |_| {
        try testing.expect(cb.canAttempt());
        cb.recordFailure();
    }

    // Circuit should now be open
    try testing.expectEqual(CircuitState.open, cb.getState());
    try testing.expect(!cb.canAttempt());

    // Wait for timeout (busy-wait since tests don't have I/O backend)
    {
        var timer = time.Timer.start() catch return;
        while (timer.read() < 15 * std.time.ns_per_ms) {
            std.atomic.spinLoopHint();
        }
    }

    // Should transition to half-open on next attempt
    try testing.expect(cb.canAttempt());
    try testing.expectEqual(CircuitState.half_open, cb.getState());

    // Two successes should close the circuit
    cb.recordSuccess();
    cb.recordSuccess();
    try testing.expectEqual(CircuitState.closed, cb.getState());
}

test "circuit breaker: half-open failure returns to open" {
    var cb = CircuitBreaker.init(.{
        .failure_threshold = 1,
        .timeout_ms = 10,
    });

    // Trip to open
    _ = cb.canAttempt();
    cb.recordFailure();
    try testing.expectEqual(CircuitState.open, cb.getState());

    // Wait and transition to half-open (busy-wait since tests don't have I/O backend)
    {
        var timer = time.Timer.start() catch return;
        while (timer.read() < 15 * std.time.ns_per_ms) {
            std.atomic.spinLoopHint();
        }
    }
    try testing.expect(cb.canAttempt());
    try testing.expectEqual(CircuitState.half_open, cb.getState());

    // Failure in half-open goes back to open
    cb.recordFailure();
    try testing.expectEqual(CircuitState.open, cb.getState());
}

test "circuit breaker: stats tracking" {
    var cb = CircuitBreaker.init(.{ .failure_threshold = 2 });

    // 1 success
    _ = cb.canAttempt();
    cb.recordSuccess();

    // 2 failures to trip
    _ = cb.canAttempt();
    cb.recordFailure();
    _ = cb.canAttempt();
    cb.recordFailure();

    // 1 rejected (circuit open)
    _ = cb.canAttempt();

    const stats = cb.getStats();
    try testing.expectEqual(@as(u64, 3), stats.total_requests);
    try testing.expectEqual(@as(u64, 1), stats.successful_requests);
    try testing.expectEqual(@as(u64, 2), stats.failed_requests);
    try testing.expectEqual(@as(u64, 1), stats.rejected_requests);
    try testing.expectEqual(@as(u64, 1), stats.times_opened);
}

// ============================================================================
// Session Cache Integration Tests
// ============================================================================

test "session cache: store and retrieve tokens" {
    const allocator = testing.allocator;
    var cache = SessionCache.init(allocator, .{
        .max_sessions = 10,
        .max_tokens_per_session = 5,
        .ttl_ms = 60_000,
    });
    defer cache.deinit();

    // Store tokens
    try cache.storeToken("session-1", 1, "Hello", .local, 12345);
    try cache.storeToken("session-1", 2, " world", .local, 12345);
    try cache.storeToken("session-1", 3, "!", .local, 12345);

    // Retrieve tokens since event 1
    if (cache.getTokensSince("session-1", 1)) |tokens| {
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try testing.expectEqualStrings(" world", tokens[0].text);
        try testing.expectEqualStrings("!", tokens[1].text);
    } else {
        return error.ExpectedTokens;
    }

    // Unknown session returns null
    try testing.expect(cache.getTokensSince("unknown-session", 0) == null);
}

test "session cache: LRU eviction" {
    const allocator = testing.allocator;
    var cache = SessionCache.init(allocator, .{
        .max_sessions = 2,
        .max_tokens_per_session = 10,
        .ttl_ms = 60_000,
    });
    defer cache.deinit();

    // Create 3 sessions (exceeds max of 2)
    try cache.storeToken("session-1", 1, "first", .local, 1);
    try cache.storeToken("session-2", 1, "second", .local, 2);
    try cache.storeToken("session-3", 1, "third", .local, 3);

    // Session 1 should be evicted (LRU)
    try testing.expect(cache.getTokensSince("session-1", 0) == null);

    // Sessions 2 and 3 should still exist
    try testing.expect(cache.getTokensSince("session-2", 0) != null);
    try testing.expect(cache.getTokensSince("session-3", 0) != null);
}

test "session cache: per-session token limit" {
    const allocator = testing.allocator;
    var cache = SessionCache.init(allocator, .{
        .max_sessions = 10,
        .max_tokens_per_session = 3,
        .ttl_ms = 60_000,
    });
    defer cache.deinit();

    // Store 5 tokens (exceeds limit of 3)
    try cache.storeToken("session-1", 1, "one", .local, 1);
    try cache.storeToken("session-1", 2, "two", .local, 1);
    try cache.storeToken("session-1", 3, "three", .local, 1);
    try cache.storeToken("session-1", 4, "four", .local, 1);
    try cache.storeToken("session-1", 5, "five", .local, 1);

    // Only last 3 tokens should be kept
    if (cache.getTokensSince("session-1", 0)) |tokens| {
        try testing.expectEqual(@as(usize, 3), tokens.len);
        try testing.expectEqualStrings("three", tokens[0].text);
        try testing.expectEqualStrings("four", tokens[1].text);
        try testing.expectEqualStrings("five", tokens[2].text);
    } else {
        return error.ExpectedTokens;
    }
}

// ============================================================================
// Stream Recovery Integration Tests
// ============================================================================

test "stream recovery: circuit breaker per backend" {
    const allocator = testing.allocator;
    var recovery = try StreamRecovery.init(allocator, .{
        .circuit_breaker = .{ .failure_threshold = 2 },
    });
    defer recovery.deinit();

    // Initially all backends available
    try testing.expect(recovery.isBackendAvailable(.local));
    try testing.expect(recovery.isBackendAvailable(.openai));

    // Fail openai backend twice
    recovery.recordFailure(.openai);
    recovery.recordFailure(.openai);

    // OpenAI should be unavailable, but local still available
    try testing.expect(!recovery.isBackendAvailable(.openai));
    try testing.expect(recovery.isBackendAvailable(.local));
}

test "stream recovery: retry delay calculation" {
    const allocator = testing.allocator;
    var recovery = try StreamRecovery.init(allocator, .{
        .retry = .{
            .base = .{
                .initial_delay_ns = 100_000_000, // 100ms
                .max_delay_ns = 1_000_000_000, // 1s
                .multiplier = 2.0,
                .jitter = false,
            },
        },
    });
    defer recovery.deinit();

    // Verify exponential backoff
    try testing.expectEqual(@as(u64, 100_000_000), recovery.calculateRetryDelay(0));
    try testing.expectEqual(@as(u64, 200_000_000), recovery.calculateRetryDelay(1));
    try testing.expectEqual(@as(u64, 400_000_000), recovery.calculateRetryDelay(2));
    try testing.expectEqual(@as(u64, 800_000_000), recovery.calculateRetryDelay(3));

    // Should cap at max
    try testing.expectEqual(@as(u64, 1_000_000_000), recovery.calculateRetryDelay(10));
}

test "stream recovery: event callback" {
    const allocator = testing.allocator;

    const Context = struct {
        circuit_opens: usize = 0,
        circuit_closes: usize = 0,

        fn callback(event: streaming.RecoveryEvent, ctx: ?*anyopaque) void {
            const self = @as(*@This(), @ptrCast(@alignCast(ctx)));
            switch (event) {
                .circuit_breaker => |cb| {
                    if (cb.new_state == .open) {
                        self.circuit_opens += 1;
                    } else if (cb.new_state == .closed) {
                        self.circuit_closes += 1;
                    }
                },
                else => {},
            }
        }
    };

    var ctx = Context{};
    var recovery = try StreamRecovery.init(allocator, .{
        .circuit_breaker = .{ .failure_threshold = 1, .success_threshold = 1, .timeout_ms = 10 },
    });
    defer recovery.deinit();

    recovery.setEventCallback(Context.callback, &ctx);

    // Trip circuit
    recovery.recordFailure(.local);
    try testing.expectEqual(@as(usize, 1), ctx.circuit_opens);

    // Wait and recover (busy-wait since tests don't have I/O backend)
    {
        var timer = time.Timer.start() catch return;
        while (timer.read() < 15 * std.time.ns_per_ms) {
            std.atomic.spinLoopHint();
        }
    }
    _ = recovery.isBackendAvailable(.local); // Transition to half-open
    recovery.recordSuccess(.local);

    try testing.expectEqual(@as(usize, 1), ctx.circuit_closes);
}

// ============================================================================
// Metrics Integration Tests
// ============================================================================

test "streaming metrics: comprehensive tracking" {
    const allocator = testing.allocator;
    var metrics = try StreamingMetrics.init(allocator, .{});
    defer metrics.deinit();

    // Simulate a stream lifecycle
    metrics.recordStreamStart(.openai);
    metrics.recordTokenLatency(.openai, 100);
    metrics.recordTokenLatency(.openai, 150);
    metrics.recordTokenLatency(.openai, 200);
    metrics.recordStreamComplete(.openai, 5000);

    // Simulate a failed stream
    metrics.recordStreamStart(.ollama);
    metrics.recordRetryAttempt(.ollama);
    metrics.recordStreamFailure(.ollama);

    // Check snapshot
    const snap = metrics.snapshot();
    try testing.expectEqual(@as(u64, 2), snap.total_streams);
    try testing.expectEqual(@as(u64, 3), snap.total_tokens);
    try testing.expectEqual(@as(u64, 1), snap.total_errors);
    try testing.expectEqual(@as(u64, 1), snap.backend_retries[@intFromEnum(BackendType.ollama)]);

    // Check backend success rate
    const openai_rate = snap.backendSuccessRate(.openai);
    try testing.expectEqual(@as(f64, 1.0), openai_rate);

    const ollama_rate = snap.backendSuccessRate(.ollama);
    try testing.expectEqual(@as(f64, 0.0), ollama_rate);
}

test "streaming metrics: cache metrics" {
    const allocator = testing.allocator;
    var metrics = try StreamingMetrics.init(allocator, .{});
    defer metrics.deinit();

    metrics.recordSessionCacheHit();
    metrics.recordSessionCacheHit();
    metrics.recordSessionCacheMiss();

    const snap = metrics.snapshot();
    try testing.expectEqual(@as(u64, 2), snap.cache_hits);
    try testing.expectEqual(@as(u64, 1), snap.cache_misses);

    const hit_rate = snap.cacheHitRate();
    try testing.expect(hit_rate > 0.66 and hit_rate < 0.67);
}

test "streaming metrics: recovery metrics" {
    const allocator = testing.allocator;
    var metrics = try StreamingMetrics.init(allocator, .{});
    defer metrics.deinit();

    metrics.recordRecoveryAttempt();
    metrics.recordRecoverySuccess();
    metrics.recordRecoveryAttempt();
    metrics.recordRecoveryFailure();

    const snap = metrics.snapshot();
    try testing.expectEqual(@as(u64, 2), snap.recovery_attempts);
    try testing.expectEqual(@as(u64, 1), snap.recovery_success);
    try testing.expectEqual(@as(u64, 1), snap.recovery_failed);

    const success_rate = snap.recoverySuccessRate();
    try testing.expectEqual(@as(f64, 0.5), success_rate);
}

// ============================================================================
// Recovery Event Serialization Tests
// ============================================================================

test "recovery event: JSON serialization" {
    const allocator = testing.allocator;

    // Test retry event
    const retry_event = streaming.RecoveryEvent{
        .retry = .{
            .attempt = 2,
            .max_attempts = 3,
            .delay_ms = 500,
            .reason = "connection timeout",
        },
    };
    const retry_json = try retry_event.toJson(allocator);
    defer allocator.free(retry_json);

    try testing.expect(std.mem.indexOf(u8, retry_json, "\"retry\"") != null);
    try testing.expect(std.mem.indexOf(u8, retry_json, "\"attempt\":2") != null);

    // Test circuit breaker event
    const cb_event = streaming.RecoveryEvent{
        .circuit_breaker = .{
            .backend = .openai,
            .old_state = .closed,
            .new_state = .open,
        },
    };
    const cb_json = try cb_event.toJson(allocator);
    defer allocator.free(cb_json);

    try testing.expect(std.mem.indexOf(u8, cb_json, "\"circuit_breaker\"") != null);
    try testing.expect(std.mem.indexOf(u8, cb_json, "\"openai\"") != null);
}

// ============================================================================
// Fault Injection Scenarios
// ============================================================================

test "fault injection: cascading failure protection" {
    const allocator = testing.allocator;

    // Create recovery with multiple backends
    var recovery = try StreamRecovery.init(allocator, .{
        .circuit_breaker = .{ .failure_threshold = 2 },
    });
    defer recovery.deinit();

    // Simulate cascading failures across backends
    recovery.recordFailure(.openai);
    recovery.recordFailure(.openai);
    recovery.recordFailure(.ollama);
    recovery.recordFailure(.ollama);

    // Both external backends unavailable
    try testing.expect(!recovery.isBackendAvailable(.openai));
    try testing.expect(!recovery.isBackendAvailable(.ollama));

    // Local should still be available (isolated failure)
    try testing.expect(recovery.isBackendAvailable(.local));

    // Metrics should track all circuit breaker opens
    if (recovery.getMetrics()) |m| {
        const snap = m.snapshot();
        try testing.expectEqual(@as(u64, 2), snap.circuit_opens);
    }
}

test "fault injection: recovery under intermittent failures" {
    const allocator = testing.allocator;
    var recovery = try StreamRecovery.init(allocator, .{
        .circuit_breaker = .{
            .failure_threshold = 3,
            .success_threshold = 2,
            .timeout_ms = 10,
        },
    });
    defer recovery.deinit();

    // Pattern: 2 failures, 1 success, 2 failures, 1 success (shouldn't trip)
    recovery.recordFailure(.local);
    recovery.recordFailure(.local);
    recovery.recordSuccess(.local);
    recovery.recordFailure(.local);
    recovery.recordFailure(.local);
    recovery.recordSuccess(.local);

    // Circuit should still be closed (success resets counter)
    try testing.expectEqual(CircuitState.closed, recovery.getCircuitBreaker(.local).getState());

    // Now trip it with 3 consecutive failures
    recovery.recordFailure(.local);
    recovery.recordFailure(.local);
    recovery.recordFailure(.local);

    try testing.expectEqual(CircuitState.open, recovery.getCircuitBreaker(.local).getState());
}

test "fault injection: session recovery simulation" {
    const allocator = testing.allocator;
    var recovery = try StreamRecovery.init(allocator, .{});
    defer recovery.deinit();

    const session_id = "test-session-123";

    // Simulate streaming some tokens
    recovery.cacheToken(session_id, 1, "Hello", .local, 12345);
    recovery.cacheToken(session_id, 2, " world", .local, 12345);
    recovery.cacheToken(session_id, 3, "!", .local, 12345);

    // Simulate client disconnect and reconnect with Last-Event-ID: 1
    if (recovery.recoverTokens(session_id, 1)) |tokens| {
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try testing.expectEqualStrings(" world", tokens[0].text);
        try testing.expectEqualStrings("!", tokens[1].text);

        // Emit recovery event
        recovery.emitSessionRecoveryEvent(session_id, tokens.len, 1);
    } else {
        return error.ExpectedRecovery;
    }

    // Metrics should show cache hit
    if (recovery.getMetrics()) |m| {
        const snap = m.snapshot();
        try testing.expectEqual(@as(u64, 1), snap.cache_hits);
        try testing.expectEqual(@as(u64, 1), snap.recovery_success);
    }
}
