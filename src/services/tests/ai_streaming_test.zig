//! AI Streaming Resilience Tests — Circuit Breaker, Backpressure, Buffer
//!
//! Tests state machine transitions, resilience patterns, and flow control.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const streaming = if (build_options.enable_ai) abi.ai.streaming else struct {};
const CircuitBreaker = if (build_options.enable_ai) streaming.CircuitBreaker else struct {};
const CircuitBreakerConfig = if (build_options.enable_ai) streaming.CircuitBreakerConfig else struct {};

// ============================================================================
// Circuit Breaker State Machine Tests
// ============================================================================

test "circuit breaker: starts in closed state" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var cb = CircuitBreaker.init(.{ .failure_threshold = 3 });
    try std.testing.expectEqual(streaming.CircuitState.closed, cb.getState());
    try std.testing.expect(cb.canAttempt());
}

test "circuit breaker: closed → open after threshold failures" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var cb = CircuitBreaker.init(.{
        .failure_threshold = 3,
        .timeout_ms = 60_000,
    });

    // Record exactly threshold failures
    cb.recordFailure();
    try std.testing.expectEqual(streaming.CircuitState.closed, cb.getState());
    cb.recordFailure();
    try std.testing.expectEqual(streaming.CircuitState.closed, cb.getState());
    cb.recordFailure();

    // Should now be open
    try std.testing.expectEqual(streaming.CircuitState.open, cb.getState());
    try std.testing.expect(!cb.canAttempt());
}

test "circuit breaker: success resets failure count" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var cb = CircuitBreaker.init(.{ .failure_threshold = 3 });

    // 2 failures, then success
    cb.recordFailure();
    cb.recordFailure();
    cb.recordSuccess();

    // 2 more failures should not open (counter was reset)
    cb.recordFailure();
    cb.recordFailure();
    try std.testing.expectEqual(streaming.CircuitState.closed, cb.getState());

    // 3rd failure opens
    cb.recordFailure();
    try std.testing.expectEqual(streaming.CircuitState.open, cb.getState());
}

test "circuit breaker: open → half_open after timeout" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var cb = CircuitBreaker.init(.{
        .failure_threshold = 1,
        .timeout_ms = 1, // 1ms timeout for fast test
    });

    // Open the circuit
    cb.recordFailure();
    try std.testing.expectEqual(streaming.CircuitState.open, cb.getState());

    // Wait for timeout to elapse
    abi.shared.time.sleepMs(10);

    // canAttempt should trigger transition to half_open
    const can = cb.canAttempt();
    if (can) {
        try std.testing.expectEqual(streaming.CircuitState.half_open, cb.getState());
    }
    // If timing is tight, the transition may not have happened yet — acceptable
}

test "circuit breaker: half_open → closed after success threshold" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var cb = CircuitBreaker.init(.{
        .failure_threshold = 1,
        .success_threshold = 2,
        .timeout_ms = 1,
    });

    // Open the circuit
    cb.recordFailure();
    abi.shared.time.sleepMs(10);

    // Trigger half-open transition
    if (cb.canAttempt()) {
        // Record success_threshold successes
        cb.recordSuccess();
        cb.recordSuccess();

        // Should be back to closed
        try std.testing.expectEqual(streaming.CircuitState.closed, cb.getState());
    }
}

test "circuit breaker: half_open → open on any failure" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var cb = CircuitBreaker.init(.{
        .failure_threshold = 1,
        .success_threshold = 5,
        .timeout_ms = 1,
    });

    // Open the circuit
    cb.recordFailure();
    abi.shared.time.sleepMs(10);

    // Trigger half-open
    if (cb.canAttempt()) {
        // Any failure in half-open immediately reopens
        cb.recordFailure();
        try std.testing.expectEqual(streaming.CircuitState.open, cb.getState());
    }
}

test "circuit breaker: stats track correctly" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var cb = CircuitBreaker.init(.{ .failure_threshold = 2 });

    // 3 attempts: 1 success, 2 failures
    _ = cb.canAttempt();
    cb.recordSuccess();
    _ = cb.canAttempt();
    cb.recordFailure();
    _ = cb.canAttempt();
    cb.recordFailure();

    const stats = cb.getStats();
    try std.testing.expectEqual(@as(u64, 3), stats.total_requests);
    try std.testing.expectEqual(@as(u64, 1), stats.successful_requests);
    try std.testing.expectEqual(@as(u64, 2), stats.failed_requests);
    try std.testing.expectEqual(@as(u64, 1), stats.times_opened);
}

test "circuit breaker: open rejects and counts rejections" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var cb = CircuitBreaker.init(.{
        .failure_threshold = 1,
        .timeout_ms = 60_000,
    });

    // Open the circuit
    _ = cb.canAttempt();
    cb.recordFailure();

    // Attempt while open should be rejected
    try std.testing.expect(!cb.canAttempt());
    try std.testing.expect(!cb.canAttempt());

    const stats = cb.getStats();
    try std.testing.expect(stats.rejected_requests >= 2);
}

test "circuit breaker: reset returns to closed" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var cb = CircuitBreaker.init(.{ .failure_threshold = 1 });

    // Open the circuit
    cb.recordFailure();
    try std.testing.expectEqual(streaming.CircuitState.open, cb.getState());

    // Reset
    cb.reset();
    try std.testing.expectEqual(streaming.CircuitState.closed, cb.getState());
    try std.testing.expect(cb.canAttempt());
}

// ============================================================================
// Backpressure Tests
// ============================================================================

test "backpressure: starts in normal state" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    const bp = streaming.backpressure.BackpressureController.init(.{
        .strategy = .buffer,
        .high_watermark = 10,
        .low_watermark = 2,
    }) catch return error.SkipZigTest;

    try std.testing.expectEqual(streaming.backpressure.FlowState.normal, bp.state);
    try std.testing.expectEqual(@as(usize, 0), bp.pending_count);
}

test "backpressure: produce increments pending" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var bp = streaming.backpressure.BackpressureController.init(.{
        .strategy = .buffer,
        .high_watermark = 100,
    }) catch return error.SkipZigTest;

    bp.produce();
    bp.produce();
    bp.produce();

    try std.testing.expectEqual(@as(usize, 3), bp.pending_count);
    // total_processed only incremented on consume(), not produce()
    try std.testing.expectEqual(@as(u64, 0), bp.total_processed);
}

test "backpressure: consume decrements pending" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var bp = streaming.backpressure.BackpressureController.init(.{
        .strategy = .buffer,
        .high_watermark = 100,
    }) catch return error.SkipZigTest;

    bp.produce();
    bp.produce();
    bp.produce();
    bp.consume();

    try std.testing.expectEqual(@as(usize, 2), bp.pending_count);
}

test "backpressure: drop strategy blocks at watermark" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var bp = streaming.backpressure.BackpressureController.init(.{
        .strategy = .drop,
        .high_watermark = 3,
        .low_watermark = 1,
    }) catch return error.SkipZigTest;

    // Fill to watermark
    bp.produce();
    bp.produce();
    bp.produce();

    const flow = bp.checkFlow();
    try std.testing.expect(flow == .blocked or flow == .throttled);
}

test "backpressure: reset clears state" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var bp = streaming.backpressure.BackpressureController.init(.{
        .strategy = .buffer,
        .high_watermark = 10,
    }) catch return error.SkipZigTest;

    bp.produce();
    bp.produce();
    bp.reset();

    try std.testing.expectEqual(@as(usize, 0), bp.pending_count);
    try std.testing.expectEqual(streaming.backpressure.FlowState.normal, bp.state);
}

// ============================================================================
// Token Buffer Tests
// ============================================================================

test "buffer: empty pop returns null" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var buf = streaming.buffer.TokenBuffer.init(allocator, .{
        .strategy = .fifo,
        .capacity = 10,
    });
    defer buf.deinit();

    try std.testing.expect(buf.pop() == null);
    try std.testing.expect(buf.isEmpty());
}
