//! Backpressure control for streaming.
//!
//! Provides flow control mechanisms to prevent overwhelming
//! downstream consumers during high-throughput streaming.

const std = @import("std");
const time = @import("../../shared/time.zig");

/// Backpressure strategy.
pub const BackpressureStrategy = enum {
    /// Drop tokens when backpressure threshold reached.
    drop,
    /// Block until consumer catches up.
    block,
    /// Buffer tokens (with limit).
    buffer,
    /// Sample tokens (keep every Nth).
    sample,
    /// Adaptive rate limiting.
    adaptive,
};

/// Backpressure configuration.
pub const BackpressureConfig = struct {
    /// Backpressure strategy.
    strategy: BackpressureStrategy = .buffer,
    /// High watermark (start applying backpressure).
    high_watermark: usize = 100,
    /// Low watermark (stop applying backpressure).
    low_watermark: usize = 25,
    /// Maximum buffer size.
    max_buffer: usize = 1000,
    /// Sample rate (for sample strategy).
    sample_rate: usize = 2,
    /// Target tokens per second (for adaptive).
    target_tps: f64 = 50.0,
    /// Measurement window (ns).
    window_ns: u64 = 100_000_000, // 100ms
};

/// Flow state.
pub const FlowState = enum {
    /// Normal flow, no backpressure.
    normal,
    /// Backpressure active, flow reduced.
    throttled,
    /// Flow completely blocked.
    blocked,
    /// Recovering from backpressure.
    recovering,
};

/// Backpressure controller.
pub const BackpressureController = struct {
    config: BackpressureConfig,
    state: FlowState,
    pending_count: usize,
    dropped_count: usize,
    total_processed: u64,
    sample_counter: usize,
    timer: std.time.Timer,
    last_check_ns: u64,
    tokens_in_window: usize,
    current_tps: f64,

    /// Initialize backpressure controller.
    /// Returns error.TimerUnavailable if platform timer cannot be started.
    pub fn init(config: BackpressureConfig) error{TimerUnavailable}!BackpressureController {
        return .{
            .config = config,
            .state = .normal,
            .pending_count = 0,
            .dropped_count = 0,
            .total_processed = 0,
            .sample_counter = 0,
            .timer = std.time.Timer.start() catch return error.TimerUnavailable,
            .last_check_ns = 0,
            .tokens_in_window = 0,
            .current_tps = 0,
        };
    }

    /// Check flow state and determine if token should be processed.
    pub fn checkFlow(self: *BackpressureController) FlowState {
        return switch (self.config.strategy) {
            .drop => self.checkDropStrategy(),
            .block => self.checkBlockStrategy(),
            .buffer => self.checkBufferStrategy(),
            .sample => self.checkSampleStrategy(),
            .adaptive => self.checkAdaptiveStrategy(),
        };
    }

    /// Signal that a token was produced.
    pub fn produce(self: *BackpressureController) void {
        self.pending_count += 1;
        self.tokens_in_window += 1;
        self.updateState();
    }

    /// Signal that a token was consumed.
    pub fn consume(self: *BackpressureController) void {
        if (self.pending_count > 0) {
            self.pending_count -= 1;
        }
        self.total_processed += 1;
        self.updateState();
    }

    /// Signal that a token was dropped.
    pub fn drop(self: *BackpressureController) void {
        self.dropped_count += 1;
    }

    /// Get current flow state.
    pub fn getState(self: *const BackpressureController) FlowState {
        return self.state;
    }

    /// Get backpressure statistics.
    pub fn getStats(self: *const BackpressureController) BackpressureStats {
        return .{
            .pending_count = self.pending_count,
            .dropped_count = self.dropped_count,
            .total_processed = self.total_processed,
            .current_tps = self.current_tps,
            .state = self.state,
            .utilization = if (self.config.max_buffer > 0)
                @as(f64, @floatFromInt(self.pending_count)) /
                    @as(f64, @floatFromInt(self.config.max_buffer))
            else
                0,
        };
    }

    /// Reset the controller.
    pub fn reset(self: *BackpressureController) void {
        self.state = .normal;
        self.pending_count = 0;
        self.dropped_count = 0;
        self.sample_counter = 0;
        self.tokens_in_window = 0;
    }

    fn checkDropStrategy(self: *BackpressureController) FlowState {
        if (self.pending_count >= self.config.high_watermark) {
            return .blocked;
        }
        return .normal;
    }

    fn checkBlockStrategy(self: *BackpressureController) FlowState {
        if (self.pending_count >= self.config.max_buffer) {
            return .blocked;
        } else if (self.pending_count >= self.config.high_watermark) {
            return .throttled;
        }
        return .normal;
    }

    fn checkBufferStrategy(self: *BackpressureController) FlowState {
        if (self.pending_count >= self.config.max_buffer) {
            return .blocked;
        } else if (self.pending_count >= self.config.high_watermark) {
            return .throttled;
        } else if (self.state == .throttled and self.pending_count > self.config.low_watermark) {
            return .recovering;
        }
        return .normal;
    }

    fn checkSampleStrategy(self: *BackpressureController) FlowState {
        if (self.pending_count >= self.config.high_watermark) {
            self.sample_counter += 1;
            if (self.sample_counter >= self.config.sample_rate) {
                self.sample_counter = 0;
                return .throttled; // Allow this one
            }
            return .blocked; // Skip this one
        }
        return .normal;
    }

    fn checkAdaptiveStrategy(self: *BackpressureController) FlowState {
        const now = self.timer.read();

        // Update TPS measurement
        if (now - self.last_check_ns >= self.config.window_ns) {
            const window_secs = @as(f64, @floatFromInt(self.config.window_ns)) / 1_000_000_000.0;
            self.current_tps = @as(f64, @floatFromInt(self.tokens_in_window)) / window_secs;
            self.tokens_in_window = 0;
            self.last_check_ns = now;
        }

        // Adjust based on TPS vs target
        if (self.current_tps > self.config.target_tps * 1.2) {
            return .throttled;
        } else if (self.current_tps > self.config.target_tps * 1.5) {
            return .blocked;
        }
        return .normal;
    }

    fn updateState(self: *BackpressureController) void {
        if (self.pending_count >= self.config.max_buffer) {
            self.state = .blocked;
        } else if (self.pending_count >= self.config.high_watermark) {
            self.state = .throttled;
        } else if (self.pending_count <= self.config.low_watermark) {
            self.state = .normal;
        } else if (self.state == .throttled or self.state == .blocked) {
            self.state = .recovering;
        }
    }
};

/// Backpressure statistics.
pub const BackpressureStats = struct {
    pending_count: usize,
    dropped_count: usize,
    total_processed: u64,
    current_tps: f64,
    state: FlowState,
    utilization: f64,
};

/// Rate limiter for token emission.
pub const RateLimiter = struct {
    tokens_per_second: f64,
    bucket_size: f64,
    available_tokens: f64,
    timer: std.time.Timer,
    last_refill_ns: u64,

    /// Initialize rate limiter.
    /// Returns error.TimerUnavailable if platform timer cannot be started.
    pub fn init(tokens_per_second: f64, bucket_size: ?f64) error{TimerUnavailable}!RateLimiter {
        return .{
            .tokens_per_second = tokens_per_second,
            .bucket_size = bucket_size orelse tokens_per_second,
            .available_tokens = bucket_size orelse tokens_per_second,
            .timer = std.time.Timer.start() catch return error.TimerUnavailable,
            .last_refill_ns = 0,
        };
    }

    /// Try to acquire a token.
    pub fn tryAcquire(self: *RateLimiter) bool {
        self.refill();

        if (self.available_tokens >= 1.0) {
            self.available_tokens -= 1.0;
            return true;
        }
        return false;
    }

    /// Acquire a token, blocking if necessary.
    pub fn acquire(self: *RateLimiter) void {
        while (!self.tryAcquire()) {
            // Calculate wait time
            const wait_ns = @as(u64, @intFromFloat(1_000_000_000.0 / self.tokens_per_second));
            time.sleepNs(wait_ns);
        }
    }

    /// Refill tokens based on elapsed time.
    fn refill(self: *RateLimiter) void {
        const now = self.timer.read();
        const elapsed_ns = now - self.last_refill_ns;

        if (elapsed_ns > 0) {
            const elapsed_secs = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
            const new_tokens = elapsed_secs * self.tokens_per_second;

            self.available_tokens = @min(self.available_tokens + new_tokens, self.bucket_size);
            self.last_refill_ns = now;
        }
    }

    /// Get available tokens.
    pub fn getAvailable(self: *RateLimiter) f64 {
        self.refill();
        return self.available_tokens;
    }

    /// Reset the rate limiter.
    pub fn reset(self: *RateLimiter) void {
        self.available_tokens = self.bucket_size;
        self.timer.reset();
        self.last_refill_ns = 0;
    }
};

test "backpressure controller basic" {
    var ctrl = BackpressureController.init(.{
        .strategy = .buffer,
        .high_watermark = 10,
        .low_watermark = 5,
        .max_buffer = 20,
    }) catch return error.SkipZigTest;

    try std.testing.expectEqual(FlowState.normal, ctrl.getState());

    // Fill up to high watermark
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        ctrl.produce();
    }

    try std.testing.expectEqual(FlowState.throttled, ctrl.getState());

    // Consume to low watermark
    while (i > 5) : (i -= 1) {
        ctrl.consume();
    }

    try std.testing.expectEqual(FlowState.normal, ctrl.getState());
}

test "backpressure controller blocking" {
    var ctrl = BackpressureController.init(.{
        .strategy = .buffer,
        .high_watermark = 5,
        .max_buffer = 10,
    }) catch return error.SkipZigTest;

    // Fill to max
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        ctrl.produce();
    }

    try std.testing.expectEqual(FlowState.blocked, ctrl.checkFlow());
}

test "backpressure statistics" {
    var ctrl = BackpressureController.init(.{}) catch return error.SkipZigTest;

    ctrl.produce();
    ctrl.produce();
    ctrl.consume();
    ctrl.drop();

    const stats = ctrl.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.pending_count);
    try std.testing.expectEqual(@as(usize, 1), stats.dropped_count);
    try std.testing.expectEqual(@as(u64, 1), stats.total_processed);
}

test "rate limiter" {
    var limiter = RateLimiter.init(100.0, 10.0) catch return error.SkipZigTest;

    // Should have 10 tokens available initially
    try std.testing.expect(limiter.getAvailable() >= 9.9);

    // Acquire some tokens
    try std.testing.expect(limiter.tryAcquire());
    try std.testing.expect(limiter.tryAcquire());

    try std.testing.expect(limiter.getAvailable() < 10.0);
}
