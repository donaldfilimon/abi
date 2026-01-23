//! Fallback Handling Module
//!
//! Provides automatic fallback mechanisms when models fail, including
//! health tracking, circuit breakers, and failover strategies.

const std = @import("std");
// Shared utilities for millisecond timestamps
const utils = @import("../../shared/utils.zig");

// ============================================================================
// Types
// ============================================================================

/// Health status of a model.
pub const HealthStatus = enum {
    /// Model is functioning normally.
    healthy,
    /// Model is experiencing intermittent issues.
    degraded,
    /// Model is not responding.
    unhealthy,
    /// Circuit breaker is open (not accepting requests).
    circuit_open,
    /// Model is being probed for recovery.
    recovering,

    pub fn toString(self: HealthStatus) []const u8 {
        return @tagName(self);
    }

    pub fn isAvailable(self: HealthStatus) bool {
        return self == .healthy or self == .degraded;
    }
};

/// Policy for handling fallbacks.
pub const FallbackPolicy = enum {
    /// Fail immediately on first error.
    fail_fast,
    /// Retry same model before falling back.
    retry_then_fallback,
    /// Immediately try next model on failure.
    immediate_fallback,
    /// Use circuit breaker pattern.
    circuit_breaker,

    pub fn toString(self: FallbackPolicy) []const u8 {
        return @tagName(self);
    }
};

/// Configuration for fallback behavior.
pub const FallbackConfig = struct {
    /// Maximum retries before falling back.
    max_retries: u32 = 3,
    /// Base delay between retries in milliseconds.
    retry_delay_ms: u64 = 1000,
    /// Multiplier for exponential backoff.
    backoff_multiplier: f32 = 2.0,
    /// Maximum delay between retries.
    max_retry_delay_ms: u64 = 30000,
    /// Health check interval in milliseconds.
    health_check_interval_ms: u64 = 60000,
    /// Number of failures before circuit opens.
    circuit_breaker_threshold: u32 = 5,
    /// Time circuit stays open before half-open probe.
    circuit_open_duration_ms: u64 = 30000,
    /// Number of successes needed to close circuit.
    circuit_close_threshold: u32 = 3,
    /// Enable jitter for retry delays.
    enable_jitter: bool = true,
    /// Fallback policy.
    policy: FallbackPolicy = .retry_then_fallback,
};

// ============================================================================
// Circuit Breaker
// ============================================================================

/// Circuit breaker state machine for a single model.
pub const CircuitBreaker = struct {
    state: State = .closed,
    failure_count: u32 = 0,
    success_count: u32 = 0,
    last_failure_time: i64 = 0,
    last_state_change: i64 = 0,
    config: FallbackConfig,

    pub const State = enum {
        /// Normal operation, allowing requests.
        closed,
        /// Not allowing requests, waiting for timeout.
        open,
        /// Allowing limited requests to probe health.
        half_open,
    };

    pub fn init(config: FallbackConfig) CircuitBreaker {
        return .{ .config = config };
    }

    /// Record a successful request.
    pub fn recordSuccess(self: *CircuitBreaker) void {
        switch (self.state) {
            .closed => {
                self.failure_count = 0;
            },
            .half_open => {
                self.success_count += 1;
                if (self.success_count >= self.config.circuit_close_threshold) {
                    self.transitionTo(.closed);
                }
            },
            .open => {},
        }
    }

    /// Record a failed request.
    pub fn recordFailure(self: *CircuitBreaker) void {
        self.failure_count += 1;
        self.last_failure_time = utils.unixMs();

        switch (self.state) {
            .closed => {
                if (self.failure_count >= self.config.circuit_breaker_threshold) {
                    self.transitionTo(.open);
                }
            },
            .half_open => {
                // Any failure in half-open immediately reopens
                self.transitionTo(.open);
            },
            .open => {},
        }
    }

    /// Check if the circuit allows requests.
    pub fn allowRequest(self: *CircuitBreaker) bool {
        switch (self.state) {
            .closed => return true,
            .half_open => return true, // Allow probe requests
            .open => {
                // Check if enough time has passed to try half-open
                const now = utils.unixMs();
                const elapsed = now - self.last_state_change;
                if (elapsed >= @as(i64, @intCast(self.config.circuit_open_duration_ms))) {
                    self.transitionTo(.half_open);
                    return true;
                }
                return false;
            },
        }
    }

    /// Get current health status based on circuit state.
    pub fn getHealthStatus(self: *CircuitBreaker) HealthStatus {
        return switch (self.state) {
            .closed => if (self.failure_count > 0) .degraded else .healthy,
            .half_open => .recovering,
            .open => .circuit_open,
        };
    }

    fn transitionTo(self: *CircuitBreaker, new_state: State) void {
        self.state = new_state;
        self.last_state_change = utils.unixMs();

        switch (new_state) {
            .closed => {
                self.failure_count = 0;
                self.success_count = 0;
            },
            .half_open => {
                self.success_count = 0;
            },
            .open => {
                self.success_count = 0;
            },
        }
    }

    /// Reset the circuit breaker to initial state.
    pub fn reset(self: *CircuitBreaker) void {
        self.state = .closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.last_failure_time = 0;
        self.last_state_change = 0;
    }
};

// ============================================================================
// Fallback Manager
// ============================================================================

/// Manages fallback logic across multiple models.
pub const FallbackManager = struct {
    allocator: std.mem.Allocator,
    config: FallbackConfig,
    circuit_breakers: std.StringHashMapUnmanaged(CircuitBreaker),
    fallback_chain: std.ArrayListUnmanaged([]const u8),

    pub fn init(allocator: std.mem.Allocator, config: FallbackConfig) FallbackManager {
        return .{
            .allocator = allocator,
            .config = config,
            .circuit_breakers = .{},
            .fallback_chain = .{},
        };
    }

    pub fn deinit(self: *FallbackManager) void {
        // Free allocated model IDs in fallback chain
        for (self.fallback_chain.items) |id| {
            self.allocator.free(id);
        }
        self.fallback_chain.deinit(self.allocator);

        // Free circuit breaker keys
        var it = self.circuit_breakers.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.circuit_breakers.deinit(self.allocator);
    }

    /// Register a model for fallback management.
    pub fn registerModel(self: *FallbackManager, model_id: []const u8) !void {
        const id_copy = try self.allocator.dupe(u8, model_id);
        errdefer self.allocator.free(id_copy);

        try self.circuit_breakers.put(
            self.allocator,
            id_copy,
            CircuitBreaker.init(self.config),
        );
    }

    /// Unregister a model from fallback management.
    pub fn unregisterModel(self: *FallbackManager, model_id: []const u8) void {
        if (self.circuit_breakers.fetchRemove(model_id)) |kv| {
            self.allocator.free(kv.key);
        }
    }

    /// Set the fallback chain (ordered list of model IDs).
    pub fn setFallbackChain(self: *FallbackManager, model_ids: []const []const u8) !void {
        // Clear existing chain
        for (self.fallback_chain.items) |id| {
            self.allocator.free(id);
        }
        self.fallback_chain.shrinkAndFree(self.allocator, 0);

        // Set new chain
        for (model_ids) |id| {
            const id_copy = try self.allocator.dupe(u8, id);
            errdefer self.allocator.free(id_copy);
            try self.fallback_chain.append(self.allocator, id_copy);
        }
    }

    /// Get the next available model in the fallback chain.
    pub fn getNextAvailable(self: *FallbackManager, after_model_id: ?[]const u8) ?[]const u8 {
        var found_current = after_model_id == null;

        for (self.fallback_chain.items) |model_id| {
            if (!found_current) {
                if (after_model_id) |current| {
                    if (std.mem.eql(u8, model_id, current)) {
                        found_current = true;
                    }
                }
                continue;
            }

            // Check if this model is available
            if (self.circuit_breakers.getPtr(model_id)) |cb| {
                if (cb.allowRequest()) {
                    return model_id;
                }
            } else {
                // No circuit breaker = available
                return model_id;
            }
        }

        return null;
    }

    /// Record a successful request for a model.
    pub fn recordSuccess(self: *FallbackManager, model_id: []const u8) void {
        if (self.circuit_breakers.getPtr(model_id)) |cb| {
            cb.recordSuccess();
        }
    }

    /// Record a failed request for a model.
    pub fn recordFailure(self: *FallbackManager, model_id: []const u8) void {
        if (self.circuit_breakers.getPtr(model_id)) |cb| {
            cb.recordFailure();
        }
    }

    /// Check if a model is available (circuit allows requests).
    pub fn isAvailable(self: *FallbackManager, model_id: []const u8) bool {
        if (self.circuit_breakers.getPtr(model_id)) |cb| {
            return cb.allowRequest();
        }
        return true; // Unknown model is considered available
    }

    /// Get health status of a model.
    pub fn getHealth(self: *FallbackManager, model_id: []const u8) HealthStatus {
        if (self.circuit_breakers.getPtr(model_id)) |cb| {
            return cb.getHealthStatus();
        }
        return .healthy;
    }

    /// Get all available models from the fallback chain.
    pub fn getAvailableModels(self: *FallbackManager, allocator: std.mem.Allocator) ![][]const u8 {
        var available = std.ArrayListUnmanaged([]const u8){};
        errdefer available.deinit(allocator);

        for (self.fallback_chain.items) |model_id| {
            if (self.isAvailable(model_id)) {
                try available.append(allocator, model_id);
            }
        }

        return available.toOwnedSlice(allocator);
    }

    /// Reset circuit breaker for a specific model.
    pub fn resetCircuitBreaker(self: *FallbackManager, model_id: []const u8) void {
        if (self.circuit_breakers.getPtr(model_id)) |cb| {
            cb.reset();
        }
    }

    /// Reset all circuit breakers.
    pub fn resetAllCircuitBreakers(self: *FallbackManager) void {
        var it = self.circuit_breakers.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.reset();
        }
    }

    /// Calculate retry delay with exponential backoff and optional jitter.
    pub fn calculateRetryDelay(self: *FallbackManager, attempt: u32) u64 {
        const base_delay = self.config.retry_delay_ms;
        const multiplier = self.config.backoff_multiplier;

        // Calculate exponential delay
        var delay: f64 = @floatFromInt(base_delay);
        for (0..attempt) |_| {
            delay *= multiplier;
        }

        var final_delay: u64 = @intFromFloat(@min(delay, @as(f64, @floatFromInt(self.config.max_retry_delay_ms))));

        // Add jitter if enabled
        if (self.config.enable_jitter and final_delay > 0) {
            // Simple deterministic "jitter" - in real implementation would use random
            const jitter_range = final_delay / 4;
            final_delay = final_delay - jitter_range / 2;
        }

        return final_delay;
    }
};

// ============================================================================
// Fallback Execution Context
// ============================================================================

/// Context for tracking a fallback execution attempt.
pub const FallbackContext = struct {
    /// Current model being tried.
    current_model: ?[]const u8 = null,
    /// Number of retry attempts on current model.
    retry_count: u32 = 0,
    /// Number of fallback attempts (different models).
    fallback_count: u32 = 0,
    /// Models that have been tried.
    tried_models: std.ArrayListUnmanaged([]const u8),
    /// Start time of the execution.
    start_time: i64,
    /// Total allowed timeout.
    timeout_ms: u64,
    /// Allocator for the context.
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, timeout_ms: u64) FallbackContext {
        return .{
            .tried_models = .{},
            .start_time = std.time.milliTimestamp(),
            .timeout_ms = timeout_ms,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FallbackContext) void {
        self.tried_models.deinit(self.allocator);
    }

    /// Check if the execution has timed out.
    pub fn isTimedOut(self: *FallbackContext) bool {
        const elapsed = std.time.milliTimestamp() - self.start_time;
        return elapsed >= @as(i64, @intCast(self.timeout_ms));
    }

    /// Record that a model was tried.
    pub fn recordTried(self: *FallbackContext, model_id: []const u8) !void {
        try self.tried_models.append(self.allocator, model_id);
    }

    /// Check if a model has already been tried.
    pub fn wasTried(self: *FallbackContext, model_id: []const u8) bool {
        for (self.tried_models.items) |tried| {
            if (std.mem.eql(u8, tried, model_id)) {
                return true;
            }
        }
        return false;
    }

    /// Get elapsed time in milliseconds.
    pub fn elapsedMs(self: *FallbackContext) u64 {
        const elapsed = std.time.milliTimestamp() - self.start_time;
        return @intCast(@max(elapsed, 0));
    }
};

// ============================================================================
// Health Check
// ============================================================================

/// Result of a health check probe.
pub const HealthCheckResult = struct {
    model_id: []const u8,
    status: HealthStatus,
    latency_ms: u64,
    error_message: ?[]const u8 = null,
    timestamp: i64,
};

/// Configuration for health checking.
pub const HealthCheckConfig = struct {
    /// Interval between health checks.
    interval_ms: u64 = 60000,
    /// Timeout for individual health check.
    timeout_ms: u64 = 5000,
    /// Number of consecutive failures to mark unhealthy.
    failure_threshold: u32 = 3,
    /// Probe prompt for health check.
    probe_prompt: []const u8 = "ping",
    /// Expected response pattern (optional).
    expected_pattern: ?[]const u8 = null,
};

// ============================================================================
// Tests
// ============================================================================

test "circuit breaker state transitions" {
    var cb = CircuitBreaker.init(.{
        .circuit_breaker_threshold = 3,
        .circuit_close_threshold = 2,
        .circuit_open_duration_ms = 100,
    });

    // Initial state is closed
    try std.testing.expectEqual(CircuitBreaker.State.closed, cb.state);
    try std.testing.expect(cb.allowRequest());

    // Record failures to open circuit
    cb.recordFailure();
    cb.recordFailure();
    try std.testing.expectEqual(CircuitBreaker.State.closed, cb.state);

    cb.recordFailure(); // Third failure opens circuit
    try std.testing.expectEqual(CircuitBreaker.State.open, cb.state);
    try std.testing.expect(!cb.allowRequest());

    // Wait for circuit to half-open
    std.time.sleep(150 * std.time.ns_per_ms);
    try std.testing.expect(cb.allowRequest()); // Transitions to half-open
    try std.testing.expectEqual(CircuitBreaker.State.half_open, cb.state);

    // Success in half-open
    cb.recordSuccess();
    cb.recordSuccess();
    try std.testing.expectEqual(CircuitBreaker.State.closed, cb.state);
}

test "fallback manager" {
    var fm = FallbackManager.init(std.testing.allocator, .{});
    defer fm.deinit();

    try fm.registerModel("model-a");
    try fm.registerModel("model-b");
    try fm.registerModel("model-c");

    try fm.setFallbackChain(&.{ "model-a", "model-b", "model-c" });

    // All models should be available initially
    try std.testing.expect(fm.isAvailable("model-a"));
    try std.testing.expect(fm.isAvailable("model-b"));

    // First available should be model-a
    const first = fm.getNextAvailable(null);
    try std.testing.expect(first != null);
    try std.testing.expectEqualStrings("model-a", first.?);

    // Next after model-a should be model-b
    const second = fm.getNextAvailable("model-a");
    try std.testing.expect(second != null);
    try std.testing.expectEqualStrings("model-b", second.?);
}

test "retry delay calculation" {
    var fm = FallbackManager.init(std.testing.allocator, .{
        .retry_delay_ms = 1000,
        .backoff_multiplier = 2.0,
        .max_retry_delay_ms = 10000,
        .enable_jitter = false,
    });
    defer fm.deinit();

    // First retry: 1000ms
    try std.testing.expectEqual(@as(u64, 1000), fm.calculateRetryDelay(0));

    // Second retry: 2000ms
    try std.testing.expectEqual(@as(u64, 2000), fm.calculateRetryDelay(1));

    // Third retry: 4000ms
    try std.testing.expectEqual(@as(u64, 4000), fm.calculateRetryDelay(2));

    // Fourth retry: 8000ms
    try std.testing.expectEqual(@as(u64, 8000), fm.calculateRetryDelay(3));

    // Fifth retry: capped at 10000ms
    try std.testing.expectEqual(@as(u64, 10000), fm.calculateRetryDelay(4));
}

test "health status availability" {
    try std.testing.expect(HealthStatus.healthy.isAvailable());
    try std.testing.expect(HealthStatus.degraded.isAvailable());
    try std.testing.expect(!HealthStatus.unhealthy.isAvailable());
    try std.testing.expect(!HealthStatus.circuit_open.isAvailable());
    try std.testing.expect(!HealthStatus.recovering.isAvailable());
}
