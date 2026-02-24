//! Stream recovery module.
//!
//! Integrates circuit breakers, retry logic, session caching, and metrics
//! for resilient streaming operations.
//!
//! ## Components
//!
//! - **CircuitBreaker**: Per-backend circuit breakers to isolate failures
//! - **RetryExecutor**: Exponential backoff retry with jitter
//! - **SessionCache**: Token caching for reconnection recovery
//! - **StreamingMetrics**: Comprehensive observability
//!
//! ## Usage
//!
//! ```zig
//! var recovery = try StreamRecovery.init(allocator, .{});
//! defer recovery.deinit();
//!
//! // Execute with retry and circuit breaker protection
//! const result = try recovery.executeWithRecovery(.openai, struct {
//!     pub fn call(ctx: *anyopaque) !TokenStream {
//!         const backend = @ptrCast(*Backend, ctx);
//!         return backend.streamTokens(prompt, config);
//!     }
//! }.call, backend);
//! ```

const std = @import("std");
const platform_time = @import("../../../services/shared/time.zig");
const circuit_breaker = @import("circuit_breaker.zig");
const retry_config = @import("retry_config.zig");
const session_cache = @import("session_cache.zig");
const metrics = @import("metrics.zig");
const backends = @import("backends/mod.zig");

pub const CircuitBreaker = circuit_breaker.CircuitBreaker;
pub const CircuitBreakerConfig = circuit_breaker.CircuitBreakerConfig;
pub const CircuitState = circuit_breaker.CircuitState;
pub const StreamingRetryConfig = retry_config.StreamingRetryConfig;
pub const StreamingError = retry_config.StreamingError;
pub const SessionCache = session_cache.SessionCache;
pub const SessionCacheConfig = session_cache.SessionCacheConfig;
pub const StreamingMetrics = metrics.StreamingMetrics;
pub const BackendType = backends.BackendType;

/// Recovery configuration.
pub const RecoveryConfig = struct {
    /// Enable recovery features.
    enabled: bool = true,

    /// Retry configuration.
    retry: StreamingRetryConfig = .{},

    /// Circuit breaker configuration (applied per-backend).
    circuit_breaker: CircuitBreakerConfig = .{
        .failure_threshold = 5,
        .success_threshold = 2,
        .timeout_ms = 60_000, // 1 minute
        .half_open_max_requests = 3,
    },

    /// Session cache configuration.
    session_cache: SessionCacheConfig = .{
        .max_sessions = 1000,
        .max_tokens_per_session = 100,
        .ttl_ms = 300_000, // 5 minutes
    },

    /// Metrics configuration.
    metrics: metrics.StreamingMetricsConfig = .{},

    /// Use local backend defaults (faster timeouts).
    pub fn forLocalBackend() RecoveryConfig {
        return .{
            .retry = StreamingRetryConfig.forLocalBackend(),
            .circuit_breaker = .{
                .failure_threshold = 3,
                .timeout_ms = 10_000, // 10 seconds
            },
        };
    }

    /// Use external backend defaults (more tolerant).
    pub fn forExternalBackend() RecoveryConfig {
        return .{
            .retry = StreamingRetryConfig.forExternalBackend(),
            .circuit_breaker = .{
                .failure_threshold = 5,
                .timeout_ms = 120_000, // 2 minutes
            },
        };
    }
};

/// Recovery event types sent to clients.
pub const RecoveryEvent = union(enum) {
    /// Retry attempt in progress.
    retry: struct {
        attempt: u32,
        max_attempts: u32,
        delay_ms: u64,
        reason: []const u8,
    },

    /// Circuit breaker state change.
    circuit_breaker: struct {
        backend: BackendType,
        old_state: CircuitState,
        new_state: CircuitState,
    },

    /// Recovery from session cache.
    session_recovery: struct {
        session_id: []const u8,
        tokens_recovered: usize,
        from_event_id: u64,
    },

    /// Backend failover.
    failover: struct {
        from_backend: BackendType,
        to_backend: BackendType,
        reason: []const u8,
    },

    /// Error that client may be able to recover from.
    recoverable_error: struct {
        error_type: StreamingError,
        message: []const u8,
        retry_after_ms: ?u64,
    },

    /// Format as JSON for SSE/WebSocket transmission.
    pub fn toJson(self: RecoveryEvent, allocator: std.mem.Allocator) ![]u8 {
        var json = std.ArrayListUnmanaged(u8).empty;
        errdefer json.deinit(allocator);

        try json.appendSlice(allocator, "{\"type\":\"recovery\",\"event\":");

        switch (self) {
            .retry => |r| {
                try json.appendSlice(allocator, "{\"kind\":\"retry\",\"attempt\":");
                var buf: [32]u8 = undefined;
                const attempt_str = std.fmt.bufPrint(&buf, "{d}", .{r.attempt}) catch "0";
                try json.appendSlice(allocator, attempt_str);
                try json.appendSlice(allocator, ",\"max_attempts\":");
                const max_str = std.fmt.bufPrint(&buf, "{d}", .{r.max_attempts}) catch "0";
                try json.appendSlice(allocator, max_str);
                try json.appendSlice(allocator, ",\"delay_ms\":");
                const delay_str = std.fmt.bufPrint(&buf, "{d}", .{r.delay_ms}) catch "0";
                try json.appendSlice(allocator, delay_str);
                try json.appendSlice(allocator, ",\"reason\":\"");
                try json.appendSlice(allocator, r.reason);
                try json.appendSlice(allocator, "\"}");
            },
            .circuit_breaker => |cb| {
                try json.appendSlice(allocator, "{\"kind\":\"circuit_breaker\",\"backend\":\"");
                try json.appendSlice(allocator, cb.backend.toString());
                try json.appendSlice(allocator, "\",\"old_state\":\"");
                var buf: [16]u8 = undefined;
                const old_state = std.fmt.bufPrint(&buf, "{t}", .{cb.old_state}) catch "unknown";
                try json.appendSlice(allocator, old_state);
                try json.appendSlice(allocator, "\",\"new_state\":\"");
                const new_state = std.fmt.bufPrint(&buf, "{t}", .{cb.new_state}) catch "unknown";
                try json.appendSlice(allocator, new_state);
                try json.appendSlice(allocator, "\"}");
            },
            .session_recovery => |sr| {
                try json.appendSlice(allocator, "{\"kind\":\"session_recovery\",\"session_id\":\"");
                try json.appendSlice(allocator, sr.session_id);
                try json.appendSlice(allocator, "\",\"tokens_recovered\":");
                var buf: [32]u8 = undefined;
                const count_str = std.fmt.bufPrint(&buf, "{d}", .{sr.tokens_recovered}) catch "0";
                try json.appendSlice(allocator, count_str);
                try json.appendSlice(allocator, ",\"from_event_id\":");
                const id_str = std.fmt.bufPrint(&buf, "{d}", .{sr.from_event_id}) catch "0";
                try json.appendSlice(allocator, id_str);
                try json.appendSlice(allocator, "}");
            },
            .failover => |f| {
                try json.appendSlice(allocator, "{\"kind\":\"failover\",\"from_backend\":\"");
                try json.appendSlice(allocator, f.from_backend.toString());
                try json.appendSlice(allocator, "\",\"to_backend\":\"");
                try json.appendSlice(allocator, f.to_backend.toString());
                try json.appendSlice(allocator, "\",\"reason\":\"");
                try json.appendSlice(allocator, f.reason);
                try json.appendSlice(allocator, "\"}");
            },
            .recoverable_error => |e| {
                try json.appendSlice(allocator, "{\"kind\":\"recoverable_error\",\"error_type\":\"");
                var buf: [64]u8 = undefined;
                const err_str = std.fmt.bufPrint(&buf, "{t}", .{e.error_type}) catch "unknown";
                try json.appendSlice(allocator, err_str);
                try json.appendSlice(allocator, "\",\"message\":\"");
                try json.appendSlice(allocator, e.message);
                try json.appendSlice(allocator, "\"");
                if (e.retry_after_ms) |retry_ms| {
                    try json.appendSlice(allocator, ",\"retry_after_ms\":");
                    const retry_str = std.fmt.bufPrint(&buf, "{d}", .{retry_ms}) catch "0";
                    try json.appendSlice(allocator, retry_str);
                }
                try json.appendSlice(allocator, "}");
            },
        }

        try json.appendSlice(allocator, "}");
        return json.toOwnedSlice(allocator);
    }
};

/// Callback for receiving recovery events.
pub const RecoveryCallback = *const fn (event: RecoveryEvent, ctx: ?*anyopaque) void;

/// Stream recovery manager.
///
/// Provides resilient streaming with:
/// - Per-backend circuit breakers
/// - Exponential backoff retry
/// - Session token caching for reconnection
/// - Comprehensive metrics
pub const StreamRecovery = struct {
    allocator: std.mem.Allocator,
    config: RecoveryConfig,

    /// Per-backend circuit breakers.
    circuit_breakers: [4]CircuitBreaker,

    /// Session cache for reconnection.
    cache: ?*SessionCache,

    /// Metrics collector.
    metrics_collector: ?*StreamingMetrics,

    /// Optional callback for recovery events.
    event_callback: ?RecoveryCallback,
    event_callback_ctx: ?*anyopaque,

    const Self = @This();

    /// Initialize stream recovery.
    pub fn init(allocator: std.mem.Allocator, config: RecoveryConfig) !Self {
        var circuit_breakers: [4]CircuitBreaker = undefined;
        for (0..4) |i| {
            circuit_breakers[i] = CircuitBreaker.init(config.circuit_breaker);
        }

        var cache: ?*SessionCache = null;
        if (config.enabled) {
            cache = try allocator.create(SessionCache);
            cache.?.* = SessionCache.init(allocator, config.session_cache);
        }

        var metrics_collector: ?*StreamingMetrics = null;
        if (config.enabled) {
            metrics_collector = try allocator.create(StreamingMetrics);
            metrics_collector.?.* = try StreamingMetrics.init(allocator, config.metrics);
        }

        return Self{
            .allocator = allocator,
            .config = config,
            .circuit_breakers = circuit_breakers,
            .cache = cache,
            .metrics_collector = metrics_collector,
            .event_callback = null,
            .event_callback_ctx = null,
        };
    }

    /// Clean up resources.
    pub fn deinit(self: *Self) void {
        if (self.cache) |c| {
            c.deinit();
            self.allocator.destroy(c);
        }
        if (self.metrics_collector) |m| {
            m.deinit();
            self.allocator.destroy(m);
        }
    }

    /// Set callback for recovery events.
    pub fn setEventCallback(self: *Self, callback: RecoveryCallback, ctx: ?*anyopaque) void {
        self.event_callback = callback;
        self.event_callback_ctx = ctx;
    }

    /// Get circuit breaker for a backend.
    pub fn getCircuitBreaker(self: *Self, backend: BackendType) *CircuitBreaker {
        return &self.circuit_breakers[@intFromEnum(backend)];
    }

    /// Get the session cache.
    pub fn getCache(self: *Self) ?*SessionCache {
        return self.cache;
    }

    /// Get metrics collector.
    pub fn getMetrics(self: *Self) ?*StreamingMetrics {
        return self.metrics_collector;
    }

    /// Check if a backend is available (circuit breaker allows).
    pub fn isBackendAvailable(self: *Self, backend: BackendType) bool {
        if (!self.config.enabled) return true;
        return self.circuit_breakers[@intFromEnum(backend)].canAttempt();
    }

    /// Record a successful operation for a backend.
    pub fn recordSuccess(self: *Self, backend: BackendType) void {
        if (!self.config.enabled) return;

        const cb = &self.circuit_breakers[@intFromEnum(backend)];
        const old_state = cb.getState();
        cb.recordSuccess();
        const new_state = cb.getState();

        // Notify on state change
        if (old_state != new_state) {
            if (self.metrics_collector) |m| {
                if (new_state == .closed) {
                    m.recordCircuitBreakerClose(backend);
                }
            }

            self.emitEvent(.{
                .circuit_breaker = .{
                    .backend = backend,
                    .old_state = old_state,
                    .new_state = new_state,
                },
            });
        }
    }

    /// Record a failed operation for a backend.
    pub fn recordFailure(self: *Self, backend: BackendType) void {
        if (!self.config.enabled) return;

        const cb = &self.circuit_breakers[@intFromEnum(backend)];
        const old_state = cb.getState();
        cb.recordFailure();
        const new_state = cb.getState();

        // Notify on state change
        if (old_state != new_state) {
            if (self.metrics_collector) |m| {
                if (new_state == .open) {
                    m.recordCircuitBreakerOpen(backend);
                } else if (new_state == .closed) {
                    m.recordCircuitBreakerClose(backend);
                }
            }

            self.emitEvent(.{
                .circuit_breaker = .{
                    .backend = backend,
                    .old_state = old_state,
                    .new_state = new_state,
                },
            });
        }
    }

    /// Calculate retry delay with exponential backoff and jitter.
    pub fn calculateRetryDelay(self: *Self, attempt: u32) u64 {
        const base_delay = self.config.retry.base.initial_delay_ns;
        const max_delay = self.config.retry.base.max_delay_ns;
        const multiplier = self.config.retry.base.multiplier;

        // Exponential backoff
        var delay = base_delay;
        var i: u32 = 0;
        while (i < attempt) : (i += 1) {
            const new_delay = @as(u64, @intFromFloat(@as(f64, @floatFromInt(delay)) * multiplier));
            if (new_delay > max_delay) {
                delay = max_delay;
                break;
            }
            delay = new_delay;
        }

        // Add jitter if configured
        if (self.config.retry.base.jitter) {
            const jitter_factor = self.config.retry.base.jitter_factor;
            const jitter_range = @as(u64, @intFromFloat(@as(f64, @floatFromInt(delay)) * jitter_factor));
            if (jitter_range > 0) {
                // Simple PRNG for jitter
                const time_ns: u64 = @intCast(platform_time.nowMs());
                const jitter = time_ns % jitter_range;
                delay = delay + jitter - (jitter_range / 2);
            }
        }

        return delay;
    }

    /// Get the maximum number of retry attempts.
    pub fn maxRetries(self: *Self) u32 {
        return self.config.retry.base.max_retries;
    }

    /// Emit a recovery event to the callback.
    fn emitEvent(self: *Self, event: RecoveryEvent) void {
        if (self.event_callback) |callback| {
            callback(event, self.event_callback_ctx);
        }
    }

    /// Emit a retry event.
    pub fn emitRetryEvent(self: *Self, attempt: u32, delay_ms: u64, reason: []const u8) void {
        self.emitEvent(.{
            .retry = .{
                .attempt = attempt,
                .max_attempts = self.maxRetries(),
                .delay_ms = delay_ms,
                .reason = reason,
            },
        });
    }

    /// Emit a session recovery event.
    pub fn emitSessionRecoveryEvent(
        self: *Self,
        session_id: []const u8,
        tokens_recovered: usize,
        from_event_id: u64,
    ) void {
        if (self.metrics_collector) |m| {
            m.recordRecoveryAttempt();
            if (tokens_recovered > 0) {
                m.recordRecoverySuccess();
                if (tokens_recovered < 10) { // Partial if few tokens
                    m.recordPartialRecovery();
                }
            } else {
                m.recordRecoveryFailure();
            }
        }

        self.emitEvent(.{
            .session_recovery = .{
                .session_id = session_id,
                .tokens_recovered = tokens_recovered,
                .from_event_id = from_event_id,
            },
        });
    }

    /// Emit a failover event.
    pub fn emitFailoverEvent(
        self: *Self,
        from_backend: BackendType,
        to_backend: BackendType,
        reason: []const u8,
    ) void {
        self.emitEvent(.{
            .failover = .{
                .from_backend = from_backend,
                .to_backend = to_backend,
                .reason = reason,
            },
        });
    }

    /// Emit a recoverable error event.
    pub fn emitRecoverableError(
        self: *Self,
        error_type: StreamingError,
        message: []const u8,
        retry_after_ms: ?u64,
    ) void {
        self.emitEvent(.{
            .recoverable_error = .{
                .error_type = error_type,
                .message = message,
                .retry_after_ms = retry_after_ms,
            },
        });
    }

    /// Store a token in the session cache.
    pub fn cacheToken(
        self: *Self,
        session_id: []const u8,
        event_id: u64,
        text: []const u8,
        backend: BackendType,
        prompt_hash: u64,
    ) void {
        if (self.cache) |cache| {
            cache.storeToken(session_id, event_id, text, backend, prompt_hash) catch |err| {
                std.log.warn("StreamRecovery: failed to cache token: {t}", .{err});
            };
        }
    }

    /// Recover tokens from session cache.
    pub fn recoverTokens(
        self: *Self,
        session_id: []const u8,
        last_event_id: u64,
    ) ?[]const session_cache.CachedToken {
        if (self.cache) |cache| {
            const tokens = cache.getTokensSince(session_id, last_event_id);
            if (tokens) |t| {
                if (self.metrics_collector) |m| {
                    m.recordSessionCacheHit();
                }
                return t;
            } else {
                if (self.metrics_collector) |m| {
                    m.recordSessionCacheMiss();
                }
            }
        }
        return null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "StreamRecovery initialization" {
    const allocator = std.testing.allocator;
    var recovery = try StreamRecovery.init(allocator, .{});
    defer recovery.deinit();

    try std.testing.expect(recovery.cache != null);
    try std.testing.expect(recovery.metrics_collector != null);
}

test "StreamRecovery disabled" {
    const allocator = std.testing.allocator;
    var recovery = try StreamRecovery.init(allocator, .{ .enabled = false });
    defer recovery.deinit();

    try std.testing.expect(recovery.cache == null);
    try std.testing.expect(recovery.metrics_collector == null);

    // Should still work but bypass recovery
    try std.testing.expect(recovery.isBackendAvailable(.openai));
}

test "StreamRecovery circuit breaker integration" {
    const allocator = std.testing.allocator;
    var recovery = try StreamRecovery.init(allocator, .{
        .circuit_breaker = .{ .failure_threshold = 2 },
    });
    defer recovery.deinit();

    // Backend should be available initially
    try std.testing.expect(recovery.isBackendAvailable(.openai));

    // Record failures
    recovery.recordFailure(.openai);
    try std.testing.expect(recovery.isBackendAvailable(.openai));
    recovery.recordFailure(.openai);

    // Circuit should be open now
    try std.testing.expect(!recovery.isBackendAvailable(.openai));
}

test "StreamRecovery retry delay calculation" {
    const allocator = std.testing.allocator;
    var recovery = try StreamRecovery.init(allocator, .{
        .retry = .{
            .base = .{
                .initial_delay_ns = 100_000_000, // 100ms
                .max_delay_ns = 1_000_000_000, // 1s
                .multiplier = 2.0,
                .jitter = false, // Disable for predictable testing
            },
        },
    });
    defer recovery.deinit();

    // First attempt: 100ms
    const delay0 = recovery.calculateRetryDelay(0);
    try std.testing.expectEqual(@as(u64, 100_000_000), delay0);

    // Second attempt: 200ms
    const delay1 = recovery.calculateRetryDelay(1);
    try std.testing.expectEqual(@as(u64, 200_000_000), delay1);

    // Third attempt: 400ms
    const delay2 = recovery.calculateRetryDelay(2);
    try std.testing.expectEqual(@as(u64, 400_000_000), delay2);

    // Should cap at max
    const delay5 = recovery.calculateRetryDelay(5);
    try std.testing.expectEqual(@as(u64, 1_000_000_000), delay5);
}

test "RecoveryEvent JSON serialization" {
    const allocator = std.testing.allocator;

    // Test retry event
    const retry_event = RecoveryEvent{
        .retry = .{
            .attempt = 1,
            .max_attempts = 3,
            .delay_ms = 1000,
            .reason = "timeout",
        },
    };
    const retry_json = try retry_event.toJson(allocator);
    defer allocator.free(retry_json);
    try std.testing.expect(std.mem.indexOf(u8, retry_json, "\"retry\"") != null);

    // Test circuit breaker event
    const cb_event = RecoveryEvent{
        .circuit_breaker = .{
            .backend = .openai,
            .old_state = .closed,
            .new_state = .open,
        },
    };
    const cb_json = try cb_event.toJson(allocator);
    defer allocator.free(cb_json);
    try std.testing.expect(std.mem.indexOf(u8, cb_json, "\"circuit_breaker\"") != null);
}

test "StreamRecovery event callback" {
    const allocator = std.testing.allocator;

    const TestContext = struct {
        events_received: usize = 0,

        fn callback(event: RecoveryEvent, ctx: ?*anyopaque) void {
            _ = event;
            const self = @as(*@This(), @ptrCast(@alignCast(ctx)));
            self.events_received += 1;
        }
    };

    var test_ctx = TestContext{};
    var recovery = try StreamRecovery.init(allocator, .{
        .circuit_breaker = .{ .failure_threshold = 1 },
    });
    defer recovery.deinit();

    recovery.setEventCallback(TestContext.callback, &test_ctx);

    // Trigger circuit breaker open
    recovery.recordFailure(.local);

    // Should have received one event
    try std.testing.expectEqual(@as(usize, 1), test_ctx.events_received);
}

test "RecoveryConfig presets" {
    const local_config = RecoveryConfig.forLocalBackend();
    try std.testing.expectEqual(@as(u32, 3), local_config.circuit_breaker.failure_threshold);
    try std.testing.expectEqual(@as(u64, 10_000), local_config.circuit_breaker.timeout_ms);

    const external_config = RecoveryConfig.forExternalBackend();
    try std.testing.expectEqual(@as(u32, 5), external_config.circuit_breaker.failure_threshold);
    try std.testing.expectEqual(@as(u64, 120_000), external_config.circuit_breaker.timeout_ms);
}

test {
    std.testing.refAllDecls(@This());
}
