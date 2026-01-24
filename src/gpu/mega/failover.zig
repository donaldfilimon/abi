//! Backend Auto-Failover with Circuit Breaker Pattern
//!
//! Provides automatic failover between GPU backends with circuit breaker
//! protection, exponential backoff, and health monitoring.

const std = @import("std");
const backend_mod = @import("../backend.zig");

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

/// Failover policy configuration.
pub const FailoverPolicy = struct {
    failure_threshold: u32 = 5,
    success_threshold: u32 = 3,
    timeout_ms: i64 = 30000,
    max_retries: u32 = 3,
    backoff_base_ms: i64 = 100,
    backoff_max_ms: i64 = 10000,
    enable_auto_recovery: bool = true,

    /// Calculate exponential backoff delay for a retry attempt.
    pub fn calculateBackoff(self: FailoverPolicy, attempt: u32) i64 {
        const exp: u6 = @intCast(@min(attempt, 10));
        const backoff = self.backoff_base_ms * (@as(i64, 1) << exp);
        return @min(backoff, self.backoff_max_ms);
    }
};

/// A failover event for logging/metrics.
pub const FailoverEvent = struct {
    timestamp: i64,
    from_backend: backend_mod.Backend,
    to_backend: backend_mod.Backend,
    reason: FailoverReason,
    success: bool,
};

/// Reason for a failover.
pub const FailoverReason = enum {
    circuit_open,
    timeout,
    error_threshold,
    manual,
    health_check,
};

/// Aggregate failover statistics.
pub const FailoverStats = struct {
    total_failovers: u64 = 0,
    successful_failovers: u64 = 0,
    failed_failovers: u64 = 0,
    current_primary: ?backend_mod.Backend = null,
    backends_available: u32 = 0,
    backends_unavailable: u32 = 0,
};

/// Manages backend health and failover decisions.
pub const FailoverManager = struct {
    allocator: std.mem.Allocator,
    policy: FailoverPolicy,
    health: std.AutoHashMap(backend_mod.Backend, BackendHealth),
    priority_order: std.ArrayList(backend_mod.Backend),
    current_primary: ?backend_mod.Backend,
    events: std.ArrayList(FailoverEvent),
    stats: FailoverStats,
    mutex: std.Thread.Mutex,

    const max_events = 1000;

    pub fn init(allocator: std.mem.Allocator, policy: FailoverPolicy) !*FailoverManager {
        const self = try allocator.create(FailoverManager);
        self.* = .{
            .allocator = allocator,
            .policy = policy,
            .health = std.AutoHashMap(backend_mod.Backend, BackendHealth).init(allocator),
            .priority_order = std.ArrayList(backend_mod.Backend).init(allocator),
            .current_primary = null,
            .events = std.ArrayList(FailoverEvent).init(allocator),
            .stats = .{},
            .mutex = .{},
        };
        return self;
    }

    pub fn deinit(self: *FailoverManager) void {
        self.health.deinit();
        self.priority_order.deinit();
        self.events.deinit();
        self.allocator.destroy(self);
    }

    /// Register a backend for health tracking.
    pub fn registerBackend(self: *FailoverManager, backend: backend_mod.Backend) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.health.put(backend, .{ .backend = backend });
        try self.priority_order.append(backend);
        self.stats.backends_available += 1;
        if (self.current_primary == null) {
            self.current_primary = backend;
        }
    }

    /// Record a successful operation on a backend.
    pub fn recordSuccess(self: *FailoverManager, backend: backend_mod.Backend) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.health.getPtr(backend)) |h| {
            h.success_count += 1;
            h.consecutive_successes += 1;
            h.consecutive_failures = 0;
            h.last_success_time = std.time.milliTimestamp();
            h.total_requests += 1;

            // Half-open to closed transition
            if (h.state == .half_open and h.consecutive_successes >= self.policy.success_threshold) {
                h.state = .closed;
                self.stats.backends_available += 1;
                self.stats.backends_unavailable -|= 1;
            }
        }
    }

    /// Record a failed operation on a backend.
    pub fn recordFailure(self: *FailoverManager, backend: backend_mod.Backend) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.health.getPtr(backend)) |h| {
            h.failure_count += 1;
            h.consecutive_failures += 1;
            h.consecutive_successes = 0;
            h.last_failure_time = std.time.milliTimestamp();
            h.total_requests += 1;
            h.total_failures += 1;

            // Closed to open transition
            if (h.state == .closed and h.consecutive_failures >= self.policy.failure_threshold) {
                h.state = .open;
                self.stats.backends_available -|= 1;
                self.stats.backends_unavailable += 1;
            }
        }
    }

    /// Get the first healthy backend in priority order.
    pub fn getHealthyBackend(self: *FailoverManager) ?backend_mod.Backend {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.tryRecoverBackends();

        for (self.priority_order.items) |backend| {
            if (self.health.get(backend)) |h| {
                if (h.isHealthy()) return backend;
            }
        }
        return null;
    }

    /// Failover from a backend to the next healthy one.
    /// Wraps around the priority list to find any available backend.
    pub fn failover(self: *FailoverManager, from: backend_mod.Backend, reason: FailoverReason) ?backend_mod.Backend {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.tryRecoverBackends();

        // Find current backend index
        var from_idx: ?usize = null;
        for (self.priority_order.items, 0..) |backend, i| {
            if (backend == from) {
                from_idx = i;
                break;
            }
        }

        if (from_idx == null) {
            self.recordEvent(from, from, reason, false);
            return null;
        }

        // Try all backends starting from next one, wrapping around
        const start = from_idx.? + 1;
        const count = self.priority_order.items.len;
        for (0..count) |offset| {
            const idx = (start + offset) % count;
            const backend = self.priority_order.items[idx];
            if (backend == from) continue; // Skip the failing backend

            if (self.health.get(backend)) |h| {
                if (h.isHealthy()) {
                    self.recordEvent(from, backend, reason, true);
                    self.current_primary = backend;
                    self.stats.current_primary = backend;
                    return backend;
                }
            }
        }

        self.recordEvent(from, from, reason, false);
        return null;
    }

    fn tryRecoverBackends(self: *FailoverManager) void {
        const now = std.time.milliTimestamp();
        var iter = self.health.iterator();
        while (iter.next()) |entry| {
            const h = entry.value_ptr;
            if (h.state == .open) {
                // Open to half-open transition after timeout
                if (now - h.last_failure_time >= self.policy.timeout_ms) {
                    h.state = .half_open;
                    h.consecutive_failures = 0;
                }
            }
        }
    }

    fn recordEvent(self: *FailoverManager, from: backend_mod.Backend, to: backend_mod.Backend, reason: FailoverReason, success: bool) void {
        // Limit event history
        if (self.events.items.len >= max_events) {
            _ = self.events.orderedRemove(0);
        }
        self.events.append(.{
            .timestamp = std.time.milliTimestamp(),
            .from_backend = from,
            .to_backend = to,
            .reason = reason,
            .success = success,
        }) catch {};

        self.stats.total_failovers += 1;
        if (success) {
            self.stats.successful_failovers += 1;
        } else {
            self.stats.failed_failovers += 1;
        }
    }

    /// Get health status for a backend.
    pub fn getBackendHealth(self: *FailoverManager, backend: backend_mod.Backend) ?BackendHealth {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.health.get(backend);
    }

    /// Get aggregate failover statistics.
    pub fn getStats(self: *FailoverManager) FailoverStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Get the current primary backend.
    pub fn getPrimary(self: *FailoverManager) ?backend_mod.Backend {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.current_primary;
    }

    /// Check if a backend is healthy.
    pub fn isHealthy(self: *FailoverManager, backend: backend_mod.Backend) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.health.get(backend)) |h| {
            return h.isHealthy();
        }
        return false;
    }

    /// Get recent failover events.
    pub fn getRecentEvents(self: *FailoverManager, limit: usize) []const FailoverEvent {
        self.mutex.lock();
        defer self.mutex.unlock();
        const start = if (self.events.items.len > limit) self.events.items.len - limit else 0;
        return self.events.items[start..];
    }
};

test "failover manager basic" {
    const allocator = std.testing.allocator;
    const manager = try FailoverManager.init(allocator, .{});
    defer manager.deinit();

    try manager.registerBackend(.cuda);
    try manager.registerBackend(.vulkan);

    manager.recordSuccess(.cuda);
    const health = manager.getBackendHealth(.cuda);
    try std.testing.expect(health != null);
    try std.testing.expectEqual(@as(u32, 1), health.?.success_count);
}

test "circuit breaker transitions" {
    const allocator = std.testing.allocator;
    const manager = try FailoverManager.init(allocator, .{ .failure_threshold = 3 });
    defer manager.deinit();

    try manager.registerBackend(.cuda);

    // Start closed
    var health = manager.getBackendHealth(.cuda);
    try std.testing.expectEqual(CircuitState.closed, health.?.state);

    // Accumulate failures
    manager.recordFailure(.cuda);
    manager.recordFailure(.cuda);
    manager.recordFailure(.cuda);

    // Should be open now
    health = manager.getBackendHealth(.cuda);
    try std.testing.expectEqual(CircuitState.open, health.?.state);
}

test "failover wraparound" {
    const allocator = std.testing.allocator;
    const manager = try FailoverManager.init(allocator, .{ .failure_threshold = 1 });
    defer manager.deinit();

    try manager.registerBackend(.cuda);
    try manager.registerBackend(.vulkan);
    try manager.registerBackend(.metal);

    // Make vulkan unhealthy (it's in the middle)
    manager.recordFailure(.vulkan);

    // Failover from cuda should skip vulkan and go to metal
    const target = manager.failover(.cuda, .error_threshold);
    try std.testing.expect(target != null);
    try std.testing.expectEqual(backend_mod.Backend.metal, target.?);
}
