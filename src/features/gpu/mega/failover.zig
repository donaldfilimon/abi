//! Backend Auto-Failover with Circuit Breaker Pattern
//!
//! Provides automatic failover between GPU backends with circuit breaker
//! protection, exponential backoff, and health monitoring.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const backend_mod = @import("../backend.zig");
const failover_types = @import("../failover_types.zig");

// Re-export shared types so existing consumers (mega/mod.zig) continue to work.
pub const CircuitState = failover_types.CircuitState;
pub const BackendHealth = failover_types.BackendHealth;
pub const FailoverReason = failover_types.FailoverReason;
pub const FailoverEvent = failover_types.FailoverEvent;
pub const FailoverStats = failover_types.FailoverStats;

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

/// Manages backend health and failover decisions.
pub const FailoverManager = struct {
    allocator: std.mem.Allocator,
    policy: FailoverPolicy,
    health: std.AutoHashMapUnmanaged(backend_mod.Backend, BackendHealth),
    priority_order: std.ArrayListUnmanaged(backend_mod.Backend),
    current_primary: ?backend_mod.Backend,
    events: std.ArrayListUnmanaged(FailoverEvent),
    stats: FailoverStats,
    mutex: sync.Mutex,

    const max_events = 1000;

    pub fn init(allocator: std.mem.Allocator, policy: FailoverPolicy) !*FailoverManager {
        const self = try allocator.create(FailoverManager);
        self.* = .{
            .allocator = allocator,
            .policy = policy,
            .health = .empty,
            .priority_order = .{},
            .current_primary = null,
            .events = .{},
            .stats = .{},
            .mutex = .{},
        };
        return self;
    }

    pub fn deinit(self: *FailoverManager) void {
        self.health.deinit(self.allocator);
        self.priority_order.deinit(self.allocator);
        self.events.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Register a backend for health tracking.
    pub fn registerBackend(self: *FailoverManager, backend: backend_mod.Backend) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.health.put(self.allocator, backend, .{ .backend = backend });
        try self.priority_order.append(self.allocator, backend);
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
            h.last_success_time = time.nowMs();
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
            h.last_failure_time = time.nowMs();
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
        const now = time.nowMs();
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
        self.events.append(self.allocator, .{
            .timestamp = time.nowMs(),
            .from_backend = from,
            .to_backend = to,
            .reason = reason,
            .success = success,
        }) catch |err| {
            std.log.debug("Failed to record failover event: {t}", .{err});
        };

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
