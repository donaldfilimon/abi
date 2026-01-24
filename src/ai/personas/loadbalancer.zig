//! Persona Load Balancer
//!
//! Manages a pool of persona instances, distributing requests based on
//! health, capacity, and historical performance. Includes circuit breaker
//! logic for resilience.
//!
//! Enhanced Features:
//! - Integration with network circuit breaker for proper state management
//! - Health-weighted routing with weighted round-robin
//! - Fallback persona selection
//! - Latency-based weight adjustment

const std = @import("std");
const types = @import("types.zig");
const config = @import("config.zig");
const cb = @import("../../network/circuit_breaker.zig");
const time = @import("../../shared/time.zig");

/// Status of a persona node.
pub const NodeStatus = enum {
    healthy,
    degraded,
    unhealthy,
    circuit_broken,

    pub fn toWeight(self: NodeStatus) f32 {
        return switch (self) {
            .healthy => 1.0,
            .degraded => 0.5,
            .unhealthy => 0.1,
            .circuit_broken => 0.0,
        };
    }
};

/// Represents a single persona instance in the load balancer.
pub const PersonaNode = struct {
    persona_type: types.PersonaType,
    status: NodeStatus = .healthy,
    weight: f32 = 1.0,
    active_requests: u32 = 0,
    consecutive_failures: u32 = 0,
    last_failure_timestamp: i64 = 0,
    /// Average response latency in milliseconds (exponential moving average).
    avg_latency_ms: f32 = 0.0,
    /// Total successful requests (for statistics).
    total_successes: u64 = 0,
    /// Total failed requests (for statistics).
    total_failures: u64 = 0,
    /// Last health check timestamp.
    last_health_check: i64 = 0,
    /// Health score (0.0 - 1.0) from health checker.
    health_score: f32 = 1.0,

    const Self = @This();

    /// Check if the node can accept new requests.
    pub fn isAvailable(self: *const PersonaNode, cfg: config.LoadBalancingConfig) bool {
        if (self.status == .unhealthy or self.status == .circuit_broken) return false;
        if (cfg.max_concurrent_requests > 0 and self.active_requests >= cfg.max_concurrent_requests) return false;
        return true;
    }

    /// Get effective weight considering health and status.
    pub fn getEffectiveWeight(self: *const Self) f32 {
        return self.weight * self.status.toWeight() * self.health_score;
    }

    /// Update latency with exponential moving average.
    pub fn updateLatency(self: *Self, latency_ms: u64) void {
        const alpha: f32 = 0.2; // Smoothing factor
        const lat_f32: f32 = @floatFromInt(latency_ms);
        self.avg_latency_ms = alpha * lat_f32 + (1.0 - alpha) * self.avg_latency_ms;
    }

    /// Get success rate.
    pub fn getSuccessRate(self: *const Self) f32 {
        const total = self.total_successes + self.total_failures;
        if (total == 0) return 1.0;
        return @as(f32, @floatFromInt(self.total_successes)) / @as(f32, @floatFromInt(total));
    }
};

/// Combined score for persona selection.
pub const PersonaScore = struct {
    persona_type: types.PersonaType,
    score: f32,
};

/// Configuration for fallback behavior.
pub const FallbackConfig = struct {
    /// Enable fallback to alternative personas when primary is unavailable.
    enabled: bool = true,
    /// Ordered list of fallback personas (if empty, use all available).
    fallback_order: []const types.PersonaType = &.{},
    /// Maximum fallback attempts.
    max_attempts: u32 = 3,
};

/// Load balancer statistics.
pub const LoadBalancerStats = struct {
    total_requests: u64 = 0,
    successful_routes: u64 = 0,
    failed_routes: u64 = 0,
    fallback_routes: u64 = 0,
    rejected_requests: u64 = 0,

    pub fn successRate(self: LoadBalancerStats) f32 {
        if (self.total_requests == 0) return 1.0;
        return @as(f32, @floatFromInt(self.successful_routes)) / @as(f32, @floatFromInt(self.total_requests));
    }
};

/// Implementation of the persona load balancer.
pub const PersonaLoadBalancer = struct {
    allocator: std.mem.Allocator,
    config: config.LoadBalancingConfig,
    fallback_config: FallbackConfig,
    nodes: std.AutoHashMapUnmanaged(types.PersonaType, PersonaNode),
    /// Circuit breaker registry for per-persona circuit breakers.
    circuit_breakers: cb.CircuitRegistry,
    /// Load balancer statistics.
    stats: LoadBalancerStats,
    mutex: std.Thread.Mutex,

    const Self = @This();

    /// Initialize a new persona load balancer.
    pub fn init(allocator: std.mem.Allocator, cfg: config.LoadBalancingConfig) Self {
        return initWithFallback(allocator, cfg, .{});
    }

    /// Initialize with custom fallback configuration.
    pub fn initWithFallback(allocator: std.mem.Allocator, cfg: config.LoadBalancingConfig, fallback_cfg: FallbackConfig) Self {
        // Configure circuit breaker based on load balancing config
        const cb_config = cb.CircuitConfig{
            .failure_threshold = cfg.circuit_breaker_threshold,
            .timeout_ms = cfg.circuit_breaker_timeout_ms,
            .success_threshold = 2,
            .half_open_max_calls = 3,
            .auto_half_open = true,
        };

        return .{
            .allocator = allocator,
            .config = cfg,
            .fallback_config = fallback_cfg,
            .nodes = .{},
            .circuit_breakers = cb.CircuitRegistry.initWithConfig(allocator, cb_config),
            .stats = .{},
            .mutex = .{},
        };
    }

    /// Deinitialize the load balancer.
    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.circuit_breakers.deinit();
        self.nodes.deinit(self.allocator);
    }

    /// Register a persona in the load balancer.
    pub fn registerPersona(self: *Self, persona_type: types.PersonaType, weight: f32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.nodes.put(self.allocator, persona_type, .{
            .persona_type = persona_type,
            .weight = weight,
        });

        // Register a circuit breaker for this persona
        const name = @tagName(persona_type);
        try self.circuit_breakers.register(name, self.circuit_breakers.default_config);
    }

    /// Select the best available persona instance based on routing scores and node health.
    pub fn selectWithScores(self: *Self, scores: []const PersonaScore) !types.PersonaType {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.stats.total_requests += 1;

        var best_persona: ?types.PersonaType = null;
        var max_combined_score: f32 = -1.0;

        for (scores) |ps| {
            if (self.nodes.getPtr(ps.persona_type)) |node| {
                // Check circuit breaker state
                const cb_name = @tagName(ps.persona_type);
                if (self.circuit_breakers.getBreaker(cb_name)) |breaker| {
                    const state = breaker.getState();
                    if (state == .open) {
                        node.status = .circuit_broken;
                        continue;
                    } else if (state == .half_open) {
                        node.status = .degraded;
                    } else if (node.status == .circuit_broken) {
                        node.status = .healthy;
                    }
                }

                if (!node.isAvailable(self.config)) continue;

                // Combined score = routing_score * effective_weight / (active_requests + 1)
                const load_factor = @as(f32, @floatFromInt(node.active_requests)) + 1.0;
                const latency_penalty = if (node.avg_latency_ms > 0)
                    1.0 / (1.0 + node.avg_latency_ms / 1000.0)
                else
                    1.0;
                const combined = (ps.score * node.getEffectiveWeight() * latency_penalty) / load_factor;

                if (combined > max_combined_score) {
                    max_combined_score = combined;
                    best_persona = ps.persona_type;
                }
            }
        }

        if (best_persona) |p| {
            if (self.nodes.getPtr(p)) |node| {
                node.active_requests += 1;
            }
            self.stats.successful_routes += 1;
            return p;
        }

        // Try fallback if enabled
        if (self.fallback_config.enabled) {
            if (self.selectFallbackUnlocked(scores)) |fallback| {
                self.stats.fallback_routes += 1;
                return fallback;
            }
        }

        self.stats.rejected_requests += 1;
        return error.NoHealthyNodes;
    }

    /// Select a fallback persona (unlocked version).
    fn selectFallbackUnlocked(self: *Self, original_scores: []const PersonaScore) ?types.PersonaType {
        // If fallback order is specified, use it
        if (self.fallback_config.fallback_order.len > 0) {
            for (self.fallback_config.fallback_order) |persona_type| {
                if (self.nodes.getPtr(persona_type)) |node| {
                    // Skip if it was in the original scores (already rejected)
                    var in_original = false;
                    for (original_scores) |ps| {
                        if (ps.persona_type == persona_type) {
                            in_original = true;
                            break;
                        }
                    }
                    if (in_original) continue;

                    if (node.status != .circuit_broken and node.status != .unhealthy) {
                        node.active_requests += 1;
                        return persona_type;
                    }
                }
            }
        }

        // Otherwise, try any available node
        var it = self.nodes.iterator();
        while (it.next()) |entry| {
            const node = entry.value_ptr;
            if (node.status != .circuit_broken and node.status != .unhealthy) {
                // Check it wasn't already tried
                var was_tried = false;
                for (original_scores) |ps| {
                    if (ps.persona_type == entry.key_ptr.*) {
                        was_tried = true;
                        break;
                    }
                }
                if (!was_tried) {
                    node.active_requests += 1;
                    return entry.key_ptr.*;
                }
            }
        }

        return null;
    }

    /// Record a successful interaction for a persona.
    pub fn recordSuccess(self: *Self, persona_type: types.PersonaType) void {
        self.recordSuccessWithLatency(persona_type, 0);
    }

    /// Record a successful interaction with latency measurement.
    pub fn recordSuccessWithLatency(self: *Self, persona_type: types.PersonaType, latency_ms: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.nodes.getPtr(persona_type)) |node| {
            if (node.active_requests > 0) node.active_requests -= 1;
            node.consecutive_failures = 0;
            node.total_successes += 1;
            if (latency_ms > 0) {
                node.updateLatency(latency_ms);
            }
            if (node.status == .circuit_broken or node.status == .degraded) {
                node.status = .healthy;
            }
        }

        // Record success in circuit breaker
        const cb_name = @tagName(persona_type);
        if (self.circuit_breakers.getBreaker(cb_name)) |breaker| {
            breaker.recordSuccess();
        }
    }

    /// Record a failed interaction for a persona.
    pub fn recordFailure(self: *Self, persona_type: types.PersonaType) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.nodes.getPtr(persona_type)) |node| {
            if (node.active_requests > 0) node.active_requests -= 1;
            node.consecutive_failures += 1;
            node.total_failures += 1;
            node.last_failure_timestamp = time.unixSeconds();

            // Status is managed by circuit breaker now
            if (self.config.enable_circuit_breaker) {
                const cb_name = @tagName(persona_type);
                if (self.circuit_breakers.getBreaker(cb_name)) |breaker| {
                    breaker.recordFailure();
                    const state = breaker.getState();
                    node.status = switch (state) {
                        .open => .circuit_broken,
                        .half_open => .degraded,
                        .closed => .healthy,
                    };
                }
            } else if (node.consecutive_failures >= self.config.circuit_breaker_threshold) {
                node.status = .circuit_broken;
            }
        }

        self.stats.failed_routes += 1;
    }

    /// Update node statuses (check for circuit breaker timeouts).
    pub fn update(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Update node status based on circuit breaker state
        var it = self.nodes.iterator();
        while (it.next()) |entry| {
            const node = entry.value_ptr;
            const cb_name = @tagName(entry.key_ptr.*);

            if (self.circuit_breakers.getBreaker(cb_name)) |breaker| {
                const state = breaker.getState();
                node.status = switch (state) {
                    .open => .circuit_broken,
                    .half_open => .degraded,
                    .closed => if (node.health_score < 0.5) .unhealthy else if (node.health_score < 0.8) .degraded else .healthy,
                };
            }
        }
    }

    /// Update health score for a persona (called by health checker).
    pub fn updateHealthScore(self: *Self, persona_type: types.PersonaType, score: f32) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.nodes.getPtr(persona_type)) |node| {
            node.health_score = std.math.clamp(score, 0.0, 1.0);
            node.last_health_check = time.unixSeconds();

            // Update status based on health score
            if (node.status != .circuit_broken) {
                if (score < 0.3) {
                    node.status = .unhealthy;
                } else if (score < 0.7) {
                    node.status = .degraded;
                } else {
                    node.status = .healthy;
                }
            }
        }
    }

    /// Get node statistics.
    pub fn getNodeStats(self: *Self, persona_type: types.PersonaType) ?PersonaNode {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.nodes.get(persona_type);
    }

    /// Get load balancer statistics.
    pub fn getStats(self: *Self) LoadBalancerStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Reset all circuit breakers.
    pub fn resetCircuitBreakers(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.circuit_breakers.resetAll();

        // Reset node statuses
        var it = self.nodes.valueIterator();
        while (it.next()) |node| {
            node.status = .healthy;
            node.consecutive_failures = 0;
        }
    }

    /// Get aggregate circuit breaker stats.
    pub fn getCircuitBreakerStats(self: *Self) cb.AggregateStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.circuit_breakers.getAggregateStats();
    }
};

// Tests

test "load balancer initialization" {
    const allocator = std.testing.allocator;
    var lb = PersonaLoadBalancer.init(allocator, .{});
    defer lb.deinit();

    try lb.registerPersona(.abbey, 1.0);
    try lb.registerPersona(.aviva, 1.0);

    try std.testing.expectEqual(@as(usize, 2), lb.nodes.count());
}

test "node effective weight" {
    var node = PersonaNode{
        .persona_type = .abbey,
        .weight = 1.0,
        .health_score = 0.8,
        .status = .healthy,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 0.8), node.getEffectiveWeight(), 0.01);

    node.status = .degraded;
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), node.getEffectiveWeight(), 0.01);
}

test "node success rate" {
    var node = PersonaNode{
        .persona_type = .abbey,
        .total_successes = 8,
        .total_failures = 2,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 0.8), node.getSuccessRate(), 0.01);
}

test "load balancer stats" {
    var stats = LoadBalancerStats{
        .total_requests = 100,
        .successful_routes = 90,
        .failed_routes = 5,
        .fallback_routes = 5,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 0.9), stats.successRate(), 0.01);
}

test "node status weight" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), NodeStatus.healthy.toWeight(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), NodeStatus.degraded.toWeight(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), NodeStatus.circuit_broken.toWeight(), 0.01);
}
