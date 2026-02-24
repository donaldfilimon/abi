//! Persona Health Checker Module
//!
//! Monitors the health of persona instances and provides health scores
//! to the load balancer for intelligent routing decisions.
//!
//! Features:
//! - Periodic health checks with configurable intervals
//! - Multi-factor health scoring (latency, error rate, availability)
//! - Health history tracking for trend analysis
//! - Integration with metrics and alerting

const std = @import("std");
const types = @import("types.zig");
const metrics_mod = @import("metrics.zig");
const alerts = @import("alerts.zig");
const loadbalancer = @import("loadbalancer.zig");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");

/// Health status of a persona.
pub const HealthStatus = enum {
    healthy,
    degraded,
    unhealthy,
    unknown,

    pub fn toScore(self: HealthStatus) f32 {
        return switch (self) {
            .healthy => 1.0,
            .degraded => 0.6,
            .unhealthy => 0.2,
            .unknown => 0.5,
        };
    }

    pub fn fromScore(score: f32) HealthStatus {
        if (score >= 0.8) return .healthy;
        if (score >= 0.5) return .degraded;
        if (score > 0.0) return .unhealthy;
        return .unknown;
    }
};

/// Detailed health check result.
pub const HealthCheckResult = struct {
    persona: types.PersonaType,
    status: HealthStatus,
    score: f32,
    /// Latency component (0.0 - 1.0).
    latency_score: f32 = 1.0,
    /// Error rate component (0.0 - 1.0).
    error_rate_score: f32 = 1.0,
    /// Availability component (0.0 - 1.0).
    availability_score: f32 = 1.0,
    /// Custom health indicators.
    custom_score: f32 = 1.0,
    /// Timestamp of check.
    timestamp: i64,
    /// Optional message.
    message: ?[]const u8 = null,
};

/// Health history entry.
const HealthHistoryEntry = struct {
    timestamp: i64,
    score: f32,
    status: HealthStatus,
};

/// Configuration for health checking.
pub const HealthCheckerConfig = struct {
    /// Interval between health checks in milliseconds.
    check_interval_ms: u64 = 30000,
    /// Number of history entries to retain.
    history_size: usize = 100,
    /// Weight for latency in health score calculation.
    latency_weight: f32 = 0.3,
    /// Weight for error rate in health score calculation.
    error_rate_weight: f32 = 0.4,
    /// Weight for availability in health score calculation.
    availability_weight: f32 = 0.3,
    /// Latency threshold for degraded status (ms).
    latency_threshold_degraded_ms: u64 = 1000,
    /// Latency threshold for unhealthy status (ms).
    latency_threshold_unhealthy_ms: u64 = 5000,
    /// Error rate threshold for degraded status.
    error_rate_threshold_degraded: f32 = 0.05,
    /// Error rate threshold for unhealthy status.
    error_rate_threshold_unhealthy: f32 = 0.15,
    /// Enable health check logging.
    enable_logging: bool = false,
};

/// Health checker for personas.
pub const HealthChecker = struct {
    allocator: std.mem.Allocator,
    config: HealthCheckerConfig,
    /// Reference to metrics manager.
    metrics: ?*metrics_mod.PersonaMetrics = null,
    /// Reference to load balancer.
    load_balancer: ?*loadbalancer.PersonaLoadBalancer = null,
    /// Health history per persona.
    history: std.AutoHashMapUnmanaged(types.PersonaType, std.ArrayListUnmanaged(HealthHistoryEntry)),
    /// Last check results.
    last_results: std.AutoHashMapUnmanaged(types.PersonaType, HealthCheckResult),
    /// Custom health check functions.
    custom_checks: std.ArrayListUnmanaged(CustomHealthCheck),
    mutex: sync.Mutex,

    const Self = @This();

    /// Custom health check function type.
    pub const CustomHealthCheckFn = *const fn (types.PersonaType) f32;

    const CustomHealthCheck = struct {
        name: []const u8,
        check_fn: CustomHealthCheckFn,
        weight: f32,
    };

    /// Initialize the health checker.
    pub fn init(allocator: std.mem.Allocator) Self {
        return initWithConfig(allocator, .{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, config: HealthCheckerConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .metrics = null,
            .load_balancer = null,
            .history = .{},
            .last_results = .{},
            .custom_checks = .{},
            .mutex = .{},
        };
    }

    /// Initialize with metrics and load balancer references.
    pub fn initWithDependencies(
        allocator: std.mem.Allocator,
        config: HealthCheckerConfig,
        metrics_ref: *metrics_mod.PersonaMetrics,
        lb_ref: *loadbalancer.PersonaLoadBalancer,
    ) Self {
        var checker = initWithConfig(allocator, config);
        checker.metrics = metrics_ref;
        checker.load_balancer = lb_ref;
        return checker;
    }

    /// Shutdown the health checker.
    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var it = self.history.valueIterator();
        while (it.next()) |list| {
            list.deinit(self.allocator);
        }
        self.history.deinit(self.allocator);
        self.last_results.deinit(self.allocator);
        self.custom_checks.deinit(self.allocator);
    }

    /// Register a persona for health checking.
    pub fn registerPersona(self: *Self, persona_type: types.PersonaType) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.history.contains(persona_type)) {
            try self.history.put(self.allocator, persona_type, .{});
        }
    }

    /// Add a custom health check.
    pub fn addCustomCheck(self: *Self, name: []const u8, check_fn: CustomHealthCheckFn, weight: f32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.custom_checks.append(self.allocator, .{
            .name = name,
            .check_fn = check_fn,
            .weight = weight,
        });
    }

    /// Perform health check for a specific persona.
    pub fn checkPersona(self: *Self, persona_type: types.PersonaType) !HealthCheckResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.unixSeconds();

        // Calculate component scores
        var latency_score: f32 = 1.0;
        var error_rate_score: f32 = 1.0;
        var availability_score: f32 = 1.0;
        var custom_score: f32 = 1.0;

        // Get latency and error rate from metrics
        if (self.metrics) |metrics_ref| {
            if (metrics_ref.getStats(persona_type)) |stats| {
                // Latency score
                if (stats.latency) |lat| {
                    latency_score = self.calculateLatencyScore(lat.p99);
                }

                // Error rate score
                const error_rate = 1.0 - stats.success_rate;
                error_rate_score = self.calculateErrorRateScore(error_rate);
            }
        }

        // Get availability from load balancer
        if (self.load_balancer) |lb_ref| {
            if (lb_ref.getNodeStats(persona_type)) |node| {
                availability_score = switch (node.status) {
                    .healthy => 1.0,
                    .degraded => 0.7,
                    .unhealthy => 0.3,
                    .circuit_broken => 0.0,
                };
            }
        }

        // Run custom checks
        if (self.custom_checks.items.len > 0) {
            var total_weight: f32 = 0;
            var weighted_sum: f32 = 0;
            for (self.custom_checks.items) |check| {
                const check_result = check.check_fn(persona_type);
                weighted_sum += check_result * check.weight;
                total_weight += check.weight;
            }
            if (total_weight > 0) {
                custom_score = weighted_sum / total_weight;
            }
        }

        // Calculate overall score
        const score = self.calculateOverallScore(
            latency_score,
            error_rate_score,
            availability_score,
            custom_score,
        );

        const status = HealthStatus.fromScore(score);

        const result = HealthCheckResult{
            .persona = persona_type,
            .status = status,
            .score = score,
            .latency_score = latency_score,
            .error_rate_score = error_rate_score,
            .availability_score = availability_score,
            .custom_score = custom_score,
            .timestamp = now,
        };

        // Store result
        try self.last_results.put(self.allocator, persona_type, result);

        // Add to history
        if (self.history.getPtr(persona_type)) |history_list| {
            if (history_list.items.len >= self.config.history_size) {
                _ = history_list.orderedRemove(0);
            }
            try history_list.append(self.allocator, .{
                .timestamp = now,
                .score = score,
                .status = status,
            });
        }

        // Update load balancer
        if (self.load_balancer) |lb_ref| {
            lb_ref.updateHealthScore(persona_type, score);
        }

        return result;
    }

    /// Check all registered personas.
    pub fn checkAll(self: *Self) ![]HealthCheckResult {
        var results: std.ArrayListUnmanaged(HealthCheckResult) = .{};
        errdefer results.deinit(self.allocator);

        // Get list of personas
        self.mutex.lock();
        var persona_list: std.ArrayListUnmanaged(types.PersonaType) = .{};
        defer persona_list.deinit(self.allocator);

        var it = self.history.keyIterator();
        while (it.next()) |key| {
            try persona_list.append(self.allocator, key.*);
        }
        self.mutex.unlock();

        // Check each persona
        for (persona_list.items) |persona| {
            const result = try self.checkPersona(persona);
            try results.append(self.allocator, result);
        }

        return results.toOwnedSlice(self.allocator);
    }

    /// Get last health check result for a persona.
    pub fn getLastResult(self: *Self, persona_type: types.PersonaType) ?HealthCheckResult {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.last_results.get(persona_type);
    }

    /// Get health trend for a persona (average score over recent history).
    pub fn getHealthTrend(self: *Self, persona_type: types.PersonaType) ?f32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.history.getPtr(persona_type)) |history_list| {
            if (history_list.items.len == 0) return null;

            var sum: f32 = 0;
            for (history_list.items) |entry| {
                sum += entry.score;
            }
            return sum / @as(f32, @floatFromInt(history_list.items.len));
        }
        return null;
    }

    /// Get aggregate health status across all personas.
    pub fn getAggregateHealth(self: *Self) AggregateHealth {
        self.mutex.lock();
        defer self.mutex.unlock();

        var result = AggregateHealth{};

        var it = self.last_results.valueIterator();
        while (it.next()) |check_result| {
            result.total_personas += 1;
            result.total_score += check_result.score;

            switch (check_result.status) {
                .healthy => result.healthy_count += 1,
                .degraded => result.degraded_count += 1,
                .unhealthy => result.unhealthy_count += 1,
                .unknown => result.unknown_count += 1,
            }
        }

        if (result.total_personas > 0) {
            result.average_score = result.total_score / @as(f32, @floatFromInt(result.total_personas));
        }

        return result;
    }

    /// Calculate latency score (1.0 = fast, 0.0 = slow).
    fn calculateLatencyScore(self: *const Self, latency_ms: f64) f32 {
        const lat_f32: f32 = @floatCast(latency_ms);
        const threshold_degraded: f32 = @floatFromInt(self.config.latency_threshold_degraded_ms);
        const threshold_unhealthy: f32 = @floatFromInt(self.config.latency_threshold_unhealthy_ms);

        if (lat_f32 <= threshold_degraded) {
            return 1.0;
        } else if (lat_f32 <= threshold_unhealthy) {
            // Linear interpolation between degraded and unhealthy
            const range = threshold_unhealthy - threshold_degraded;
            const excess = lat_f32 - threshold_degraded;
            return 1.0 - (excess / range) * 0.5; // 1.0 -> 0.5
        } else {
            // Beyond unhealthy threshold
            const excess = lat_f32 - threshold_unhealthy;
            return @max(0.1, 0.5 - excess / threshold_unhealthy * 0.4);
        }
    }

    /// Calculate error rate score (1.0 = no errors, 0.0 = all errors).
    fn calculateErrorRateScore(self: *const Self, error_rate: f32) f32 {
        if (error_rate <= self.config.error_rate_threshold_degraded) {
            return 1.0;
        } else if (error_rate <= self.config.error_rate_threshold_unhealthy) {
            // Linear interpolation
            const range = self.config.error_rate_threshold_unhealthy - self.config.error_rate_threshold_degraded;
            const excess = error_rate - self.config.error_rate_threshold_degraded;
            return 1.0 - (excess / range) * 0.5;
        } else {
            return @max(0.1, 0.5 - (error_rate - self.config.error_rate_threshold_unhealthy) * 2.0);
        }
    }

    /// Calculate overall health score from components.
    fn calculateOverallScore(
        self: *const Self,
        latency_score: f32,
        error_rate_score: f32,
        availability_score: f32,
        custom_score: f32,
    ) f32 {
        const base_weight = self.config.latency_weight + self.config.error_rate_weight + self.config.availability_weight;

        var score = latency_score * self.config.latency_weight +
            error_rate_score * self.config.error_rate_weight +
            availability_score * self.config.availability_weight;

        // Include custom score if custom checks exist
        if (self.custom_checks.items.len > 0) {
            const custom_weight: f32 = 0.2; // 20% weight for custom checks
            score = score * (1.0 - custom_weight) / base_weight + custom_score * custom_weight;
        } else {
            score = score / base_weight;
        }

        return std.math.clamp(score, 0.0, 1.0);
    }
};

/// Aggregate health across all personas.
pub const AggregateHealth = struct {
    total_personas: u32 = 0,
    healthy_count: u32 = 0,
    degraded_count: u32 = 0,
    unhealthy_count: u32 = 0,
    unknown_count: u32 = 0,
    average_score: f32 = 1.0,
    total_score: f32 = 0,

    pub fn isSystemHealthy(self: AggregateHealth) bool {
        return self.unhealthy_count == 0 and self.healthy_count > 0;
    }

    pub fn getOverallStatus(self: AggregateHealth) HealthStatus {
        if (self.total_personas == 0) return .unknown;
        if (self.unhealthy_count > self.total_personas / 2) return .unhealthy;
        if (self.degraded_count + self.unhealthy_count > self.healthy_count) return .degraded;
        return .healthy;
    }
};

// Tests

test "health status scoring" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), HealthStatus.healthy.toScore(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), HealthStatus.degraded.toScore(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), HealthStatus.unhealthy.toScore(), 0.01);
}

test "health status from score" {
    try std.testing.expectEqual(HealthStatus.healthy, HealthStatus.fromScore(0.9));
    try std.testing.expectEqual(HealthStatus.degraded, HealthStatus.fromScore(0.6));
    try std.testing.expectEqual(HealthStatus.unhealthy, HealthStatus.fromScore(0.3));
}

test "health checker initialization" {
    var checker = HealthChecker.init(std.testing.allocator);
    defer checker.deinit();

    try checker.registerPersona(.abbey);
    try checker.registerPersona(.aviva);

    try std.testing.expectEqual(@as(usize, 2), checker.history.count());
}

test "aggregate health" {
    var health = AggregateHealth{
        .total_personas = 3,
        .healthy_count = 2,
        .degraded_count = 1,
        .unhealthy_count = 0,
    };

    try std.testing.expect(health.isSystemHealthy());
    try std.testing.expectEqual(HealthStatus.healthy, health.getOverallStatus());
}

test "aggregate health degraded" {
    var health = AggregateHealth{
        .total_personas = 3,
        .healthy_count = 1,
        .degraded_count = 2,
        .unhealthy_count = 0,
    };

    try std.testing.expect(health.isSystemHealthy());
    try std.testing.expectEqual(HealthStatus.degraded, health.getOverallStatus());
}

test "latency score calculation" {
    const checker = HealthChecker.init(std.testing.allocator);

    // Fast response
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), checker.calculateLatencyScore(500), 0.01);

    // At degraded threshold
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), checker.calculateLatencyScore(1000), 0.01);

    // Between degraded and unhealthy
    try std.testing.expect(checker.calculateLatencyScore(3000) < 1.0);
    try std.testing.expect(checker.calculateLatencyScore(3000) > 0.5);
}

test "error rate score calculation" {
    const checker = HealthChecker.init(std.testing.allocator);

    // No errors
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), checker.calculateErrorRateScore(0.0), 0.01);

    // Below threshold
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), checker.calculateErrorRateScore(0.03), 0.01);

    // At degraded threshold
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), checker.calculateErrorRateScore(0.05), 0.01);

    // Above unhealthy threshold
    try std.testing.expect(checker.calculateErrorRateScore(0.20) < 0.5);
}

test {
    std.testing.refAllDecls(@This());
}
