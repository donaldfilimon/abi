//! Persona Alert Rules Module
//!
//! Defines alerting rules for the Multi-Persona AI Assistant system.
//! Monitors key metrics and triggers alerts when thresholds are breached.
//!
//! Features:
//! - Configurable alert rules per persona
//! - Multiple severity levels
//! - Alert aggregation and deduplication
//! - Alert history tracking

const std = @import("std");
const types = @import("types.zig");
const time = @import("../../shared/time.zig");

/// Alert severity levels.
pub const AlertSeverity = enum {
    /// Informational alert, no action required.
    info,
    /// Warning level, should be investigated.
    warning,
    /// Critical alert, requires immediate attention.
    critical,
    /// Emergency level, system may be degraded.
    emergency,

    pub fn getPrefix(self: AlertSeverity) []const u8 {
        return switch (self) {
            .info => "[INFO]",
            .warning => "[WARN]",
            .critical => "[CRIT]",
            .emergency => "[EMER]",
        };
    }

    pub fn requiresNotification(self: AlertSeverity) bool {
        return switch (self) {
            .info => false,
            .warning => true,
            .critical => true,
            .emergency => true,
        };
    }
};

/// Types of conditions that can trigger an alert.
pub const AlertCondition = enum {
    /// Metric exceeds threshold.
    threshold_exceeded,
    /// Metric falls below threshold.
    threshold_below,
    /// Rate of change exceeds limit.
    rate_of_change,
    /// Consecutive failures exceed count.
    consecutive_failures,
    /// Error rate exceeds threshold.
    error_rate,
    /// Latency exceeds threshold.
    latency_exceeded,
    /// Custom condition check.
    custom,
};

/// An alert rule definition.
pub const AlertRule = struct {
    /// Unique name for the rule.
    name: []const u8,
    /// Description of what this rule monitors.
    description: []const u8,
    /// Type of condition.
    condition: AlertCondition,
    /// Threshold value (interpretation depends on condition).
    threshold: f64,
    /// Severity when triggered.
    severity: AlertSeverity,
    /// Persona this rule applies to (null = all personas).
    persona: ?types.PersonaType = null,
    /// Whether this rule is enabled.
    enabled: bool = true,
    /// Cooldown period between alerts (seconds).
    cooldown_seconds: u64 = 300,
    /// Minimum occurrences before alerting.
    min_occurrences: u32 = 1,
};

/// Pre-defined alert rules for the persona system.
pub const PERSONA_ALERTS = [_]AlertRule{
    // Latency alerts
    .{
        .name = "high_latency_p50",
        .description = "P50 response latency exceeds 1 second",
        .condition = .latency_exceeded,
        .threshold = 1000.0,
        .severity = .warning,
    },
    .{
        .name = "high_latency_p99",
        .description = "P99 response latency exceeds 5 seconds",
        .condition = .latency_exceeded,
        .threshold = 5000.0,
        .severity = .critical,
    },
    .{
        .name = "extreme_latency",
        .description = "Response latency exceeds 30 seconds",
        .condition = .latency_exceeded,
        .threshold = 30000.0,
        .severity = .emergency,
    },

    // Error rate alerts
    .{
        .name = "elevated_error_rate",
        .description = "Error rate exceeds 5%",
        .condition = .error_rate,
        .threshold = 0.05,
        .severity = .warning,
    },
    .{
        .name = "high_error_rate",
        .description = "Error rate exceeds 10%",
        .condition = .error_rate,
        .threshold = 0.10,
        .severity = .critical,
    },

    // Success rate alerts
    .{
        .name = "low_success_rate",
        .description = "Success rate falls below 95%",
        .condition = .threshold_below,
        .threshold = 0.95,
        .severity = .warning,
    },
    .{
        .name = "critical_success_rate",
        .description = "Success rate falls below 90%",
        .condition = .threshold_below,
        .threshold = 0.90,
        .severity = .critical,
    },

    // Consecutive failure alerts
    .{
        .name = "consecutive_failures",
        .description = "5 or more consecutive request failures",
        .condition = .consecutive_failures,
        .threshold = 5.0,
        .severity = .critical,
        .cooldown_seconds = 60,
    },

    // Persona-specific alerts
    .{
        .name = "abbey_low_empathy",
        .description = "Abbey empathy score below threshold",
        .condition = .threshold_below,
        .threshold = 0.7,
        .severity = .warning,
        .persona = .abbey,
    },
    .{
        .name = "aviva_low_accuracy",
        .description = "Aviva factual accuracy below threshold",
        .condition = .threshold_below,
        .threshold = 0.85,
        .severity = .warning,
        .persona = .aviva,
    },
};

/// A triggered alert instance.
pub const Alert = struct {
    /// The rule that triggered this alert.
    rule: AlertRule,
    /// When the alert was triggered.
    triggered_at: i64,
    /// Current metric value that triggered the alert.
    current_value: f64,
    /// Persona involved (if applicable).
    persona: ?types.PersonaType,
    /// Human-readable message.
    message: []const u8,
    /// Whether this alert has been acknowledged.
    acknowledged: bool = false,
    /// When the alert was resolved (0 if still active).
    resolved_at: i64 = 0,
};

/// Configuration for the alert manager.
pub const AlertManagerConfig = struct {
    /// Maximum number of active alerts to track.
    max_active_alerts: usize = 100,
    /// Maximum alert history to retain.
    max_history: usize = 1000,
    /// Default cooldown between duplicate alerts.
    default_cooldown_seconds: u64 = 300,
    /// Whether to aggregate similar alerts.
    aggregate_similar: bool = true,
    /// Custom rules to add to defaults.
    custom_rules: []const AlertRule = &.{},
};

/// Manages alert rules and triggered alerts.
pub const AlertManager = struct {
    allocator: std.mem.Allocator,
    config: AlertManagerConfig,
    /// All active alert rules.
    rules: std.ArrayListUnmanaged(AlertRule),
    /// Currently active alerts.
    active_alerts: std.ArrayListUnmanaged(Alert),
    /// Alert history.
    history: std.ArrayListUnmanaged(Alert),
    /// Last alert time per rule (for cooldown).
    last_alert_time: std.StringHashMapUnmanaged(i64),
    /// Consecutive failure counters per persona.
    failure_counters: std.AutoHashMapUnmanaged(types.PersonaType, u32),

    const Self = @This();

    /// Initialize the alert manager.
    pub fn init(allocator: std.mem.Allocator) Self {
        return initWithConfig(allocator, .{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, config: AlertManagerConfig) Self {
        var manager = Self{
            .allocator = allocator,
            .config = config,
            .rules = .{},
            .active_alerts = .{},
            .history = .{},
            .last_alert_time = .{},
            .failure_counters = .{},
        };

        // Load default rules
        for (PERSONA_ALERTS) |rule| {
            manager.rules.append(allocator, rule) catch |err| {
                std.log.debug("Failed to load default alert rule: {t}", .{err});
            };
        }

        // Load custom rules
        for (config.custom_rules) |rule| {
            manager.rules.append(allocator, rule) catch |err| {
                std.log.debug("Failed to load custom alert rule: {t}", .{err});
            };
        }

        return manager;
    }

    /// Shutdown and free resources.
    pub fn deinit(self: *Self) void {
        // Free heap-allocated alert messages
        for (self.active_alerts.items) |alert| {
            self.allocator.free(alert.message);
        }
        for (self.history.items) |alert| {
            self.allocator.free(alert.message);
        }
        self.rules.deinit(self.allocator);
        self.active_alerts.deinit(self.allocator);
        self.history.deinit(self.allocator);
        self.last_alert_time.deinit(self.allocator);
        self.failure_counters.deinit(self.allocator);
    }

    /// Check metrics against all rules and trigger alerts as needed.
    pub fn checkMetrics(self: *Self, metrics: MetricSnapshot) !void {
        const now = time.unixSeconds();

        for (self.rules.items) |rule| {
            if (!rule.enabled) continue;

            // Check persona filter
            if (rule.persona) |p| {
                if (metrics.persona != p) continue;
            }

            // Check cooldown
            if (self.isInCooldown(rule.name, now)) continue;

            // Evaluate condition
            const should_alert = self.evaluateCondition(rule, metrics);

            if (should_alert) {
                try self.triggerAlert(rule, metrics, now);
            }
        }
    }

    /// Check if a rule is in cooldown period.
    fn isInCooldown(self: *const Self, rule_name: []const u8, now: i64) bool {
        if (self.last_alert_time.get(rule_name)) |last_time| {
            const cooldown = self.config.default_cooldown_seconds;
            return (now - last_time) < @as(i64, @intCast(cooldown));
        }
        return false;
    }

    /// Evaluate a rule condition against metrics.
    fn evaluateCondition(self: *Self, rule: AlertRule, metrics: MetricSnapshot) bool {
        return switch (rule.condition) {
            .threshold_exceeded => metrics.getValue(rule.name) > rule.threshold,
            .threshold_below => metrics.getValue(rule.name) < rule.threshold,
            .latency_exceeded => metrics.latency_p99_ms > rule.threshold,
            .error_rate => metrics.error_rate > rule.threshold,
            .consecutive_failures => blk: {
                const count = self.failure_counters.get(metrics.persona) orelse 0;
                break :blk @as(f64, @floatFromInt(count)) >= rule.threshold;
            },
            .rate_of_change => false, // Would need historical data
            .custom => false, // Would need callback
        };
    }

    /// Trigger an alert.
    fn triggerAlert(self: *Self, rule: AlertRule, metrics: MetricSnapshot, now: i64) !void {
        // Build message - heap-allocate to avoid use-after-scope
        const message = try std.fmt.allocPrint(self.allocator, "{s}: {s} (current: {d:.2}, threshold: {d:.2})", .{
            rule.severity.getPrefix(),
            rule.description,
            metrics.getValue(rule.name),
            rule.threshold,
        });
        errdefer self.allocator.free(message);

        const alert = Alert{
            .rule = rule,
            .triggered_at = now,
            .current_value = metrics.getValue(rule.name),
            .persona = metrics.persona,
            .message = message,
        };

        // Add to active alerts (with limit)
        if (self.active_alerts.items.len < self.config.max_active_alerts) {
            try self.active_alerts.append(self.allocator, alert);
        } else {
            // If we can't add the alert, free the message we allocated
            self.allocator.free(message);
        }

        // Update last alert time
        try self.last_alert_time.put(self.allocator, rule.name, now);
    }

    /// Record a request result for consecutive failure tracking.
    pub fn recordResult(self: *Self, persona: types.PersonaType, success: bool) !void {
        if (success) {
            // Reset counter on success
            _ = self.failure_counters.fetchRemove(persona);
        } else {
            // Increment counter on failure
            const current = self.failure_counters.get(persona) orelse 0;
            try self.failure_counters.put(self.allocator, persona, current + 1);
        }
    }

    /// Acknowledge an alert.
    pub fn acknowledgeAlert(self: *Self, index: usize) void {
        if (index < self.active_alerts.items.len) {
            self.active_alerts.items[index].acknowledged = true;
        }
    }

    /// Resolve an alert.
    pub fn resolveAlert(self: *Self, index: usize) void {
        if (index < self.active_alerts.items.len) {
            var alert = self.active_alerts.orderedRemove(index);
            alert.resolved_at = time.unixSeconds();

            // Add to history
            if (self.history.items.len >= self.config.max_history) {
                _ = self.history.orderedRemove(0);
            }
            self.history.append(self.allocator, alert) catch |err| {
                std.log.debug("Failed to append alert to history: {t}", .{err});
            };
        }
    }

    /// Get active alerts.
    pub fn getActiveAlerts(self: *const Self) []const Alert {
        return self.active_alerts.items;
    }

    /// Get alerts by severity.
    pub fn getAlertsBySeverity(self: *const Self, severity: AlertSeverity) []const Alert {
        var count: usize = 0;
        for (self.active_alerts.items) |alert| {
            if (alert.rule.severity == severity) count += 1;
        }
        return self.active_alerts.items[0..count];
    }

    /// Get alert count by severity.
    pub fn getAlertCounts(self: *const Self) AlertCounts {
        var counts = AlertCounts{};
        for (self.active_alerts.items) |alert| {
            switch (alert.rule.severity) {
                .info => counts.info += 1,
                .warning => counts.warning += 1,
                .critical => counts.critical += 1,
                .emergency => counts.emergency += 1,
            }
        }
        return counts;
    }

    /// Add a custom alert rule.
    pub fn addRule(self: *Self, rule: AlertRule) !void {
        try self.rules.append(self.allocator, rule);
    }

    /// Enable or disable a rule by name.
    pub fn setRuleEnabled(self: *Self, rule_name: []const u8, enabled: bool) void {
        for (self.rules.items) |*rule| {
            if (std.mem.eql(u8, rule.name, rule_name)) {
                rule.enabled = enabled;
                break;
            }
        }
    }
};

/// Snapshot of metrics for alert evaluation.
pub const MetricSnapshot = struct {
    persona: types.PersonaType,
    total_requests: u64 = 0,
    success_rate: f64 = 1.0,
    error_rate: f64 = 0.0,
    latency_p50_ms: f64 = 0.0,
    latency_p99_ms: f64 = 0.0,
    satisfaction_score: f64 = 1.0,
    custom_metrics: std.StringHashMapUnmanaged(f64) = .{},

    pub fn getValue(self: *const MetricSnapshot, metric_name: []const u8) f64 {
        if (std.mem.indexOf(u8, metric_name, "success") != null) return self.success_rate;
        if (std.mem.indexOf(u8, metric_name, "error") != null) return self.error_rate;
        if (std.mem.indexOf(u8, metric_name, "latency_p99") != null) return self.latency_p99_ms;
        if (std.mem.indexOf(u8, metric_name, "latency_p50") != null) return self.latency_p50_ms;
        if (std.mem.indexOf(u8, metric_name, "satisfaction") != null) return self.satisfaction_score;
        if (std.mem.indexOf(u8, metric_name, "empathy") != null) return self.satisfaction_score;
        if (std.mem.indexOf(u8, metric_name, "accuracy") != null) return self.satisfaction_score;
        return self.custom_metrics.get(metric_name) orelse 0.0;
    }
};

/// Summary of alert counts by severity.
pub const AlertCounts = struct {
    info: usize = 0,
    warning: usize = 0,
    critical: usize = 0,
    emergency: usize = 0,

    pub fn total(self: AlertCounts) usize {
        return self.info + self.warning + self.critical + self.emergency;
    }

    pub fn hasUrgent(self: AlertCounts) bool {
        return self.critical > 0 or self.emergency > 0;
    }
};

// Tests

test "alert manager initialization" {
    var manager = AlertManager.init(std.testing.allocator);
    defer manager.deinit();

    try std.testing.expect(manager.rules.items.len > 0);
    try std.testing.expectEqual(@as(usize, 0), manager.active_alerts.items.len);
}

test "alert severity levels" {
    try std.testing.expect(AlertSeverity.critical.requiresNotification());
    try std.testing.expect(!AlertSeverity.info.requiresNotification());
}

test "metric snapshot getValue" {
    const snapshot = MetricSnapshot{
        .persona = .abbey,
        .success_rate = 0.95,
        .error_rate = 0.05,
        .latency_p99_ms = 1500.0,
    };

    try std.testing.expectApproxEqAbs(@as(f64, 0.95), snapshot.getValue("low_success_rate"), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 1500.0), snapshot.getValue("high_latency_p99"), 0.01);
}

test "record success resets failure counter" {
    var manager = AlertManager.init(std.testing.allocator);
    defer manager.deinit();

    // Record failures
    try manager.recordResult(.abbey, false);
    try manager.recordResult(.abbey, false);

    // Record success should reset
    try manager.recordResult(.abbey, true);

    try std.testing.expect(!manager.failure_counters.contains(.abbey));
}

test "alert counts" {
    var counts = AlertCounts{
        .warning = 2,
        .critical = 1,
    };

    try std.testing.expectEqual(@as(usize, 3), counts.total());
    try std.testing.expect(counts.hasUrgent());
}

test "default alert rules loaded" {
    var manager = AlertManager.init(std.testing.allocator);
    defer manager.deinit();

    // Should have loaded the default rules
    try std.testing.expect(manager.rules.items.len >= PERSONA_ALERTS.len);
}
