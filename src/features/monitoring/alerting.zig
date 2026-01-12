//! Alerting rules system for monitoring and observability.
//!
//! Provides configurable alert rules with thresholds, conditions,
//! and notification handlers for proactive system monitoring.
//!
//! Features:
//! - Configurable alert rules with thresholds
//! - Multiple severity levels (info, warning, critical)
//! - Alert state tracking (pending, firing, resolved)
//! - Callback-based notification handlers
//! - Rate limiting to prevent alert storms
//! - Alert grouping and deduplication
//!
//! Usage:
//!   var manager = try AlertManager.init(allocator, .{});
//!   defer manager.deinit();
//!   try manager.addRule(.{
//!       .name = "high_error_rate",
//!       .metric = "errors_total",
//!       .condition = .greater_than,
//!       .threshold = 100,
//!       .severity = .critical,
//!   });
//!   try manager.evaluate(metrics);

const std = @import("std");
const observability = @import("../../shared/observability/mod.zig");

/// Alert severity levels.
pub const AlertSeverity = enum {
    info,
    warning,
    critical,

    pub fn toString(self: AlertSeverity) []const u8 {
        return switch (self) {
            .info => "info",
            .warning => "warning",
            .critical => "critical",
        };
    }

    pub fn toInt(self: AlertSeverity) u8 {
        return switch (self) {
            .info => 1,
            .warning => 2,
            .critical => 3,
        };
    }
};

/// Alert state.
pub const AlertState = enum {
    /// Alert condition not met.
    inactive,
    /// Alert condition met, waiting for duration.
    pending,
    /// Alert is actively firing.
    firing,
    /// Alert was firing but is now resolved.
    resolved,

    pub fn toString(self: AlertState) []const u8 {
        return switch (self) {
            .inactive => "inactive",
            .pending => "pending",
            .firing => "firing",
            .resolved => "resolved",
        };
    }
};

/// Comparison condition for alert rules.
pub const AlertCondition = enum {
    greater_than,
    greater_than_or_equal,
    less_than,
    less_than_or_equal,
    equal,
    not_equal,
    rate_of_change,
    absent,

    pub fn evaluate(self: AlertCondition, value: f64, threshold: f64) bool {
        return switch (self) {
            .greater_than => value > threshold,
            .greater_than_or_equal => value >= threshold,
            .less_than => value < threshold,
            .less_than_or_equal => value <= threshold,
            .equal => @abs(value - threshold) < 0.0001,
            .not_equal => @abs(value - threshold) >= 0.0001,
            .rate_of_change => false, // Requires historical data
            .absent => false, // Special case
        };
    }

    pub fn toString(self: AlertCondition) []const u8 {
        return switch (self) {
            .greater_than => ">",
            .greater_than_or_equal => ">=",
            .less_than => "<",
            .less_than_or_equal => "<=",
            .equal => "==",
            .not_equal => "!=",
            .rate_of_change => "rate",
            .absent => "absent",
        };
    }
};

/// Alert rule definition.
pub const AlertRule = struct {
    /// Unique name for the alert rule.
    name: []const u8,
    /// Metric name to evaluate.
    metric: []const u8,
    /// Condition for triggering.
    condition: AlertCondition = .greater_than,
    /// Threshold value.
    threshold: f64,
    /// Alert severity.
    severity: AlertSeverity = .warning,
    /// Duration the condition must be true before firing (ms).
    for_duration_ms: u64 = 0,
    /// Labels to attach to the alert.
    labels: ?[]const u8 = null,
    /// Human-readable description.
    description: ?[]const u8 = null,
    /// Runbook URL for remediation.
    runbook_url: ?[]const u8 = null,
    /// Whether the rule is enabled.
    enabled: bool = true,
};

/// Active alert instance.
pub const Alert = struct {
    /// Rule that triggered this alert.
    rule_name: []const u8,
    /// Current alert state.
    state: AlertState,
    /// Current metric value.
    current_value: f64,
    /// Severity level.
    severity: AlertSeverity,
    /// Labels for grouping/routing.
    labels: ?[]const u8,
    /// When the alert started pending.
    started_at_ms: u64,
    /// When the alert started firing.
    fired_at_ms: ?u64,
    /// When the alert was resolved.
    resolved_at_ms: ?u64,
    /// Number of times this alert has fired.
    fire_count: u64,
    /// Description from the rule.
    description: ?[]const u8,
};

/// Alert notification callback.
pub const AlertCallback = *const fn (alert: Alert, user_data: ?*anyopaque) void;

/// Alert notification handler.
pub const AlertHandler = struct {
    callback: AlertCallback,
    user_data: ?*anyopaque,
    /// Minimum severity to handle.
    min_severity: AlertSeverity,
    /// Rate limit (minimum ms between notifications).
    rate_limit_ms: u64,
    /// Last notification time.
    last_notified_ms: u64,
};

/// Configuration for AlertManager.
pub const AlertManagerConfig = struct {
    /// How often to evaluate rules (ms).
    evaluation_interval_ms: u64 = 15_000,
    /// Default duration before firing.
    default_for_duration_ms: u64 = 60_000,
    /// Maximum number of active alerts.
    max_active_alerts: usize = 1000,
    /// Enable alert grouping.
    enable_grouping: bool = true,
    /// Group wait time (ms).
    group_wait_ms: u64 = 30_000,
    /// Repeat interval for firing alerts (ms).
    repeat_interval_ms: u64 = 300_000,
    /// Resolve timeout (ms without condition being met).
    resolve_timeout_ms: u64 = 300_000,
};

/// Alert manager error types.
pub const AlertError = error{
    RuleNotFound,
    DuplicateRule,
    MaxAlertsExceeded,
    InvalidRule,
    HandlerNotFound,
    OutOfMemory,
};

/// Alert statistics.
pub const AlertStats = struct {
    /// Total rules configured.
    total_rules: usize,
    /// Active rules (enabled).
    active_rules: usize,
    /// Currently firing alerts.
    firing_alerts: usize,
    /// Pending alerts.
    pending_alerts: usize,
    /// Resolved alerts (recent).
    resolved_alerts: usize,
    /// Total evaluations performed.
    evaluations: u64,
    /// Total notifications sent.
    notifications_sent: u64,
    /// Notifications suppressed by rate limiting.
    notifications_suppressed: u64,
};

/// Alert manager for handling alerting rules and notifications.
pub const AlertManager = struct {
    allocator: std.mem.Allocator,
    config: AlertManagerConfig,
    /// Configured alert rules.
    rules: std.StringHashMapUnmanaged(AlertRule),
    /// Active alerts (keyed by rule name).
    alerts: std.StringHashMapUnmanaged(Alert),
    /// Notification handlers.
    handlers: std.ArrayListUnmanaged(AlertHandler),
    /// Rule state tracking (for duration).
    rule_state: std.StringHashMapUnmanaged(RuleState),
    /// Statistics.
    stats: AlertStats,
    /// Current timestamp (for testing).
    current_time_ms: u64,

    const RuleState = struct {
        condition_met_since: ?u64,
        last_value: f64,
        evaluation_count: u64,
    };

    /// Initialize a new AlertManager.
    pub fn init(allocator: std.mem.Allocator, config: AlertManagerConfig) !AlertManager {
        return AlertManager{
            .allocator = allocator,
            .config = config,
            .rules = .{},
            .alerts = .{},
            .handlers = .{},
            .rule_state = .{},
            .stats = .{
                .total_rules = 0,
                .active_rules = 0,
                .firing_alerts = 0,
                .pending_alerts = 0,
                .resolved_alerts = 0,
                .evaluations = 0,
                .notifications_sent = 0,
                .notifications_suppressed = 0,
            },
            .current_time_ms = 0,
        };
    }

    /// Clean up resources.
    pub fn deinit(self: *AlertManager) void {
        // Free rule names
        var rules_iter = self.rules.iterator();
        while (rules_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.rules.deinit(self.allocator);

        // Free alert data
        var alerts_iter = self.alerts.iterator();
        while (alerts_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.alerts.deinit(self.allocator);

        // Free rule state
        var state_iter = self.rule_state.iterator();
        while (state_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.rule_state.deinit(self.allocator);

        self.handlers.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add an alert rule.
    pub fn addRule(self: *AlertManager, rule: AlertRule) !void {
        // Check for duplicate
        if (self.rules.contains(rule.name)) {
            return AlertError.DuplicateRule;
        }

        // Copy rule name for ownership
        const name_copy = try self.allocator.dupe(u8, rule.name);
        errdefer self.allocator.free(name_copy);

        // Store rule
        try self.rules.put(self.allocator, name_copy, rule);

        // Initialize rule state
        const state_key = try self.allocator.dupe(u8, rule.name);
        errdefer self.allocator.free(state_key);

        try self.rule_state.put(self.allocator, state_key, RuleState{
            .condition_met_since = null,
            .last_value = 0,
            .evaluation_count = 0,
        });

        self.stats.total_rules += 1;
        if (rule.enabled) {
            self.stats.active_rules += 1;
        }
    }

    /// Remove an alert rule.
    pub fn removeRule(self: *AlertManager, name: []const u8) !void {
        const result = self.rules.fetchRemove(name) orelse return AlertError.RuleNotFound;
        self.allocator.free(result.key);

        // Remove associated state
        if (self.rule_state.fetchRemove(name)) |state_result| {
            self.allocator.free(state_result.key);
        }

        // Remove any active alerts
        if (self.alerts.fetchRemove(name)) |alert_result| {
            self.allocator.free(alert_result.key);
            if (alert_result.value.state == .firing) {
                self.stats.firing_alerts -= 1;
            } else if (alert_result.value.state == .pending) {
                self.stats.pending_alerts -= 1;
            }
        }

        self.stats.total_rules -= 1;
        if (result.value.enabled) {
            self.stats.active_rules -= 1;
        }
    }

    /// Enable or disable a rule.
    pub fn setRuleEnabled(self: *AlertManager, name: []const u8, enabled: bool) !void {
        const rule = self.rules.getPtr(name) orelse return AlertError.RuleNotFound;
        if (rule.enabled != enabled) {
            rule.enabled = enabled;
            if (enabled) {
                self.stats.active_rules += 1;
            } else {
                self.stats.active_rules -= 1;
            }
        }
    }

    /// Register a notification handler.
    pub fn addHandler(
        self: *AlertManager,
        callback: AlertCallback,
        user_data: ?*anyopaque,
        min_severity: AlertSeverity,
        rate_limit_ms: u64,
    ) !void {
        const handler = AlertHandler{
            .callback = callback,
            .user_data = user_data,
            .min_severity = min_severity,
            .rate_limit_ms = rate_limit_ms,
            .last_notified_ms = 0,
        };
        try self.handlers.append(self.allocator, handler);
    }

    /// Evaluate all rules against current metric values.
    pub fn evaluate(self: *AlertManager, metrics: *const MetricValues) !void {
        self.stats.evaluations += 1;

        var rules_iter = self.rules.iterator();
        while (rules_iter.next()) |entry| {
            const rule = entry.value_ptr.*;
            if (!rule.enabled) continue;

            // Get metric value
            const state = self.rule_state.getPtr(rule.name) orelse continue;
            const value = metrics.get(rule.metric) orelse {
                // Handle absent metric
                if (rule.condition == .absent) {
                    try self.processConditionMet(rule);
                } else {
                    try self.processConditionNotMet(rule);
                }
                continue;
            };

            // Evaluate condition
            const condition_met = self.evaluateCondition(rule, value, state);

            // Update rule state
            state.last_value = value;
            state.evaluation_count += 1;

            if (condition_met) {
                try self.processConditionMet(rule);
            } else {
                try self.processConditionNotMet(rule);
            }
        }
    }

    /// Evaluate a single rule with a specific value.
    pub fn evaluateRule(self: *AlertManager, rule_name: []const u8, value: f64) !bool {
        const rule = self.rules.get(rule_name) orelse return AlertError.RuleNotFound;
        if (!rule.enabled) return false;

        const state = self.rule_state.getPtr(rule.name) orelse return false;
        const condition_met = self.evaluateCondition(rule, value, state);

        // Update rule state
        state.last_value = value;
        state.evaluation_count += 1;

        if (condition_met) {
            try self.processConditionMet(rule);
        } else {
            try self.processConditionNotMet(rule);
        }

        return condition_met;
    }

    /// Get current alert for a rule.
    pub fn getAlert(self: *const AlertManager, rule_name: []const u8) ?Alert {
        return self.alerts.get(rule_name);
    }

    /// Get all firing alerts.
    pub fn getFiringAlerts(self: *const AlertManager) !std.ArrayListUnmanaged(Alert) {
        var list = std.ArrayListUnmanaged(Alert){};
        errdefer list.deinit(self.allocator);

        var iter = self.alerts.valueIterator();
        while (iter.next()) |alert| {
            if (alert.state == .firing) {
                try list.append(self.allocator, alert.*);
            }
        }

        return list;
    }

    /// Get all pending alerts.
    pub fn getPendingAlerts(self: *const AlertManager) !std.ArrayListUnmanaged(Alert) {
        var list = std.ArrayListUnmanaged(Alert){};
        errdefer list.deinit(self.allocator);

        var iter = self.alerts.valueIterator();
        while (iter.next()) |alert| {
            if (alert.state == .pending) {
                try list.append(self.allocator, alert.*);
            }
        }

        return list;
    }

    /// Get statistics.
    pub fn getStats(self: *const AlertManager) AlertStats {
        return self.stats;
    }

    /// Set current time (for testing/simulation).
    pub fn setCurrentTime(self: *AlertManager, time_ms: u64) void {
        self.current_time_ms = time_ms;
    }

    /// Advance time and process timeouts.
    pub fn tick(self: *AlertManager, elapsed_ms: u64) !void {
        self.current_time_ms += elapsed_ms;

        // Check for alerts that should transition to resolved
        var iter = self.alerts.iterator();
        while (iter.next()) |entry| {
            const alert = entry.value_ptr;
            if (alert.state == .pending) {
                const state = self.rule_state.get(alert.rule_name) orelse continue;
                if (state.condition_met_since == null) {
                    // Condition no longer met, cancel pending
                    alert.state = .inactive;
                    self.stats.pending_alerts -= 1;
                }
            }
        }
    }

    // Private methods

    fn processConditionMet(self: *AlertManager, rule: AlertRule) !void {
        const state = self.rule_state.getPtr(rule.name) orelse return;

        if (state.condition_met_since == null) {
            state.condition_met_since = self.current_time_ms;
        }

        const duration_met = self.current_time_ms - state.condition_met_since.?;
        const required_duration = if (rule.for_duration_ms > 0)
            rule.for_duration_ms
        else
            self.config.default_for_duration_ms;

        // Get or create alert
        const existing = self.alerts.getPtr(rule.name);

        if (existing) |alert| {
            if (alert.state == .pending and duration_met >= required_duration) {
                // Transition to firing
                alert.state = .firing;
                alert.fired_at_ms = self.current_time_ms;
                alert.fire_count += 1;
                alert.current_value = state.last_value;
                self.stats.pending_alerts -= 1;
                self.stats.firing_alerts += 1;
                try self.notify(alert.*);
            } else if (alert.state == .firing) {
                // Already firing, update value
                alert.current_value = state.last_value;
            } else if (alert.state == .resolved or alert.state == .inactive) {
                // Start pending again
                alert.state = .pending;
                alert.started_at_ms = self.current_time_ms;
                alert.current_value = state.last_value;
                alert.resolved_at_ms = null;
                self.stats.pending_alerts += 1;
                if (self.stats.resolved_alerts > 0) {
                    self.stats.resolved_alerts -= 1;
                }
            }
        } else {
            // Create new alert
            const name_copy = try self.allocator.dupe(u8, rule.name);
            errdefer self.allocator.free(name_copy);

            var new_alert = Alert{
                .rule_name = rule.name,
                .state = if (required_duration == 0) .firing else .pending,
                .current_value = state.last_value,
                .severity = rule.severity,
                .labels = rule.labels,
                .started_at_ms = self.current_time_ms,
                .fired_at_ms = if (required_duration == 0) self.current_time_ms else null,
                .resolved_at_ms = null,
                .fire_count = if (required_duration == 0) 1 else 0,
                .description = rule.description,
            };

            try self.alerts.put(self.allocator, name_copy, new_alert);

            if (new_alert.state == .firing) {
                self.stats.firing_alerts += 1;
                try self.notify(new_alert);
            } else {
                self.stats.pending_alerts += 1;
            }
        }
    }

    fn processConditionNotMet(self: *AlertManager, rule: AlertRule) !void {
        const state = self.rule_state.getPtr(rule.name) orelse return;
        state.condition_met_since = null;

        if (self.alerts.getPtr(rule.name)) |alert| {
            if (alert.state == .firing) {
                // Transition to resolved
                alert.state = .resolved;
                alert.resolved_at_ms = self.current_time_ms;
                self.stats.firing_alerts -= 1;
                self.stats.resolved_alerts += 1;
                try self.notify(alert.*);
            } else if (alert.state == .pending) {
                // Cancel pending
                alert.state = .inactive;
                self.stats.pending_alerts -= 1;
            }
        }
    }

    fn notify(self: *AlertManager, alert: Alert) !void {
        for (self.handlers.items) |*handler| {
            // Check severity filter
            if (alert.severity.toInt() < handler.min_severity.toInt()) continue;

            // Check rate limit
            if (handler.rate_limit_ms > 0) {
                if (self.current_time_ms - handler.last_notified_ms < handler.rate_limit_ms) {
                    self.stats.notifications_suppressed += 1;
                    continue;
                }
            }

            // Call handler
            handler.callback(alert, handler.user_data);
            handler.last_notified_ms = self.current_time_ms;
            self.stats.notifications_sent += 1;
        }
    }

    fn evaluateCondition(
        self: *const AlertManager,
        rule: AlertRule,
        value: f64,
        state: *const RuleState,
    ) bool {
        _ = self;
        return switch (rule.condition) {
            .rate_of_change => {
                if (state.evaluation_count == 0) {
                    return false;
                }
                return (value - state.last_value) >= rule.threshold;
            },
            .absent => false,
            else => rule.condition.evaluate(value, rule.threshold),
        };
    }
};

/// Simple metric values container for evaluation.
pub const MetricValues = struct {
    values: std.StringHashMapUnmanaged(f64),

    pub fn init() MetricValues {
        return .{ .values = .{} };
    }

    pub fn deinit(self: *MetricValues, allocator: std.mem.Allocator) void {
        self.values.deinit(allocator);
    }

    pub fn set(self: *MetricValues, allocator: std.mem.Allocator, name: []const u8, value: f64) !void {
        try self.values.put(allocator, name, value);
    }

    pub fn get(self: *const MetricValues, name: []const u8) ?f64 {
        return self.values.get(name);
    }
};

/// Builder for creating alert rules.
pub const AlertRuleBuilder = struct {
    rule: AlertRule,

    pub fn init(name: []const u8, metric: []const u8) AlertRuleBuilder {
        return .{
            .rule = .{
                .name = name,
                .metric = metric,
                .threshold = 0,
            },
        };
    }

    pub fn threshold(self: *AlertRuleBuilder, value: f64) *AlertRuleBuilder {
        self.rule.threshold = value;
        return self;
    }

    pub fn condition(self: *AlertRuleBuilder, cond: AlertCondition) *AlertRuleBuilder {
        self.rule.condition = cond;
        return self;
    }

    pub fn severity(self: *AlertRuleBuilder, sev: AlertSeverity) *AlertRuleBuilder {
        self.rule.severity = sev;
        return self;
    }

    pub fn forDuration(self: *AlertRuleBuilder, duration_ms: u64) *AlertRuleBuilder {
        self.rule.for_duration_ms = duration_ms;
        return self;
    }

    pub fn labels(self: *AlertRuleBuilder, l: []const u8) *AlertRuleBuilder {
        self.rule.labels = l;
        return self;
    }

    pub fn description(self: *AlertRuleBuilder, desc: []const u8) *AlertRuleBuilder {
        self.rule.description = desc;
        return self;
    }

    pub fn runbook(self: *AlertRuleBuilder, url: []const u8) *AlertRuleBuilder {
        self.rule.runbook_url = url;
        return self;
    }

    pub fn enabled(self: *AlertRuleBuilder, e: bool) *AlertRuleBuilder {
        self.rule.enabled = e;
        return self;
    }

    pub fn build(self: *const AlertRuleBuilder) AlertRule {
        return self.rule;
    }
};

/// Create a new alert rule builder.
pub fn createRule(name: []const u8, metric: []const u8) AlertRuleBuilder {
    return AlertRuleBuilder.init(name, metric);
}

// Tests

fn testCallback(alert: Alert, user_data: ?*anyopaque) void {
    if (user_data) |data| {
        const counter: *u64 = @ptrCast(@alignCast(data));
        counter.* += 1;
        _ = alert;
    }
}

test "alert manager initialization" {
    const allocator = std.testing.allocator;

    var manager = try AlertManager.init(allocator, .{});
    defer manager.deinit();

    const stats = manager.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.total_rules);
}

test "add and remove rules" {
    const allocator = std.testing.allocator;

    var manager = try AlertManager.init(allocator, .{});
    defer manager.deinit();

    try manager.addRule(.{
        .name = "high_cpu",
        .metric = "cpu_percent",
        .threshold = 80,
        .severity = .warning,
    });

    try manager.addRule(.{
        .name = "low_memory",
        .metric = "memory_available_mb",
        .condition = .less_than,
        .threshold = 1000,
        .severity = .critical,
    });

    var stats = manager.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.total_rules);
    try std.testing.expectEqual(@as(usize, 2), stats.active_rules);

    try manager.removeRule("high_cpu");

    stats = manager.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.total_rules);
}

test "evaluate rule triggers alert" {
    const allocator = std.testing.allocator;

    var manager = try AlertManager.init(allocator, .{
        .default_for_duration_ms = 0, // Fire immediately
    });
    defer manager.deinit();

    try manager.addRule(.{
        .name = "high_errors",
        .metric = "errors_total",
        .condition = .greater_than,
        .threshold = 100,
        .severity = .critical,
        .for_duration_ms = 0,
    });

    // Should trigger
    const triggered = try manager.evaluateRule("high_errors", 150);
    try std.testing.expect(triggered);

    const alert = manager.getAlert("high_errors");
    try std.testing.expect(alert != null);
    try std.testing.expectEqual(AlertState.firing, alert.?.state);

    const stats = manager.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.firing_alerts);
}

test "alert resolves when condition not met" {
    const allocator = std.testing.allocator;

    var manager = try AlertManager.init(allocator, .{
        .default_for_duration_ms = 0,
    });
    defer manager.deinit();

    try manager.addRule(.{
        .name = "test_alert",
        .metric = "test_metric",
        .threshold = 50,
        .for_duration_ms = 0,
    });

    // Trigger alert
    _ = try manager.evaluateRule("test_alert", 100);
    try std.testing.expectEqual(AlertState.firing, manager.getAlert("test_alert").?.state);

    // Resolve alert
    _ = try manager.evaluateRule("test_alert", 30);
    try std.testing.expectEqual(AlertState.resolved, manager.getAlert("test_alert").?.state);
}

test "notification callback" {
    const allocator = std.testing.allocator;

    var manager = try AlertManager.init(allocator, .{
        .default_for_duration_ms = 0,
    });
    defer manager.deinit();

    var callback_count: u64 = 0;
    try manager.addHandler(testCallback, &callback_count, .info, 0);

    try manager.addRule(.{
        .name = "notify_test",
        .metric = "test",
        .threshold = 10,
        .for_duration_ms = 0,
    });

    _ = try manager.evaluateRule("notify_test", 20);

    try std.testing.expectEqual(@as(u64, 1), callback_count);
}

test "alert conditions" {
    try std.testing.expect(AlertCondition.greater_than.evaluate(100, 50));
    try std.testing.expect(!AlertCondition.greater_than.evaluate(50, 100));

    try std.testing.expect(AlertCondition.less_than.evaluate(50, 100));
    try std.testing.expect(!AlertCondition.less_than.evaluate(100, 50));

    try std.testing.expect(AlertCondition.equal.evaluate(100, 100));
    try std.testing.expect(!AlertCondition.equal.evaluate(100, 101));

    try std.testing.expect(AlertCondition.not_equal.evaluate(100, 101));
    try std.testing.expect(!AlertCondition.not_equal.evaluate(100, 100));

    try std.testing.expect(AlertCondition.greater_than_or_equal.evaluate(100, 100));
    try std.testing.expect(AlertCondition.greater_than_or_equal.evaluate(101, 100));

    try std.testing.expect(AlertCondition.less_than_or_equal.evaluate(100, 100));
    try std.testing.expect(AlertCondition.less_than_or_equal.evaluate(99, 100));
}

test "rule builder" {
    var builder = createRule("high_latency", "request_latency_ms");
    const alert_rule = builder
        .threshold(500)
        .condition(.greater_than)
        .severity(.warning)
        .forDuration(60000)
        .description("Request latency is high")
        .build();

    try std.testing.expectEqualStrings("high_latency", alert_rule.name);
    try std.testing.expectEqualStrings("request_latency_ms", alert_rule.metric);
    try std.testing.expectEqual(@as(f64, 500), alert_rule.threshold);
    try std.testing.expectEqual(AlertCondition.greater_than, alert_rule.condition);
    try std.testing.expectEqual(AlertSeverity.warning, alert_rule.severity);
}

test "severity ordering" {
    try std.testing.expect(AlertSeverity.info.toInt() < AlertSeverity.warning.toInt());
    try std.testing.expect(AlertSeverity.warning.toInt() < AlertSeverity.critical.toInt());
}
