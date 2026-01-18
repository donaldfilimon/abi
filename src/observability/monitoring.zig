//! Monitoring and Alerting System for Observability.
//!
//! Provides metrics exporters (Prometheus, StatsD) and a configurable
//! alerting rules system for proactive system monitoring.

const std = @import("std");
const observability = @import("mod.zig");

// ============================================================================
// Alerting System (from alerting.zig)
// ============================================================================

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
            .absent => false, // Special case handled in evaluate()
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
    name: []const u8,
    metric: []const u8,
    condition: AlertCondition = .greater_than,
    threshold: f64,
    severity: AlertSeverity = .warning,
    for_duration_ms: u64 = 0,
    labels: ?[]const u8 = null,
    description: ?[]const u8 = null,
    runbook_url: ?[]const u8 = null,
    enabled: bool = true,
};

/// Active alert instance.
pub const Alert = struct {
    rule_name: []const u8,
    state: AlertState,
    current_value: f64,
    severity: AlertSeverity,
    labels: ?[]const u8,
    started_at_ms: u64,
    fired_at_ms: ?u64,
    resolved_at_ms: ?u64,
    fire_count: u64,
    description: ?[]const u8,
};

pub const AlertCallback = *const fn (alert: Alert, user_data: ?*anyopaque) void;

pub const AlertHandler = struct {
    callback: AlertCallback,
    user_data: ?*anyopaque,
    min_severity: AlertSeverity,
    rate_limit_ms: u64,
    last_notified_ms: u64,
};

pub const AlertManagerConfig = struct {
    evaluation_interval_ms: u64 = 15_000,
    default_for_duration_ms: u64 = 60_000,
    max_active_alerts: usize = 1000,
    enable_grouping: bool = true,
    group_wait_ms: u64 = 30_000,
    repeat_interval_ms: u64 = 300_000,
    resolve_timeout_ms: u64 = 300_000,
};

pub const AlertError = error{
    RuleNotFound,
    DuplicateRule,
    MaxAlertsExceeded,
    InvalidRule,
    HandlerNotFound,
    OutOfMemory,
};

pub const AlertStats = struct {
    total_rules: usize,
    active_rules: usize,
    firing_alerts: usize,
    pending_alerts: usize,
    resolved_alerts: usize,
    evaluations: u64,
    notifications_sent: u64,
    notifications_suppressed: u64,
};

pub const AlertManager = struct {
    allocator: std.mem.Allocator,
    config: AlertManagerConfig,
    rules: std.StringHashMapUnmanaged(AlertRule),
    alerts: std.StringHashMapUnmanaged(Alert),
    handlers: std.ArrayListUnmanaged(AlertHandler),
    rule_state: std.StringHashMapUnmanaged(RuleState),
    stats: AlertStats,
    current_time_ms: u64,

    const RuleState = struct {
        condition_met_since: ?u64,
        last_value: f64,
        evaluation_count: u64,
    };

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

    pub fn deinit(self: *AlertManager) void {
        var rules_iter = self.rules.iterator();
        while (rules_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.rules.deinit(self.allocator);

        var alerts_iter = self.alerts.iterator();
        while (alerts_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.alerts.deinit(self.allocator);

        var state_iter = self.rule_state.iterator();
        while (state_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.rule_state.deinit(self.allocator);

        self.handlers.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn addRule(self: *AlertManager, rule: AlertRule) !void {
        if (self.rules.contains(rule.name)) return AlertError.DuplicateRule;
        const name_copy = try self.allocator.dupe(u8, rule.name);
        errdefer self.allocator.free(name_copy);
        try self.rules.put(self.allocator, name_copy, rule);
        const state_key = try self.allocator.dupe(u8, rule.name);
        errdefer self.allocator.free(state_key);
        try self.rule_state.put(self.allocator, state_key, RuleState{
            .condition_met_since = null,
            .last_value = 0,
            .evaluation_count = 0,
        });
        self.stats.total_rules += 1;
        if (rule.enabled) self.stats.active_rules += 1;
    }

    pub fn removeRule(self: *AlertManager, name: []const u8) !void {
        const result = self.rules.fetchRemove(name) orelse return AlertError.RuleNotFound;
        self.allocator.free(result.key);
        if (self.rule_state.fetchRemove(name)) |state_result| self.allocator.free(state_result.key);
        if (self.alerts.fetchRemove(name)) |alert_result| {
            self.allocator.free(alert_result.key);
            if (alert_result.value.state == .firing) self.stats.firing_alerts -= 1 else if (alert_result.value.state == .pending) self.stats.pending_alerts -= 1;
        }
        self.stats.total_rules -= 1;
        if (result.value.enabled) self.stats.active_rules -= 1;
    }

    pub fn setRuleEnabled(self: *AlertManager, name: []const u8, enabled: bool) !void {
        const rule = self.rules.getPtr(name) orelse return AlertError.RuleNotFound;
        if (rule.enabled != enabled) {
            rule.enabled = enabled;
            if (enabled) self.stats.active_rules += 1 else self.stats.active_rules -= 1;
        }
    }

    pub fn addHandler(
        self: *AlertManager,
        callback: AlertCallback,
        user_data: ?*anyopaque,
        min_severity: AlertSeverity,
        rate_limit_ms: u64,
    ) !void {
        try self.handlers.append(self.allocator, AlertHandler{
            .callback = callback,
            .user_data = user_data,
            .min_severity = min_severity,
            .rate_limit_ms = rate_limit_ms,
            .last_notified_ms = 0,
        });
    }

    pub fn evaluate(self: *AlertManager, metrics: *const MetricValues) !void {
        self.stats.evaluations += 1;
        var rules_iter = self.rules.iterator();
        while (rules_iter.next()) |entry| {
            const rule = entry.value_ptr.*;
            if (!rule.enabled) continue;
            const state = self.rule_state.getPtr(rule.name) orelse continue;
            const value = metrics.get(rule.metric) orelse {
                if (rule.condition == .absent) try self.processConditionMet(rule) else try self.processConditionNotMet(rule);
                continue;
            };
            const condition_met = self.evaluateCondition(rule, value, state);
            state.last_value = value;
            state.evaluation_count += 1;
            if (condition_met) try self.processConditionMet(rule) else try self.processConditionNotMet(rule);
        }
    }

    pub fn evaluateRule(self: *AlertManager, rule_name: []const u8, value: f64) !bool {
        const rule = self.rules.get(rule_name) orelse return AlertError.RuleNotFound;
        if (!rule.enabled) return false;
        const state = self.rule_state.getPtr(rule_name) orelse return false;
        const condition_met = self.evaluateCondition(rule, value, state);
        state.last_value = value;
        state.evaluation_count += 1;
        if (condition_met) try self.processConditionMet(rule) else try self.processConditionNotMet(rule);
        return condition_met;
    }

    pub fn getAlert(self: *const AlertManager, rule_name: []const u8) ?Alert {
        return self.alerts.get(rule_name);
    }

    pub fn getStats(self: *const AlertManager) AlertStats {
        return self.stats;
    }

    pub fn tick(self: *AlertManager, elapsed_ms: u64) !void {
        self.current_time_ms += elapsed_ms;
        var iter = self.alerts.iterator();
        while (iter.next()) |entry| {
            const alert = entry.value_ptr;
            if (alert.state == .pending) {
                const state = self.rule_state.get(alert.rule_name) orelse continue;
                if (state.condition_met_since == null) {
                    alert.state = .inactive;
                    self.stats.pending_alerts -= 1;
                }
            }
        }
    }

    fn processConditionMet(self: *AlertManager, rule: AlertRule) !void {
        const state = self.rule_state.getPtr(rule.name) orelse return;
        if (state.condition_met_since == null) state.condition_met_since = self.current_time_ms;
        const duration_met = self.current_time_ms - state.condition_met_since.?;
        const required_duration = if (rule.for_duration_ms > 0) rule.for_duration_ms else self.config.default_for_duration_ms;
        const existing = self.alerts.getPtr(rule.name);

        if (existing) |alert| {
            if (alert.state == .pending and duration_met >= required_duration) {
                alert.state = .firing;
                alert.fired_at_ms = self.current_time_ms;
                alert.fire_count += 1;
                alert.current_value = state.last_value;
                self.stats.pending_alerts -= 1;
                self.stats.firing_alerts += 1;
                try self.notify(alert.*);
            } else if (alert.state == .firing) {
                alert.current_value = state.last_value;
            } else if (alert.state == .resolved or alert.state == .inactive) {
                alert.state = .pending;
                alert.started_at_ms = self.current_time_ms;
                alert.current_value = state.last_value;
                alert.resolved_at_ms = null;
                self.stats.pending_alerts += 1;
                if (self.stats.resolved_alerts > 0) self.stats.resolved_alerts -= 1;
            }
        } else {
            const name_copy = try self.allocator.dupe(u8, rule.name);
            errdefer self.allocator.free(name_copy);
            const new_alert = Alert{
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
            } else self.stats.pending_alerts += 1;
        }
    }

    fn processConditionNotMet(self: *AlertManager, rule: AlertRule) !void {
        const state = self.rule_state.getPtr(rule.name) orelse return;
        state.condition_met_since = null;
        if (self.alerts.getPtr(rule.name)) |alert| {
            if (alert.state == .firing) {
                alert.state = .resolved;
                alert.resolved_at_ms = self.current_time_ms;
                self.stats.firing_alerts -= 1;
                self.stats.resolved_alerts += 1;
                try self.notify(alert.*);
            } else if (alert.state == .pending) {
                alert.state = .inactive;
                self.stats.pending_alerts -= 1;
            }
        }
    }

    fn notify(self: *AlertManager, alert: Alert) !void {
        for (self.handlers.items) |*handler| {
            if (alert.severity.toInt() < handler.min_severity.toInt()) continue;
            if (handler.rate_limit_ms > 0) {
                if (self.current_time_ms - handler.last_notified_ms < handler.rate_limit_ms) {
                    self.stats.notifications_suppressed += 1;
                    continue;
                }
            }
            handler.callback(alert, handler.user_data);
            handler.last_notified_ms = self.current_time_ms;
            self.stats.notifications_sent += 1;
        }
    }

    fn evaluateCondition(_: *const AlertManager, rule: AlertRule, value: f64, state: *const RuleState) bool {
        return switch (rule.condition) {
            .rate_of_change => if (state.evaluation_count == 0) false else (value - state.last_value) >= rule.threshold,
            .absent => false,
            else => rule.condition.evaluate(value, rule.threshold),
        };
    }
};

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

pub const AlertRuleBuilder = struct {
    rule: AlertRule,
    pub fn init(name: []const u8, metric: []const u8) AlertRuleBuilder {
        return .{ .rule = .{ .name = name, .metric = metric, .threshold = 0 } };
    }
    pub fn threshold(self: *AlertRuleBuilder, v: f64) *AlertRuleBuilder {
        self.rule.threshold = v;
        return self;
    }
    pub fn condition(self: *AlertRuleBuilder, c: AlertCondition) *AlertRuleBuilder {
        self.rule.condition = c;
        return self;
    }
    pub fn severity(self: *AlertRuleBuilder, s: AlertSeverity) *AlertRuleBuilder {
        self.rule.severity = s;
        return self;
    }
    pub fn forDuration(self: *AlertRuleBuilder, d: u64) *AlertRuleBuilder {
        self.rule.for_duration_ms = d;
        return self;
    }
    pub fn description(self: *AlertRuleBuilder, desc: []const u8) *AlertRuleBuilder {
        self.rule.description = desc;
        return self;
    }
    pub fn build(self: *const AlertRuleBuilder) AlertRule {
        return self.rule;
    }
};

pub fn createRule(name: []const u8, metric: []const u8) AlertRuleBuilder {
    return AlertRuleBuilder.init(name, metric);
}

// ============================================================================
// Prometheus Exporter (from prometheus.zig)
// ============================================================================

pub const PrometheusConfig = struct {
    enabled: bool = true,
    path: []const u8 = "/metrics",
    port: u16 = 9090,
    namespace: []const u8 = "abi",
    include_timestamp: bool = false,
};

pub const PrometheusExporter = struct {
    allocator: std.mem.Allocator,
    config: PrometheusConfig,
    metrics: *observability.MetricsCollector,
    running: std.atomic.Value(bool),
    server_thread: ?std.Thread = null,

    pub fn init(allocator: std.mem.Allocator, config: PrometheusConfig, metrics: *observability.MetricsCollector) !PrometheusExporter {
        return .{
            .allocator = allocator,
            .config = config,
            .metrics = metrics,
            .running = std.atomic.Value(bool).init(false),
            .server_thread = null,
        };
    }

    pub fn deinit(self: *PrometheusExporter) void {
        self.stop();
        self.* = undefined;
    }

    pub fn start(self: *PrometheusExporter) !void {
        if (self.running.load(.acquire)) return;
        self.running.store(true, .release);
        // In a real implementation, this would spawn a thread for the HTTP server
    }

    pub fn stop(self: *PrometheusExporter) void {
        if (!self.running.load(.acquire)) return;
        self.running.store(false, .release);
        if (self.server_thread) |t| {
            t.join();
            self.server_thread = null;
        }
    }

    pub fn generateMetrics(_: *PrometheusExporter, allocator: std.mem.Allocator) ![]const u8 {
        var output = std.ArrayListUnmanaged(u8).empty;
        errdefer output.deinit(allocator);

        try output.appendSlice(allocator, "# HELP abi_build_info Build information\n");
        try output.appendSlice(allocator, "# TYPE abi_build_info gauge\n");
        try output.appendSlice(allocator, "abi_build_info{version=\"0.3.0\",commit=\"unknown\"} 1\n\n");

        return output.toOwnedSlice(allocator);
    }
};

pub fn generateMetricsOutput(
    allocator: std.mem.Allocator,
    namespace: []const u8,
    counters: []const struct { name: []const u8, value: u64 },
    histograms: []const struct { name: []const u8, sum: u64, count: u64, buckets: []const u64, bounds: []const u64 },
) ![]const u8 {
    var output = std.ArrayListUnmanaged(u8).empty;
    errdefer output.deinit(allocator);

    try output.appendSlice(allocator, "# Prometheus metrics exported by ABI framework\n\n");

    for (counters) |counter| {
        try std.fmt.format(output.writer(allocator), "{s}_{s} {d}\n", .{ namespace, counter.name, counter.value });
    }

    for (histograms) |hist| {
        try std.fmt.format(output.writer(allocator), "{s}_{s}_sum {d}\n", .{ namespace, hist.name, hist.sum });
        try std.fmt.format(output.writer(allocator), "{s}_{s}_count {d}\n", .{ namespace, hist.name, hist.count });
        for (hist.buckets, hist.bounds) |bucket_count, bound| {
            try std.fmt.format(output.writer(allocator), "{s}_{s}_bucket{{le=\"{d}\"}} {d}\n", .{ namespace, hist.name, bound, bucket_count });
        }
        try output.append(allocator, '\n');
    }

    return output.toOwnedSlice(allocator);
}

// ============================================================================
// StatsD Client (from statsd.zig)
// ============================================================================

pub const StatsDError = error{
    ConnectionFailed,
    SendFailed,
    InvalidMetricName,
};

pub const StatsDConfig = struct {
    host: []const u8 = "localhost",
    port: u16 = 8125,
    prefix: []const u8 = "",
    sample_rate: f64 = 1.0,
    max_packet_size: usize = 1400,
    buffer_size: usize = 65536,
    flush_interval_ms: u32 = 1000,
};

pub const StatsDClient = struct {
    allocator: std.mem.Allocator,
    config: StatsDConfig,
    socket: ?std.net.Stream = null,
    buffer: std.ArrayListUnmanaged(u8),
    connected: bool,

    pub fn init(allocator: std.mem.Allocator, config: StatsDConfig) !StatsDClient {
        return .{
            .allocator = allocator,
            .config = config,
            .socket = null,
            .buffer = .{},
            .connected = false,
        };
    }

    pub fn deinit(self: *StatsDClient) void {
        self.disconnect();
        self.buffer.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn connect(self: *StatsDClient) !void {
        if (self.connected) return;
        const address = try std.net.Address.parseIp4(self.config.host, self.config.port);
        self.socket = std.net.tcpConnectToAddress(address) catch return StatsDError.ConnectionFailed;
        self.connected = true;
    }

    pub fn disconnect(self: *StatsDClient) void {
        if (self.socket) |*socket| {
            socket.close();
            self.socket = null;
        }
        self.connected = false;
    }

    pub fn increment(self: *StatsDClient, name: []const u8, value: f64, tags: []const []const u8) !void {
        try self.send(name, value, "c", tags);
    }

    fn send(self: *StatsDClient, name: []const u8, value: f64, type_: []const u8, tags: []const []const u8) !void {
        _ = self;
        _ = name;
        _ = value;
        _ = type_;
        _ = tags;
    }
};

pub fn createClient(allocator: std.mem.Allocator, host: []const u8, port: u16) !StatsDClient {
    return StatsDClient.init(allocator, .{ .host = host, .port = port });
}
