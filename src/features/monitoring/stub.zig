//! Stub for Monitoring feature when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.MonitoringDisabled for all operations.

const std = @import("std");

pub const MonitoringError = error{
    MonitoringDisabled,
};

// Top-level type definitions
pub const AlertManager = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: AlertManagerConfig) @This() {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn addRule(self: *@This(), rule: AlertRule) AlertError!void {
        _ = self;
        _ = rule;
        return error.MonitoringDisabled;
    }

    pub fn removeRule(self: *@This(), name: []const u8) bool {
        _ = self;
        _ = name;
        return false;
    }

    pub fn addHandler(self: *@This(), handler: AlertHandler) AlertError!void {
        _ = self;
        _ = handler;
        return error.MonitoringDisabled;
    }

    pub fn evaluate(self: *@This(), metrics: MetricValues) AlertError!void {
        _ = self;
        _ = metrics;
        return error.MonitoringDisabled;
    }

    pub fn getStats(self: *@This()) AlertStats {
        _ = self;
        return .{};
    }

    pub fn getActiveAlerts(self: *@This(), allocator: std.mem.Allocator) AlertError![]Alert {
        _ = self;
        _ = allocator;
        return error.MonitoringDisabled;
    }
};

pub const AlertManagerConfig = struct {
    evaluation_interval_ms: u64 = 15_000,
    default_for_duration_ms: u64 = 60_000,
    max_alerts: u32 = 1000,
};

pub const AlertRule = struct {
    name: []const u8,
    metric: []const u8,
    condition: AlertCondition = .greater_than,
    threshold: f64 = 0.0,
    severity: AlertSeverity = .warning,
    for_duration_ms: u64 = 0,
    labels: ?std.StringHashMap([]const u8) = null,
    annotations: ?std.StringHashMap([]const u8) = null,
};

pub const AlertRuleBuilder = struct {
    rule: AlertRule,

    pub fn init(name: []const u8) @This() {
        return .{ .rule = .{ .name = name, .metric = "" } };
    }

    pub fn metric(self: *@This(), m: []const u8) *@This() {
        self.rule.metric = m;
        return self;
    }

    pub fn condition(self: *@This(), c: AlertCondition) *@This() {
        self.rule.condition = c;
        return self;
    }

    pub fn threshold(self: *@This(), t: f64) *@This() {
        self.rule.threshold = t;
        return self;
    }

    pub fn severity(self: *@This(), s: AlertSeverity) *@This() {
        self.rule.severity = s;
        return self;
    }

    pub fn forDuration(self: *@This(), ms: u64) *@This() {
        self.rule.for_duration_ms = ms;
        return self;
    }

    pub fn build(self: *@This()) AlertRule {
        return self.rule;
    }
};

pub const Alert = struct {
    rule_name: []const u8 = "",
    state: AlertState = .inactive,
    severity: AlertSeverity = .info,
    fired_at: ?i64 = null,
    resolved_at: ?i64 = null,
    value: f64 = 0.0,
};

pub const AlertState = enum {
    inactive,
    pending,
    firing,
    resolved,
};

pub const AlertSeverity = enum {
    info,
    warning,
    critical,
};

pub const AlertCondition = enum {
    greater_than,
    less_than,
    equal_to,
    not_equal_to,
    greater_than_or_equal,
    less_than_or_equal,
};

pub const AlertError = error{
    MonitoringDisabled,
    RuleNotFound,
    InvalidRule,
    OutOfMemory,
};

pub const AlertStats = struct {
    rules_count: u32 = 0,
    active_alerts: u32 = 0,
    total_evaluations: u64 = 0,
    total_firings: u64 = 0,
};

pub const AlertCallback = *const fn (alert: Alert) void;

pub const AlertHandler = struct {
    callback: AlertCallback,
    min_severity: AlertSeverity = .info,
};

pub const MetricValues = std.StringHashMap(f64);

pub const MetricsCollector = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn registerCounter(self: *@This(), name: []const u8) MonitoringError!*Counter {
        _ = self;
        _ = name;
        return error.MonitoringDisabled;
    }

    pub fn registerHistogram(self: *@This(), name: []const u8, bounds: []const u64) MonitoringError!*Histogram {
        _ = self;
        _ = name;
        _ = bounds;
        return error.MonitoringDisabled;
    }
};

pub const Counter = struct {
    value: u64 = 0,

    pub fn inc(self: *@This(), delta: u64) void {
        _ = self;
        _ = delta;
    }

    pub fn get(self: *const @This()) u64 {
        _ = self;
        return 0;
    }
};

pub const Histogram = struct {
    pub fn record(self: *@This(), value: u64) void {
        _ = self;
        _ = value;
    }
};

pub const DefaultMetrics = struct {
    requests: *Counter,
    errors: *Counter,
    latency_ms: *Histogram,
};

pub const DefaultCollector = struct {
    collector: MetricsCollector,
    defaults: DefaultMetrics,

    pub fn init(allocator: std.mem.Allocator) MonitoringError!@This() {
        _ = allocator;
        return error.MonitoringDisabled;
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

// Top-level function for creating alert rules
pub fn createAlertRule(name: []const u8) AlertRuleBuilder {
    return AlertRuleBuilder.init(name);
}

// Sub-module namespace (alerting)
pub const alerting = struct {
    pub const AlertManager = @import("stub.zig").AlertManager;
    pub const AlertManagerConfig = @import("stub.zig").AlertManagerConfig;
    pub const AlertRule = @import("stub.zig").AlertRule;
    pub const AlertRuleBuilder = @import("stub.zig").AlertRuleBuilder;
    pub const Alert = @import("stub.zig").Alert;
    pub const AlertState = @import("stub.zig").AlertState;
    pub const AlertSeverity = @import("stub.zig").AlertSeverity;
    pub const AlertCondition = @import("stub.zig").AlertCondition;
    pub const AlertError = @import("stub.zig").AlertError;
    pub const AlertStats = @import("stub.zig").AlertStats;
    pub const AlertCallback = @import("stub.zig").AlertCallback;
    pub const AlertHandler = @import("stub.zig").AlertHandler;
    pub const MetricValues = @import("stub.zig").MetricValues;

    pub const createRule = @import("stub.zig").createAlertRule;
};

// Module lifecycle
var initialized: bool = false;

pub fn init(_: std.mem.Allocator) MonitoringError!void {
    return error.MonitoringDisabled;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return initialized;
}

// Convenience functions
pub fn createCollector(allocator: std.mem.Allocator) MetricsCollector {
    return MetricsCollector.init(allocator);
}

pub fn registerDefaultMetrics(collector: *MetricsCollector) MonitoringError!DefaultMetrics {
    _ = collector;
    return error.MonitoringDisabled;
}

pub fn recordRequest(metrics: *DefaultMetrics, latency_ms: u64) void {
    _ = metrics;
    _ = latency_ms;
}

pub fn recordError(metrics: *DefaultMetrics, latency_ms: u64) void {
    _ = metrics;
    _ = latency_ms;
}
