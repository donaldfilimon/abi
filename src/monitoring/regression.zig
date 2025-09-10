//! Performance Regression Detection for WDBX
//!
//! Automated detection of performance regressions including:
//! - Baseline performance tracking
//! - Statistical analysis of performance trends
//! - Automated alerting for performance degradation
//! - Performance anomaly detection
//! - Regression reporting and analysis

const std = @import("std");

/// Regression sensitivity levels
pub const RegressionSensitivity = enum {
    low, // 99% confidence interval
    medium, // 95% confidence interval
    high, // 1 standard deviation
    critical, // 3 standard deviations
};

/// Performance metric for regression analysis
pub const PerformanceMetric = struct {
    name: []const u8,
    value: f64,
    timestamp: i64,
    metadata: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator, name: []const u8, value: f64) !PerformanceMetric {
        return PerformanceMetric{
            .name = try allocator.dupe(u8, name),
            .value = value,
            .timestamp = std.time.timestamp(),
            .metadata = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *PerformanceMetric, allocator: std.mem.Allocator) void {
        allocator.free(self.name);

        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }

    pub fn addMetadata(self: *PerformanceMetric, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
        try self.metadata.put(try allocator.dupe(u8, key), try allocator.dupe(u8, value));
    }
};

/// Statistical performance baseline
pub const PerformanceBaseline = struct {
    metric_name: []const u8,
    mean: f64,
    std_dev: f64,
    min_value: f64,
    max_value: f64,
    sample_count: u64,
    created_at: i64,
    last_updated: i64,

    // Confidence intervals
    confidence_95_lower: f64,
    confidence_95_upper: f64,
    confidence_99_lower: f64,
    confidence_99_upper: f64,

    pub fn init(allocator: std.mem.Allocator, metric_name: []const u8) !PerformanceBaseline {
        return PerformanceBaseline{
            .metric_name = try allocator.dupe(u8, metric_name),
            .mean = 0.0,
            .std_dev = 0.0,
            .min_value = std.math.inf(f64),
            .max_value = -std.math.inf(f64),
            .sample_count = 0,
            .created_at = std.time.timestamp(),
            .last_updated = std.time.timestamp(),
            .confidence_95_lower = 0.0,
            .confidence_95_upper = 0.0,
            .confidence_99_lower = 0.0,
            .confidence_99_upper = 0.0,
        };
    }

    pub fn deinit(self: *PerformanceBaseline, allocator: std.mem.Allocator) void {
        allocator.free(self.metric_name);
    }

    pub fn updateWithValue(self: *PerformanceBaseline, value: f64) void {
        self.sample_count += 1;

        // Update min/max
        self.min_value = @min(self.min_value, value);
        self.max_value = @max(self.max_value, value);

        // Update running mean and standard deviation using Welford's algorithm
        const delta = value - self.mean;
        self.mean += delta / @as(f64, @floatFromInt(self.sample_count));

        if (self.sample_count > 1) {
            const delta2 = value - self.mean;
            self.std_dev = @sqrt(((self.std_dev * self.std_dev * @as(f64, @floatFromInt(self.sample_count - 1))) + (delta * delta2)) / @as(f64, @floatFromInt(self.sample_count)));
        }

        // Update confidence intervals
        self.updateConfidenceIntervals();
        self.last_updated = std.time.timestamp();
    }

    fn updateConfidenceIntervals(self: *PerformanceBaseline) void {
        if (self.sample_count < 2) return;

        const std_error = self.std_dev / @sqrt(@as(f64, @floatFromInt(self.sample_count)));

        // 95% confidence interval (z = 1.96)
        self.confidence_95_lower = self.mean - (1.96 * std_error);
        self.confidence_95_upper = self.mean + (1.96 * std_error);

        // 99% confidence interval (z = 2.576)
        self.confidence_99_lower = self.mean - (2.576 * std_error);
        self.confidence_99_upper = self.mean + (2.576 * std_error);
    }

    pub fn isRegression(self: *const PerformanceBaseline, value: f64, sensitivity: RegressionSensitivity) bool {
        return switch (sensitivity) {
            .low => value > self.confidence_99_upper or value < self.confidence_99_lower,
            .medium => value > self.confidence_95_upper or value < self.confidence_95_lower,
            .high => value > self.mean + self.std_dev or value < self.mean - self.std_dev,
        };
    }

    pub fn getRegressionSeverity(self: *const PerformanceBaseline, value: f64) RegressionSensitivity {
        const deviation = @abs(value - self.mean) / self.std_dev;

        if (deviation >= 3.0) return .critical;
        if (deviation >= 2.0) return .high;
        if (deviation >= 1.0) return .medium;
        return .low;
    }
};

/// Regression alert information
pub const RegressionAlert = struct {
    metric_name: []const u8,
    current_value: f64,
    baseline_mean: f64,
    deviation_percent: f64,
    severity: RegressionSensitivity,
    timestamp: i64,
    description: []const u8,

    pub fn init(allocator: std.mem.Allocator, metric_name: []const u8, current_value: f64, baseline_mean: f64, severity: RegressionSensitivity) !RegressionAlert {
        const deviation_percent = if (baseline_mean != 0.0)
            ((current_value - baseline_mean) / baseline_mean) * 100.0
        else
            0.0;

        const description = try std.fmt.allocPrint(allocator, "Performance regression detected in {s}: current={d:.2}, baseline={d:.2}, deviation={d:.1}%", .{ metric_name, current_value, baseline_mean, deviation_percent });

        return RegressionAlert{
            .metric_name = try allocator.dupe(u8, metric_name),
            .current_value = current_value,
            .baseline_mean = baseline_mean,
            .deviation_percent = deviation_percent,
            .severity = severity,
            .timestamp = std.time.timestamp(),
            .description = description,
        };
    }

    pub fn deinit(self: *RegressionAlert, allocator: std.mem.Allocator) void {
        allocator.free(self.metric_name);
        allocator.free(self.description);
    }
};

/// Configuration for regression detection
pub const RegressionConfig = struct {
    baseline_window_hours: u32 = 24, // How many hours of data to use for baseline
    min_samples_for_baseline: u32 = 100, // Minimum samples needed to establish baseline
    sensitivity: RegressionSensitivity = .medium,
    enable_auto_baseline_update: bool = true,
    baseline_update_interval_hours: u32 = 6,
    alert_cooldown_minutes: u32 = 30, // Minimum time between alerts for same metric
    max_stored_alerts: usize = 1000,
};

/// Performance regression detector
pub const RegressionDetector = struct {
    allocator: std.mem.Allocator,
    config: RegressionConfig,
    baselines: std.StringHashMap(PerformanceBaseline),
    metrics_history: std.StringHashMap(std.ArrayList(PerformanceMetric)),
    recent_alerts: std.ArrayList(RegressionAlert),
    last_alert_time: std.StringHashMap(i64),

    // Callbacks
    regression_callback: ?*const fn (RegressionAlert) void,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: RegressionConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .config = config,
            .baselines = std.StringHashMap(PerformanceBaseline).init(allocator),
            .metrics_history = std.StringHashMap(std.ArrayList(PerformanceMetric)).init(allocator),
            .recent_alerts = std.ArrayList(RegressionAlert).init(allocator),
            .last_alert_time = std.StringHashMap(i64).init(allocator),
            .regression_callback = null,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        // Clean up baselines
        var baseline_iter = self.baselines.iterator();
        while (baseline_iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.baselines.deinit();

        // Clean up metrics history
        var history_iter = self.metrics_history.iterator();
        while (history_iter.next()) |entry| {
            for (entry.value_ptr.items) |*metric| {
                metric.deinit(self.allocator);
            }
            entry.value_ptr.deinit();
        }
        self.metrics_history.deinit();

        // Clean up alerts
        for (self.recent_alerts.items) |*alert| {
            alert.deinit(self.allocator);
        }
        self.recent_alerts.deinit();

        // Clean up last alert times
        var alert_time_iter = self.last_alert_time.iterator();
        while (alert_time_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.last_alert_time.deinit();

        self.allocator.destroy(self);
    }

    pub fn setRegressionCallback(self: *Self, callback: *const fn (RegressionAlert) void) void {
        self.regression_callback = callback;
    }

    /// Record a performance metric and check for regressions
    pub fn recordMetric(self: *Self, metric: PerformanceMetric) !void {
        // Store metric in history
        const result = try self.metrics_history.getOrPut(metric.name);
        if (!result.found_existing) {
            result.value_ptr.* = std.ArrayList(PerformanceMetric).init(self.allocator);
        }

        try result.value_ptr.append(metric);

        // Clean up old metrics (beyond baseline window)
        const cutoff_time = std.time.timestamp() - (@as(i64, @intCast(self.config.baseline_window_hours)) * 3600);
        while (result.value_ptr.items.len > 0 and result.value_ptr.items[0].timestamp < cutoff_time) {
            var old_metric = result.value_ptr.orderedRemove(0);
            old_metric.deinit(self.allocator);
        }

        // Update or create baseline
        try self.updateBaseline(metric.name, metric.value);

        // Check for regression
        if (self.baselines.get(metric.name)) |baseline| {
            if (baseline.sample_count >= self.config.min_samples_for_baseline) {
                try self.checkForRegression(metric, &baseline);
            }
        }
    }

    fn updateBaseline(self: *Self, metric_name: []const u8, value: f64) !void {
        const result = try self.baselines.getOrPut(metric_name);
        if (!result.found_existing) {
            result.value_ptr.* = try PerformanceBaseline.init(self.allocator, metric_name);
        }

        result.value_ptr.updateWithValue(value);
    }

    fn checkForRegression(self: *Self, metric: PerformanceMetric, baseline: *const PerformanceBaseline) !void {
        if (!baseline.isRegression(metric.value, self.config.sensitivity)) {
            return; // No regression detected
        }

        // Check cooldown period
        const now = std.time.timestamp();
        if (self.last_alert_time.get(metric.name)) |last_time| {
            const cooldown_seconds = @as(i64, @intCast(self.config.alert_cooldown_minutes)) * 60;
            if (now - last_time < cooldown_seconds) {
                return; // Still in cooldown period
            }
        }

        // Create regression alert
        const severity = baseline.getRegressionSeverity(metric.value);
        const alert = try RegressionAlert.init(self.allocator, metric.name, metric.value, baseline.mean, severity);

        // Store alert
        if (self.recent_alerts.items.len >= self.config.max_stored_alerts) {
            var old_alert = self.recent_alerts.orderedRemove(0);
            old_alert.deinit(self.allocator);
        }
        try self.recent_alerts.append(alert);

        // Update last alert time
        const key = try self.allocator.dupe(u8, metric.name);
        try self.last_alert_time.put(key, now);

        // Trigger callback
        if (self.regression_callback) |callback| {
            callback(alert);
        }

        std.debug.print("REGRESSION ALERT: {s}\n", .{alert.description});
    }

    /// Get baseline for a metric
    pub fn getBaseline(self: *Self, metric_name: []const u8) ?*const PerformanceBaseline {
        return self.baselines.getPtr(metric_name);
    }

    /// Get recent alerts
    pub fn getRecentAlerts(self: *Self, limit: ?usize) []const RegressionAlert {
        const actual_limit = limit orelse self.recent_alerts.items.len;
        const start_idx = if (self.recent_alerts.items.len > actual_limit)
            self.recent_alerts.items.len - actual_limit
        else
            0;
        return self.recent_alerts.items[start_idx..];
    }

    /// Export regression detection statistics
    pub fn exportStats(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
        var baseline_count: u32 = 0;
        var active_metrics: u32 = 0;
        var total_samples: u64 = 0;

        var baseline_iter = self.baselines.iterator();
        while (baseline_iter.next()) |entry| {
            baseline_count += 1;
            total_samples += entry.value_ptr.sample_count;
            if (entry.value_ptr.sample_count >= self.config.min_samples_for_baseline) {
                active_metrics += 1;
            }
        }

        const recent_alerts_count = self.recent_alerts.items.len;

        return try std.fmt.allocPrint(allocator,
            \\{{"regression_detection":{{"baselines_established":{d},"active_metrics":{d},"total_samples":{d},"recent_alerts":{d},"config":{{"sensitivity":"{s}","baseline_window_hours":{d},"min_samples":{d},"auto_update":{any},"cooldown_minutes":{d}}}}}}}
        , .{ baseline_count, active_metrics, total_samples, recent_alerts_count, @tagName(self.config.sensitivity), self.config.baseline_window_hours, self.config.min_samples_for_baseline, self.config.enable_auto_baseline_update, self.config.alert_cooldown_minutes });
    }

    /// Force update all baselines (useful for testing)
    pub fn updateAllBaselines(self: *Self) !void {
        var history_iter = self.metrics_history.iterator();
        while (history_iter.next()) |entry| {
            const metric_name = entry.key_ptr.*;
            const metrics = entry.value_ptr.items;

            if (metrics.len == 0) continue;

            // Recreate baseline from scratch
            var baseline = try PerformanceBaseline.init(self.allocator, metric_name);
            for (metrics) |metric| {
                baseline.updateWithValue(metric.value);
            }

            // Replace existing baseline
            if (self.baselines.getPtr(metric_name)) |existing| {
                existing.deinit(self.allocator);
            }
            try self.baselines.put(metric_name, baseline);
        }
    }
};
