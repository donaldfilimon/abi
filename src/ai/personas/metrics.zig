//! Persona Metrics Module
//!
//! Tracks performance, usage, and quality metrics for the Multi-Persona AI Assistant.
//! Integrates with the core observability framework to provide persona-specific insights.
//!
//! Enhanced Features:
//! - P50, P90, P99 latency percentile tracking
//! - Sliding window metrics for trend analysis
//! - Alert integration for threshold monitoring
//!
//! ## Shared Primitives
//!
//! For new standalone metrics that don't need registry integration, use the
//! centralized primitives from `observability.core_metrics`:
//! - `core_metrics.Counter` - Thread-safe atomic counter
//! - `core_metrics.Gauge` - Thread-safe gauge (i64)
//! - `core_metrics.FloatGauge` - Mutex-protected float gauge
//! - `core_metrics.SlidingWindow` - Timestamp-based sliding window

const std = @import("std");
const types = @import("types.zig");
const obs = @import("../../observability/mod.zig");
const alerts = @import("alerts.zig");

// Shared metrics primitives (for standalone use)
const core_metrics = @import("../../observability/metrics/mod.zig");

/// Collection of metrics for a specific persona.
pub const PersonaMetricSet = struct {
    requests_total: *obs.Counter,
    success_total: *obs.Counter,
    error_total: *obs.Counter,
    latency_ms: *obs.Histogram,
    satisfaction_score: *obs.Histogram,
};

/// Percentile tracking for latency metrics.
pub const LatencyPercentiles = struct {
    /// P50 (median) latency in milliseconds.
    p50: f64 = 0.0,
    /// P90 latency in milliseconds.
    p90: f64 = 0.0,
    /// P95 latency in milliseconds.
    p95: f64 = 0.0,
    /// P99 latency in milliseconds.
    p99: f64 = 0.0,
    /// Maximum observed latency.
    max: f64 = 0.0,
    /// Minimum observed latency.
    min: f64 = std.math.floatMax(f64),
    /// Average latency.
    avg: f64 = 0.0,
    /// Sample count used for calculation.
    sample_count: usize = 0,
};

/// Sliding window for latency samples.
pub const LatencyWindow = struct {
    allocator: std.mem.Allocator,
    samples: std.ArrayListUnmanaged(u64),
    max_samples: usize,
    sorted_cache: ?[]u64 = null,
    cache_valid: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, max_samples: usize) Self {
        return .{
            .allocator = allocator,
            .samples = .{},
            .max_samples = max_samples,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.sorted_cache) |cache| {
            self.allocator.free(cache);
        }
        self.samples.deinit(self.allocator);
    }

    pub fn record(self: *Self, latency_ms: u64) !void {
        if (self.samples.items.len >= self.max_samples) {
            _ = self.samples.orderedRemove(0);
        }
        try self.samples.append(self.allocator, latency_ms);
        self.cache_valid = false;
    }

    pub fn getPercentiles(self: *Self) LatencyPercentiles {
        if (self.samples.items.len == 0) return .{};

        // Sort samples for percentile calculation
        if (!self.cache_valid or self.sorted_cache == null) {
            if (self.sorted_cache) |cache| {
                self.allocator.free(cache);
            }
            self.sorted_cache = self.allocator.dupe(u64, self.samples.items) catch return .{};
            std.mem.sort(u64, self.sorted_cache.?, {}, struct {
                fn lessThan(_: void, a: u64, b: u64) bool {
                    return a < b;
                }
            }.lessThan);
            self.cache_valid = true;
        }

        const sorted = self.sorted_cache orelse return .{};
        const n = sorted.len;

        // Calculate percentiles
        const p50_idx = (n * 50) / 100;
        const p90_idx = (n * 90) / 100;
        const p95_idx = (n * 95) / 100;
        const p99_idx = @min((n * 99) / 100, n - 1);

        // Calculate average
        var sum: u64 = 0;
        for (sorted) |s| sum += s;
        const avg = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n));

        return .{
            .p50 = @floatFromInt(sorted[p50_idx]),
            .p90 = @floatFromInt(sorted[p90_idx]),
            .p95 = @floatFromInt(sorted[p95_idx]),
            .p99 = @floatFromInt(sorted[p99_idx]),
            .max = @floatFromInt(sorted[n - 1]),
            .min = @floatFromInt(sorted[0]),
            .avg = avg,
            .sample_count = n,
        };
    }

    pub fn clear(self: *Self) void {
        self.samples.clearRetainingCapacity();
        self.cache_valid = false;
    }
};

/// High-level manager for all persona-related metrics.
pub const PersonaMetrics = struct {
    allocator: std.mem.Allocator,
    collector: *obs.MetricsCollector,
    persona_metrics: std.AutoHashMapUnmanaged(types.PersonaType, PersonaMetricSet),
    /// Latency windows for percentile tracking per persona.
    latency_windows: std.AutoHashMapUnmanaged(types.PersonaType, *LatencyWindow),
    /// Alert manager for threshold monitoring.
    alert_manager: ?*alerts.AlertManager = null,

    const Self = @This();

    /// Default latency buckets for AI response time (in milliseconds).
    const LATENCY_BUCKETS = [_]u64{ 100, 250, 500, 1000, 2000, 5000, 10000, 30000 };
    /// Default buckets for satisfaction scores (0.0 to 1.0).
    const SATISFACTION_BUCKETS = [_]u64{ 10, 25, 50, 75, 90 }; // Scaled by 100
    /// Default window size for percentile calculation.
    const DEFAULT_WINDOW_SIZE: usize = 1000;

    /// Initialize the persona metrics manager.
    pub fn init(allocator: std.mem.Allocator, collector: *obs.MetricsCollector) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .collector = collector,
            .persona_metrics = .{},
            .latency_windows = .{},
        };

        return self;
    }

    /// Initialize with alert manager.
    pub fn initWithAlerts(allocator: std.mem.Allocator, collector: *obs.MetricsCollector, alert_mgr: *alerts.AlertManager) !*Self {
        const self = try init(allocator, collector);
        self.alert_manager = alert_mgr;
        return self;
    }

    /// Shutdown the metrics manager and free resources.
    pub fn deinit(self: *Self) void {
        // Clean up latency windows
        var window_it = self.latency_windows.valueIterator();
        while (window_it.next()) |window| {
            window.*.deinit();
            self.allocator.destroy(window.*);
        }
        self.latency_windows.deinit(self.allocator);
        self.persona_metrics.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Register metrics for a specific persona type.
    pub fn registerPersona(self: *Self, persona_type: types.PersonaType) !void {
        if (self.persona_metrics.contains(persona_type)) return;

        const name = @tagName(persona_type);

        var buf: [128]u8 = undefined;

        const req_name = try std.fmt.bufPrint(&buf, "persona_{s}_requests_total", .{name});
        const requests = try self.collector.registerCounter(req_name);

        const success_name = try std.fmt.bufPrint(&buf, "persona_{s}_success_total", .{name});
        const success = try self.collector.registerCounter(success_name);

        const error_name = try std.fmt.bufPrint(&buf, "persona_{s}_error_total", .{name});
        const errors = try self.collector.registerCounter(error_name);

        const lat_name = try std.fmt.bufPrint(&buf, "persona_{s}_latency_ms", .{name});
        const latency = try self.collector.registerHistogram(lat_name, &LATENCY_BUCKETS);

        const sat_name = try std.fmt.bufPrint(&buf, "persona_{s}_satisfaction_score", .{name});
        const satisfaction = try self.collector.registerHistogram(sat_name, &SATISFACTION_BUCKETS);

        try self.persona_metrics.put(self.allocator, persona_type, .{
            .requests_total = requests,
            .success_total = success,
            .error_total = errors,
            .latency_ms = latency,
            .satisfaction_score = satisfaction,
        });

        // Create latency window for percentile tracking
        const window = try self.allocator.create(LatencyWindow);
        window.* = LatencyWindow.init(self.allocator, DEFAULT_WINDOW_SIZE);
        try self.latency_windows.put(self.allocator, persona_type, window);
    }

    /// Record a request for a persona.
    pub fn recordRequest(self: *Self, persona_type: types.PersonaType) void {
        if (self.persona_metrics.getPtr(persona_type)) |m| {
            m.requests_total.inc(1);
        }
    }

    /// Record a successful response with latency.
    pub fn recordSuccess(self: *Self, persona_type: types.PersonaType, latency_ms: u64) void {
        if (self.persona_metrics.getPtr(persona_type)) |m| {
            m.success_total.inc(1);
            m.latency_ms.record(latency_ms);
        }

        // Record to latency window for percentile tracking
        if (self.latency_windows.getPtr(persona_type)) |window| {
            window.*.record(latency_ms) catch |err| {
                std.log.debug("Failed to record latency to window: {t}", .{err});
            };
        }

        // Notify alert manager of success
        if (self.alert_manager) |alert_mgr| {
            alert_mgr.recordResult(persona_type, true) catch |err| {
                std.log.debug("Failed to record success to alert manager: {t}", .{err});
            };
        }
    }

    /// Record an error for a persona.
    pub fn recordError(self: *Self, persona_type: types.PersonaType) void {
        if (self.persona_metrics.getPtr(persona_type)) |m| {
            m.error_total.inc(1);
        }

        // Notify alert manager of failure
        if (self.alert_manager) |alert_mgr| {
            alert_mgr.recordResult(persona_type, false) catch |err| {
                std.log.debug("Failed to record error to alert manager: {t}", .{err});
            };
        }
    }

    /// Record a user satisfaction score (0.0 to 1.0).
    pub fn recordSatisfaction(self: *Self, persona_type: types.PersonaType, score: f32) void {
        if (self.persona_metrics.getPtr(persona_type)) |m| {
            const scaled: u64 = @intFromFloat(std.math.clamp(score * 100.0, 0.0, 100.0));
            m.satisfaction_score.record(scaled);
        }
    }

    /// Get usage statistics for a persona.
    pub fn getStats(self: *Self, persona_type: types.PersonaType) ?PersonaStats {
        const m = self.persona_metrics.get(persona_type) orelse return null;

        const total = m.requests_total.get();
        const success = m.success_total.get();

        // Get latency percentiles
        var latency_percentiles: ?LatencyPercentiles = null;
        if (self.latency_windows.getPtr(persona_type)) |window| {
            latency_percentiles = window.*.getPercentiles();
        }

        return PersonaStats{
            .total_requests = total,
            .success_rate = if (total > 0) @as(f32, @floatFromInt(success)) / @as(f32, @floatFromInt(total)) else 1.0,
            .error_count = m.error_total.get(),
            .latency = latency_percentiles,
        };
    }

    /// Get latency percentiles for a specific persona.
    pub fn getLatencyPercentiles(self: *Self, persona_type: types.PersonaType) ?LatencyPercentiles {
        if (self.latency_windows.getPtr(persona_type)) |window| {
            return window.*.getPercentiles();
        }
        return null;
    }

    /// Create a metric snapshot for alert evaluation.
    pub fn createSnapshot(self: *Self, persona_type: types.PersonaType) ?alerts.MetricSnapshot {
        const m = self.persona_metrics.get(persona_type) orelse return null;

        const total = m.requests_total.get();
        const success = m.success_total.get();
        const errors = m.error_total.get();

        var latency_p50: f64 = 0.0;
        var latency_p99: f64 = 0.0;

        if (self.latency_windows.getPtr(persona_type)) |window| {
            const percentiles = window.*.getPercentiles();
            latency_p50 = percentiles.p50;
            latency_p99 = percentiles.p99;
        }

        return alerts.MetricSnapshot{
            .persona = persona_type,
            .total_requests = total,
            .success_rate = if (total > 0) @as(f64, @floatFromInt(success)) / @as(f64, @floatFromInt(total)) else 1.0,
            .error_rate = if (total > 0) @as(f64, @floatFromInt(errors)) / @as(f64, @floatFromInt(total)) else 0.0,
            .latency_p50_ms = latency_p50,
            .latency_p99_ms = latency_p99,
        };
    }

    /// Check metrics against alert rules.
    pub fn checkAlerts(self: *Self, persona_type: types.PersonaType) !void {
        if (self.alert_manager) |alert_mgr| {
            if (self.createSnapshot(persona_type)) |snapshot| {
                try alert_mgr.checkMetrics(snapshot);
            }
        }
    }
};

/// Summary statistics for a persona.
pub const PersonaStats = struct {
    total_requests: u64,
    success_rate: f32,
    error_count: u64,
    /// Latency percentiles (if available).
    latency: ?LatencyPercentiles = null,

    /// Format as a human-readable string.
    pub fn format(self: PersonaStats, allocator: std.mem.Allocator) ![]const u8 {
        var buf: std.Io.Writer.Allocating = .init(allocator);
        errdefer buf.deinit();

        try buf.writer.print("Requests: {d}, Success Rate: {d:.1}%, Errors: {d}", .{
            self.total_requests,
            self.success_rate * 100.0,
            self.error_count,
        });

        if (self.latency) |lat| {
            try buf.writer.print(", Latency P50: {d:.0}ms, P99: {d:.0}ms", .{
                lat.p50,
                lat.p99,
            });
        }

        return buf.toOwnedSlice();
    }
};

// Tests

test "latency window initialization" {
    var window = LatencyWindow.init(std.testing.allocator, 100);
    defer window.deinit();

    try std.testing.expectEqual(@as(usize, 100), window.max_samples);
}

test "latency window recording" {
    var window = LatencyWindow.init(std.testing.allocator, 100);
    defer window.deinit();

    try window.record(100);
    try window.record(200);
    try window.record(300);

    try std.testing.expectEqual(@as(usize, 3), window.samples.items.len);
}

test "latency percentile calculation" {
    var window = LatencyWindow.init(std.testing.allocator, 100);
    defer window.deinit();

    // Add samples
    var i: u64 = 1;
    while (i <= 100) : (i += 1) {
        try window.record(i * 10); // 10, 20, 30, ..., 1000
    }

    const percentiles = window.getPercentiles();
    try std.testing.expectEqual(@as(usize, 100), percentiles.sample_count);
    try std.testing.expect(percentiles.p50 > 0);
    try std.testing.expect(percentiles.p99 > percentiles.p50);
    try std.testing.expect(percentiles.max >= percentiles.p99);
}

test "latency window sliding" {
    var window = LatencyWindow.init(std.testing.allocator, 5);
    defer window.deinit();

    // Fill beyond capacity
    try window.record(100);
    try window.record(200);
    try window.record(300);
    try window.record(400);
    try window.record(500);
    try window.record(600); // Should push out 100

    try std.testing.expectEqual(@as(usize, 5), window.samples.items.len);
    try std.testing.expectEqual(@as(u64, 200), window.samples.items[0]);
}
