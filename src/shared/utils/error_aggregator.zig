//! Error Aggregation and Analysis for Debugging.
//!
//! Provides centralized error collection, pattern detection, and analysis
//! for debugging distributed systems. Features include:
//! - Windowed error collection with configurable retention
//! - Error pattern detection (bursts, recurring errors)
//! - Error categorization and grouping
//! - Correlation ID tracking for distributed tracing
//! - Structured error reports for debugging

const std = @import("std");
const time = @import("time.zig");
const errors = @import("errors.zig");

/// Error severity levels.
pub const Severity = enum(u8) {
    debug = 0,
    info = 1,
    warning = 2,
    err = 3,
    critical = 4,

    pub fn format(self: Severity, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        const name = switch (self) {
            .debug => "DEBUG",
            .info => "INFO",
            .warning => "WARNING",
            .err => "ERROR",
            .critical => "CRITICAL",
        };
        try writer.writeAll(name);
    }
};

/// Error source identification.
pub const ErrorSource = struct {
    component: []const u8,
    operation: []const u8,
    node_id: ?[]const u8 = null,
    correlation_id: ?[]const u8 = null,
};

/// Recorded error entry.
pub const ErrorEntry = struct {
    timestamp_ms: i64,
    severity: Severity,
    category: errors.ErrorCategory,
    source: ErrorSource,
    message: []const u8,
    error_code: u32,
    stack_trace: ?[]const u8,
    context: ?[]const u8,
};

/// Error pattern types.
pub const ErrorPattern = enum {
    /// Sudden increase in error rate.
    burst,
    /// Same error recurring frequently.
    recurring,
    /// Error spreading across components.
    cascading,
    /// Errors at specific intervals.
    periodic,
    /// Normal isolated errors.
    isolated,
};

/// Pattern detection result.
pub const PatternResult = struct {
    pattern: ErrorPattern,
    confidence: f64,
    affected_components: []const []const u8,
    first_occurrence_ms: i64,
    last_occurrence_ms: i64,
    occurrence_count: u32,
    sample_message: ?[]const u8,
};

/// Error aggregation statistics.
pub const AggregatorStats = struct {
    total_errors: u64,
    errors_by_severity: [5]u64, // indexed by Severity
    errors_by_category: [6]u64, // indexed by ErrorCategory
    error_rate_per_second: f64,
    peak_error_rate: f64,
    unique_error_codes: u32,
    unique_components: u32,
    patterns_detected: u32,
    oldest_error_ms: i64,
    newest_error_ms: i64,
};

/// Error aggregation configuration.
pub const AggregatorConfig = struct {
    /// Maximum number of errors to retain.
    max_entries: usize = 10000,
    /// Time window for error retention (ms). 0 = no limit.
    retention_window_ms: u64 = 3600_000, // 1 hour
    /// Time window for pattern detection (ms).
    pattern_window_ms: u64 = 60_000, // 1 minute
    /// Minimum errors to trigger burst detection.
    burst_threshold: u32 = 10,
    /// Minimum occurrences for recurring pattern.
    recurring_threshold: u32 = 5,
    /// Enable automatic cleanup of old entries.
    auto_cleanup: bool = true,
    /// Callback for critical errors.
    on_critical: ?*const fn (*const ErrorEntry) void = null,
    /// Callback for pattern detection.
    on_pattern: ?*const fn (PatternResult) void = null,
};

/// Centralized error aggregator for debugging.
pub const ErrorAggregator = struct {
    allocator: std.mem.Allocator,
    config: AggregatorConfig,
    entries: std.ArrayListUnmanaged(ErrorEntry),
    stats: AggregatorStats,
    component_counts: std.StringHashMapUnmanaged(u32),
    error_code_counts: std.AutoHashMapUnmanaged(u32, u32),
    correlation_map: std.StringHashMapUnmanaged(std.ArrayListUnmanaged(usize)),
    mutex: std.Thread.Mutex,

    const Self = @This();

    /// Initialize the error aggregator.
    pub fn init(allocator: std.mem.Allocator, config: AggregatorConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .entries = .{},
            .stats = .{
                .total_errors = 0,
                .errors_by_severity = [_]u64{0} ** 5,
                .errors_by_category = [_]u64{0} ** 6,
                .error_rate_per_second = 0,
                .peak_error_rate = 0,
                .unique_error_codes = 0,
                .unique_components = 0,
                .patterns_detected = 0,
                .oldest_error_ms = 0,
                .newest_error_ms = 0,
            },
            .component_counts = .{},
            .error_code_counts = .{},
            .correlation_map = .{},
            .mutex = .{},
        };
    }

    /// Deinitialize the aggregator.
    pub fn deinit(self: *Self) void {
        // Free all stored strings
        for (self.entries.items) |entry| {
            self.allocator.free(entry.source.component);
            self.allocator.free(entry.source.operation);
            if (entry.source.node_id) |nid| self.allocator.free(nid);
            if (entry.source.correlation_id) |cid| self.allocator.free(cid);
            self.allocator.free(entry.message);
            if (entry.stack_trace) |st| self.allocator.free(st);
            if (entry.context) |ctx| self.allocator.free(ctx);
        }
        self.entries.deinit(self.allocator);

        // Free component keys
        var comp_iter = self.component_counts.keyIterator();
        while (comp_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.component_counts.deinit(self.allocator);

        self.error_code_counts.deinit(self.allocator);

        // Free correlation map
        var corr_iter = self.correlation_map.iterator();
        while (corr_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.correlation_map.deinit(self.allocator);

        self.* = undefined;
    }

    /// Record a new error.
    pub fn recordError(
        self: *Self,
        severity: Severity,
        category: errors.ErrorCategory,
        source: ErrorSource,
        message: []const u8,
        error_code: u32,
    ) !void {
        try self.recordErrorFull(severity, category, source, message, error_code, null, null);
    }

    /// Record an error with full context.
    pub fn recordErrorFull(
        self: *Self,
        severity: Severity,
        category: errors.ErrorCategory,
        source: ErrorSource,
        message: []const u8,
        error_code: u32,
        stack_trace: ?[]const u8,
        context: ?[]const u8,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now_ms = time.nowMilliseconds();

        // Auto cleanup if enabled
        if (self.config.auto_cleanup) {
            self.cleanupOldEntries(now_ms);
        }

        // Check capacity
        if (self.entries.items.len >= self.config.max_entries) {
            self.removeOldestEntry();
        }

        // Clone all strings
        const entry = ErrorEntry{
            .timestamp_ms = now_ms,
            .severity = severity,
            .category = category,
            .source = .{
                .component = try self.allocator.dupe(u8, source.component),
                .operation = try self.allocator.dupe(u8, source.operation),
                .node_id = if (source.node_id) |nid| try self.allocator.dupe(u8, nid) else null,
                .correlation_id = if (source.correlation_id) |cid| try self.allocator.dupe(u8, cid) else null,
            },
            .message = try self.allocator.dupe(u8, message),
            .error_code = error_code,
            .stack_trace = if (stack_trace) |st| try self.allocator.dupe(u8, st) else null,
            .context = if (context) |ctx| try self.allocator.dupe(u8, ctx) else null,
        };

        try self.entries.append(self.allocator, entry);

        // Update statistics
        self.updateStats(entry, now_ms);

        // Track by component
        self.trackComponent(source.component) catch {};

        // Track by error code
        self.trackErrorCode(error_code) catch {};

        // Track by correlation ID
        if (source.correlation_id) |cid| {
            self.trackCorrelation(cid) catch {};
        }

        // Trigger critical callback
        if (severity == .critical) {
            if (self.config.on_critical) |callback| {
                callback(&entry);
            }
        }

        // Check for patterns
        self.detectPatterns(now_ms);
    }

    /// Get current statistics.
    pub fn getStats(self: *Self) AggregatorStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Get errors by correlation ID.
    /// Caller must free the returned slice using the allocator.
    pub fn getByCorrelationId(self: *Self, allocator: std.mem.Allocator, correlation_id: []const u8) ![]const ErrorEntry {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.correlation_map.get(correlation_id)) |indices| {
            var result = std.ArrayListUnmanaged(ErrorEntry){};
            errdefer result.deinit(allocator);
            for (indices.items) |idx| {
                if (idx < self.entries.items.len) {
                    try result.append(allocator, self.entries.items[idx]);
                }
            }
            return result.toOwnedSlice(allocator);
        }
        return &[_]ErrorEntry{};
    }

    /// Get errors by component.
    pub fn getByComponent(self: *Self, component: []const u8) ![]const ErrorEntry {
        self.mutex.lock();
        defer self.mutex.unlock();

        var result = std.ArrayListUnmanaged(ErrorEntry){};
        for (self.entries.items) |entry| {
            if (std.mem.eql(u8, entry.source.component, component)) {
                try result.append(self.allocator, entry);
            }
        }
        return result.toOwnedSlice(self.allocator);
    }

    /// Get errors by severity (and above).
    pub fn getBySeverity(self: *Self, min_severity: Severity) ![]const ErrorEntry {
        self.mutex.lock();
        defer self.mutex.unlock();

        var result = std.ArrayListUnmanaged(ErrorEntry){};
        for (self.entries.items) |entry| {
            if (@intFromEnum(entry.severity) >= @intFromEnum(min_severity)) {
                try result.append(self.allocator, entry);
            }
        }
        return result.toOwnedSlice(self.allocator);
    }

    /// Get recent errors within time window.
    pub fn getRecent(self: *Self, window_ms: u64) ![]const ErrorEntry {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now_ms = time.nowMilliseconds();
        const cutoff_ms = now_ms - @as(i64, @intCast(window_ms));

        var result = std.ArrayListUnmanaged(ErrorEntry){};
        for (self.entries.items) |entry| {
            if (entry.timestamp_ms >= cutoff_ms) {
                try result.append(self.allocator, entry);
            }
        }
        return result.toOwnedSlice(self.allocator);
    }

    /// Generate a debug report.
    pub fn generateReport(self: *Self, allocator: std.mem.Allocator) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var buffer = std.ArrayListUnmanaged(u8){};
        const writer = buffer.writer(allocator);

        try writer.print("=== Error Aggregation Report ===\n\n", .{});
        try writer.print("Total Errors: {d}\n", .{self.stats.total_errors});
        try writer.print("Error Rate: {d:.2}/sec\n", .{self.stats.error_rate_per_second});
        try writer.print("Peak Rate: {d:.2}/sec\n", .{self.stats.peak_error_rate});
        try writer.print("Unique Components: {d}\n", .{self.stats.unique_components});
        try writer.print("Unique Error Codes: {d}\n", .{self.stats.unique_error_codes});
        try writer.print("Patterns Detected: {d}\n\n", .{self.stats.patterns_detected});

        try writer.print("--- Errors by Severity ---\n", .{});
        const severities = [_][]const u8{ "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" };
        for (severities, 0..) |name, i| {
            try writer.print("{s}: {d}\n", .{ name, self.stats.errors_by_severity[i] });
        }

        try writer.print("\n--- Top Components ---\n", .{});
        var comp_iter = self.component_counts.iterator();
        var count: u32 = 0;
        while (comp_iter.next()) |entry| {
            if (count >= 10) break;
            try writer.print("{s}: {d} errors\n", .{ entry.key_ptr.*, entry.value_ptr.* });
            count += 1;
        }

        try writer.print("\n--- Recent Critical Errors ---\n", .{});
        var crit_count: u32 = 0;
        var i = self.entries.items.len;
        while (i > 0 and crit_count < 5) {
            i -= 1;
            const entry = self.entries.items[i];
            if (entry.severity == .critical) {
                try writer.print("[{d}] {s}: {s}\n", .{
                    entry.timestamp_ms,
                    entry.source.component,
                    entry.message,
                });
                crit_count += 1;
            }
        }

        return buffer.toOwnedSlice(allocator);
    }

    /// Clear all stored errors.
    pub fn clear(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Free all stored strings
        for (self.entries.items) |entry| {
            self.allocator.free(entry.source.component);
            self.allocator.free(entry.source.operation);
            if (entry.source.node_id) |nid| self.allocator.free(nid);
            if (entry.source.correlation_id) |cid| self.allocator.free(cid);
            self.allocator.free(entry.message);
            if (entry.stack_trace) |st| self.allocator.free(st);
            if (entry.context) |ctx| self.allocator.free(ctx);
        }
        self.entries.clearRetainingCapacity();

        // Reset stats
        self.stats = .{
            .total_errors = 0,
            .errors_by_severity = [_]u64{0} ** 5,
            .errors_by_category = [_]u64{0} ** 6,
            .error_rate_per_second = 0,
            .peak_error_rate = 0,
            .unique_error_codes = 0,
            .unique_components = 0,
            .patterns_detected = 0,
            .oldest_error_ms = 0,
            .newest_error_ms = 0,
        };
    }

    // Internal helpers

    fn updateStats(self: *Self, entry: ErrorEntry, now_ms: i64) void {
        self.stats.total_errors += 1;
        self.stats.errors_by_severity[@intFromEnum(entry.severity)] += 1;
        self.stats.errors_by_category[@intFromEnum(entry.category)] += 1;
        self.stats.newest_error_ms = now_ms;

        if (self.stats.oldest_error_ms == 0) {
            self.stats.oldest_error_ms = now_ms;
        }

        // Calculate error rate over last minute
        const window_start = now_ms - 60_000;
        var recent_count: u64 = 0;
        for (self.entries.items) |e| {
            if (e.timestamp_ms >= window_start) {
                recent_count += 1;
            }
        }
        self.stats.error_rate_per_second = @as(f64, @floatFromInt(recent_count)) / 60.0;

        if (self.stats.error_rate_per_second > self.stats.peak_error_rate) {
            self.stats.peak_error_rate = self.stats.error_rate_per_second;
        }
    }

    fn trackComponent(self: *Self, component: []const u8) !void {
        const result = self.component_counts.getOrPut(self.allocator, component) catch return;
        if (!result.found_existing) {
            result.key_ptr.* = try self.allocator.dupe(u8, component);
            result.value_ptr.* = 0;
            self.stats.unique_components += 1;
        }
        result.value_ptr.* += 1;
    }

    fn trackErrorCode(self: *Self, error_code: u32) !void {
        const result = try self.error_code_counts.getOrPut(self.allocator, error_code);
        if (!result.found_existing) {
            result.value_ptr.* = 0;
            self.stats.unique_error_codes += 1;
        }
        result.value_ptr.* += 1;
    }

    fn trackCorrelation(self: *Self, correlation_id: []const u8) !void {
        const result = try self.correlation_map.getOrPut(self.allocator, correlation_id);
        if (!result.found_existing) {
            result.key_ptr.* = try self.allocator.dupe(u8, correlation_id);
            result.value_ptr.* = .{};
        }
        try result.value_ptr.append(self.allocator, self.entries.items.len - 1);
    }

    fn cleanupOldEntries(self: *Self, now_ms: i64) void {
        if (self.config.retention_window_ms == 0) return;

        const cutoff_ms = now_ms - @as(i64, @intCast(self.config.retention_window_ms));

        // Find first entry to keep (entries are in chronological order)
        var first_to_keep: usize = 0;
        for (self.entries.items, 0..) |entry, i| {
            if (entry.timestamp_ms >= cutoff_ms) {
                first_to_keep = i;
                break;
            }
            // Free the entry we're discarding
            self.freeEntry(i);
            first_to_keep = i + 1;
        }

        // Shift remaining entries to front
        if (first_to_keep > 0 and first_to_keep <= self.entries.items.len) {
            const remaining = self.entries.items.len - first_to_keep;
            if (remaining > 0) {
                std.mem.copyForwards(
                    ErrorEntry,
                    self.entries.items[0..remaining],
                    self.entries.items[first_to_keep..],
                );
            }
            self.entries.shrinkRetainingCapacity(remaining);
        }

        // Update oldest timestamp
        if (self.entries.items.len > 0) {
            self.stats.oldest_error_ms = self.entries.items[0].timestamp_ms;
        }
    }

    fn removeOldestEntry(self: *Self) void {
        if (self.entries.items.len == 0) return;
        self.freeEntry(0);
        _ = self.entries.orderedRemove(0);
    }

    fn freeEntry(self: *Self, idx: usize) void {
        const entry = self.entries.items[idx];
        self.allocator.free(entry.source.component);
        self.allocator.free(entry.source.operation);
        if (entry.source.node_id) |nid| self.allocator.free(nid);
        if (entry.source.correlation_id) |cid| self.allocator.free(cid);
        self.allocator.free(entry.message);
        if (entry.stack_trace) |st| self.allocator.free(st);
        if (entry.context) |ctx| self.allocator.free(ctx);
    }

    fn detectPatterns(self: *Self, now_ms: i64) void {
        const window_start = now_ms - @as(i64, @intCast(self.config.pattern_window_ms));

        // Count errors in window
        var window_count: u32 = 0;
        for (self.entries.items) |entry| {
            if (entry.timestamp_ms >= window_start) {
                window_count += 1;
            }
        }

        // Detect burst pattern
        if (window_count >= self.config.burst_threshold) {
            self.stats.patterns_detected += 1;
            if (self.config.on_pattern) |callback| {
                callback(.{
                    .pattern = .burst,
                    .confidence = 0.9,
                    .affected_components = &[_][]const u8{},
                    .first_occurrence_ms = window_start,
                    .last_occurrence_ms = now_ms,
                    .occurrence_count = window_count,
                    .sample_message = null,
                });
            }
        }

        // Detect recurring errors by error code
        var code_iter = self.error_code_counts.iterator();
        while (code_iter.next()) |entry| {
            if (entry.value_ptr.* >= self.config.recurring_threshold) {
                self.stats.patterns_detected += 1;
                if (self.config.on_pattern) |callback| {
                    callback(.{
                        .pattern = .recurring,
                        .confidence = 0.8,
                        .affected_components = &[_][]const u8{},
                        .first_occurrence_ms = self.stats.oldest_error_ms,
                        .last_occurrence_ms = now_ms,
                        .occurrence_count = entry.value_ptr.*,
                        .sample_message = null,
                    });
                }
            }
        }
    }
};

/// Global error aggregator singleton.
var global_aggregator: ?*ErrorAggregator = null;
var global_mutex: std.Thread.Mutex = .{};

/// Initialize global error aggregator.
pub fn initGlobal(allocator: std.mem.Allocator, config: AggregatorConfig) !void {
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_aggregator != null) return;

    const agg = try allocator.create(ErrorAggregator);
    agg.* = ErrorAggregator.init(allocator, config);
    global_aggregator = agg;
}

/// Get global error aggregator.
pub fn getGlobal() ?*ErrorAggregator {
    global_mutex.lock();
    defer global_mutex.unlock();
    return global_aggregator;
}

/// Deinitialize global error aggregator.
pub fn deinitGlobal(allocator: std.mem.Allocator) void {
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_aggregator) |agg| {
        agg.deinit();
        allocator.destroy(agg);
        global_aggregator = null;
    }
}

/// Convenience function to record an error globally.
pub fn recordGlobalError(
    severity: Severity,
    category: errors.ErrorCategory,
    source: ErrorSource,
    message: []const u8,
    error_code: u32,
) void {
    if (getGlobal()) |agg| {
        agg.recordError(severity, category, source, message, error_code) catch {};
    }
}

test "error aggregator basic" {
    const allocator = std.testing.allocator;

    var agg = ErrorAggregator.init(allocator, .{});
    defer agg.deinit();

    try agg.recordError(
        .err,
        .network,
        .{ .component = "http_client", .operation = "get" },
        "Connection refused",
        1001,
    );

    try agg.recordError(
        .warning,
        .io,
        .{ .component = "file_reader", .operation = "read" },
        "File not found",
        1002,
    );

    const stats = agg.getStats();
    try std.testing.expectEqual(@as(u64, 2), stats.total_errors);
    try std.testing.expectEqual(@as(u64, 1), stats.errors_by_severity[@intFromEnum(Severity.err)]);
    try std.testing.expectEqual(@as(u64, 1), stats.errors_by_severity[@intFromEnum(Severity.warning)]);
}

test "error aggregator severity filtering" {
    const allocator = std.testing.allocator;

    var agg = ErrorAggregator.init(allocator, .{});
    defer agg.deinit();

    try agg.recordError(.debug, .computation, .{ .component = "test", .operation = "op" }, "debug msg", 1);
    try agg.recordError(.info, .computation, .{ .component = "test", .operation = "op" }, "info msg", 2);
    try agg.recordError(.err, .computation, .{ .component = "test", .operation = "op" }, "error msg", 3);
    try agg.recordError(.critical, .computation, .{ .component = "test", .operation = "op" }, "critical msg", 4);

    const errors_only = try agg.getBySeverity(.err);
    defer allocator.free(errors_only);
    try std.testing.expectEqual(@as(usize, 2), errors_only.len); // err + critical
}

test "error aggregator component tracking" {
    const allocator = std.testing.allocator;

    var agg = ErrorAggregator.init(allocator, .{});
    defer agg.deinit();

    try agg.recordError(.err, .network, .{ .component = "api", .operation = "call" }, "error 1", 1);
    try agg.recordError(.err, .network, .{ .component = "api", .operation = "call" }, "error 2", 2);
    try agg.recordError(.err, .network, .{ .component = "db", .operation = "query" }, "error 3", 3);

    const api_errors = try agg.getByComponent("api");
    defer allocator.free(api_errors);
    try std.testing.expectEqual(@as(usize, 2), api_errors.len);

    const stats = agg.getStats();
    try std.testing.expectEqual(@as(u32, 2), stats.unique_components);
}

test "error aggregator report generation" {
    const allocator = std.testing.allocator;

    var agg = ErrorAggregator.init(allocator, .{});
    defer agg.deinit();

    try agg.recordError(.err, .network, .{ .component = "test", .operation = "op" }, "test error", 1);

    const report = try agg.generateReport(allocator);
    defer allocator.free(report);

    try std.testing.expect(std.mem.indexOf(u8, report, "Error Aggregation Report") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "Total Errors: 1") != null);
}
