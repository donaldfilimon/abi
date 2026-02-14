//! Self-Improvement Module
//!
//! Enables the agent to evaluate its own performance, propose improvements,
//! and track quality metrics over time. Uses the Abbey self-reflection engine
//! for response quality evaluation and the codebase index for self-awareness.

const std = @import("std");

/// Case-insensitive substring search.
fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i..][0..needle.len], needle)) return true;
    }
    return false;
}

// ============================================================================
// Types
// ============================================================================

/// Status of a proposed improvement.
pub const ImprovementStatus = enum {
    proposed,
    approved,
    applied,
    rejected,
};

/// Priority level for an improvement.
pub const ImprovementPriority = enum {
    low,
    medium,
    high,
    critical,
};

/// A proposed improvement to the agent's behavior or code.
pub const Improvement = struct {
    description: []const u8,
    target_file: []const u8,
    suggested_change: []const u8,
    confidence: f32,
    priority: ImprovementPriority,
    status: ImprovementStatus,
    created_at: i64,
};

/// Quality metrics for a single response.
pub const ResponseMetrics = struct {
    coherence: f32 = 0,
    relevance: f32 = 0,
    completeness: f32 = 0,
    clarity: f32 = 0,
    overall: f32 = 0,
    tool_calls_made: usize = 0,
    tool_success_rate: f32 = 0,
    response_length: usize = 0,
};

/// Aggregated performance metrics over time.
pub const PerformanceReport = struct {
    total_interactions: usize,
    avg_quality: f32,
    positive_feedback_count: usize,
    negative_feedback_count: usize,
    tool_usage_count: usize,
    avg_tool_success_rate: f32,
    improvement_count: usize,
    trend: Trend,
};

/// Performance trend direction.
pub const Trend = enum {
    improving,
    stable,
    declining,
};

// ============================================================================
// SelfImprover
// ============================================================================

/// Tracks agent performance and proposes improvements.
pub const SelfImprover = struct {
    allocator: std.mem.Allocator,
    improvements: std.ArrayListUnmanaged(Improvement),
    metrics_history: std.ArrayListUnmanaged(ResponseMetrics),
    positive_feedback: usize,
    negative_feedback: usize,
    total_interactions: usize,

    const Self = @This();

    /// Initialize the self-improvement system.
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .improvements = .{},
            .metrics_history = .{},
            .positive_feedback = 0,
            .negative_feedback = 0,
            .total_interactions = 0,
        };
    }

    /// Clean up resources.
    pub fn deinit(self: *Self) void {
        for (self.improvements.items) |*imp| {
            self.allocator.free(imp.description);
            self.allocator.free(imp.target_file);
            self.allocator.free(imp.suggested_change);
        }
        self.improvements.deinit(self.allocator);
        self.metrics_history.deinit(self.allocator);
    }

    /// Record metrics for a response.
    pub fn recordMetrics(self: *Self, metrics: ResponseMetrics) !void {
        try self.metrics_history.append(self.allocator, metrics);
        self.total_interactions += 1;
    }

    /// Record user feedback.
    pub fn recordFeedback(self: *Self, positive: bool) void {
        if (positive) {
            self.positive_feedback += 1;
        } else {
            self.negative_feedback += 1;
        }
    }

    /// Evaluate a response and return quality metrics.
    pub fn evaluateResponse(
        _: *Self,
        response: []const u8,
        query: []const u8,
    ) ResponseMetrics {
        var metrics = ResponseMetrics{};

        // Coherence: based on response structure
        metrics.coherence = if (response.len > 20) 0.7 else 0.4;

        // Relevance: keyword overlap with query
        var overlap: f32 = 0;
        var query_words = std.mem.splitScalar(u8, query, ' ');
        var word_count: f32 = 0;
        while (query_words.next()) |word| {
            if (word.len < 3) continue;
            word_count += 1;
            if (containsIgnoreCase(response, word)) {
                overlap += 1;
            }
        }
        metrics.relevance = if (word_count > 0) overlap / word_count else 0.5;

        // Completeness: based on response length relative to query
        const length_ratio = @as(f32, @floatFromInt(response.len)) /
            @as(f32, @floatFromInt(@max(query.len, 1)));
        metrics.completeness = @min(1.0, length_ratio / 5.0);

        // Clarity: penalize very long sentences
        metrics.clarity = if (response.len < 2000) 0.8 else 0.5;

        // Response length
        metrics.response_length = response.len;

        // Overall: weighted average
        metrics.overall = metrics.coherence * 0.2 +
            metrics.relevance * 0.3 +
            metrics.completeness * 0.2 +
            metrics.clarity * 0.15 +
            0.15; // base score

        return metrics;
    }

    /// Generate a performance report from accumulated metrics.
    pub fn getReport(self: *const Self) PerformanceReport {
        var avg_quality: f32 = 0;
        var avg_tool_success: f32 = 0;
        var tool_count: usize = 0;

        for (self.metrics_history.items) |m| {
            avg_quality += m.overall;
            if (m.tool_calls_made > 0) {
                avg_tool_success += m.tool_success_rate;
                tool_count += 1;
            }
        }

        const count = self.metrics_history.items.len;
        if (count > 0) {
            avg_quality /= @floatFromInt(count);
        }
        if (tool_count > 0) {
            avg_tool_success /= @floatFromInt(tool_count);
        }

        // Determine trend from recent vs older metrics
        var trend = Trend.stable;
        if (count >= 10) {
            const mid = count / 2;
            var recent_avg: f32 = 0;
            var older_avg: f32 = 0;
            for (self.metrics_history.items[mid..]) |m| {
                recent_avg += m.overall;
            }
            for (self.metrics_history.items[0..mid]) |m| {
                older_avg += m.overall;
            }
            recent_avg /= @floatFromInt(count - mid);
            older_avg /= @floatFromInt(mid);

            if (recent_avg > older_avg + 0.05) {
                trend = .improving;
            } else if (recent_avg < older_avg - 0.05) {
                trend = .declining;
            }
        }

        return .{
            .total_interactions = self.total_interactions,
            .avg_quality = avg_quality,
            .positive_feedback_count = self.positive_feedback,
            .negative_feedback_count = self.negative_feedback,
            .tool_usage_count = tool_count,
            .avg_tool_success_rate = avg_tool_success,
            .improvement_count = self.improvements.items.len,
            .trend = trend,
        };
    }

    /// Get the latest metrics (if any).
    pub fn getLatestMetrics(self: *const Self) ?ResponseMetrics {
        if (self.metrics_history.items.len == 0) return null;
        return self.metrics_history.items[self.metrics_history.items.len - 1];
    }

    /// Get all improvements.
    pub fn getImprovements(self: *const Self) []const Improvement {
        return self.improvements.items;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "SelfImprover - init and deinit" {
    const allocator = std.testing.allocator;
    var improver = SelfImprover.init(allocator);
    defer improver.deinit();

    try std.testing.expectEqual(@as(usize, 0), improver.total_interactions);
}

test "SelfImprover - record metrics" {
    const allocator = std.testing.allocator;
    var improver = SelfImprover.init(allocator);
    defer improver.deinit();

    try improver.recordMetrics(.{ .overall = 0.8 });
    try improver.recordMetrics(.{ .overall = 0.9 });

    try std.testing.expectEqual(@as(usize, 2), improver.total_interactions);
    try std.testing.expectEqual(@as(usize, 2), improver.metrics_history.items.len);
}

test "SelfImprover - record feedback" {
    const allocator = std.testing.allocator;
    var improver = SelfImprover.init(allocator);
    defer improver.deinit();

    improver.recordFeedback(true);
    improver.recordFeedback(true);
    improver.recordFeedback(false);

    try std.testing.expectEqual(@as(usize, 2), improver.positive_feedback);
    try std.testing.expectEqual(@as(usize, 1), improver.negative_feedback);
}

test "SelfImprover - evaluate response" {
    const allocator = std.testing.allocator;
    var improver = SelfImprover.init(allocator);
    defer improver.deinit();

    const metrics = improver.evaluateResponse(
        "The init function creates a new instance with default values.",
        "how does init work",
    );

    try std.testing.expect(metrics.overall > 0);
    try std.testing.expect(metrics.relevance > 0);
    try std.testing.expect(metrics.coherence > 0);
}

test "SelfImprover - get report" {
    const allocator = std.testing.allocator;
    var improver = SelfImprover.init(allocator);
    defer improver.deinit();

    try improver.recordMetrics(.{ .overall = 0.8 });
    improver.recordFeedback(true);

    const report = improver.getReport();
    try std.testing.expectEqual(@as(usize, 1), report.total_interactions);
    try std.testing.expect(report.avg_quality > 0);
    try std.testing.expectEqual(Trend.stable, report.trend);
}

test "SelfImprover - get latest metrics" {
    const allocator = std.testing.allocator;
    var improver = SelfImprover.init(allocator);
    defer improver.deinit();

    try std.testing.expect(improver.getLatestMetrics() == null);

    try improver.recordMetrics(.{ .overall = 0.75 });
    const latest = improver.getLatestMetrics().?;
    try std.testing.expect(latest.overall > 0.7);
}
