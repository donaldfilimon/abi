//! Feedback Analyzer
//!
//! Analyzes collected feedback to derive actionable insights:
//! - Aggregate statistics per persona
//! - Satisfaction scoring
//! - Trend detection (improving/declining)
//! - Category-level breakdown

const std = @import("std");
const collector_mod = @import("collector.zig");

/// Satisfaction trend direction.
pub const Trend = enum {
    improving,
    stable,
    declining,
    insufficient_data,
};

/// Aggregate statistics for a persona.
pub const PersonaStats = struct {
    persona: collector_mod.PersonaRef,
    total_entries: usize,
    average_rating: f32,
    positive_ratio: f32,
    trend: Trend,
    category_breakdown: [8]CategoryStat,
};

/// Per-category statistics.
pub const CategoryStat = struct {
    category: collector_mod.FeedbackCategory,
    count: usize,
    average_rating: f32,
};

/// Overall system satisfaction report.
pub const SatisfactionReport = struct {
    /// Overall average rating across all personas.
    overall_average: f32,
    /// Total feedback entries analyzed.
    total_entries: usize,
    /// Per-persona statistics.
    persona_stats: [5]?PersonaStats,
    /// System-wide satisfaction trend.
    overall_trend: Trend,
    /// Timestamp when this report was generated.
    generated_at: i64,
};

/// Feedback analyzer engine.
pub const FeedbackAnalyzer = struct {
    min_threshold: u32,

    const Self = @This();

    pub fn init(min_analysis_threshold: u32) Self {
        return .{ .min_threshold = min_analysis_threshold };
    }

    /// Analyze feedback entries and produce a persona stats report.
    pub fn analyzePersona(self: *const Self, entries: []const collector_mod.FeedbackEntry, persona: collector_mod.PersonaRef) PersonaStats {
        var total: f32 = 0;
        var count: usize = 0;
        var positive: usize = 0;

        // Category counts and totals
        var cat_counts: [8]usize = .{0} ** 8;
        var cat_totals: [8]f32 = .{0} ** 8;

        for (entries) |entry| {
            if (entry.persona == persona) {
                total += @floatFromInt(entry.rating);
                count += 1;
                if (entry.isPositive()) positive += 1;

                const cat_idx = @intFromEnum(entry.category);
                cat_counts[cat_idx] += 1;
                cat_totals[cat_idx] += @floatFromInt(entry.rating);
            }
        }

        const avg = if (count > 0) total / @as(f32, @floatFromInt(count)) else 0.0;
        const pos_ratio = if (count > 0) @as(f32, @floatFromInt(positive)) / @as(f32, @floatFromInt(count)) else 0.0;

        // Compute trend from recent vs older entries
        const trend = self.computeTrend(entries, persona);

        // Build category breakdown
        var cats: [8]CategoryStat = undefined;
        inline for (0..8) |i| {
            const cat: collector_mod.FeedbackCategory = @enumFromInt(i);
            cats[i] = .{
                .category = cat,
                .count = cat_counts[i],
                .average_rating = if (cat_counts[i] > 0) cat_totals[i] / @as(f32, @floatFromInt(cat_counts[i])) else 0.0,
            };
        }

        return .{
            .persona = persona,
            .total_entries = count,
            .average_rating = avg,
            .positive_ratio = pos_ratio,
            .trend = trend,
            .category_breakdown = cats,
        };
    }

    /// Generate a full satisfaction report across all personas.
    pub fn generateReport(self: *const Self, entries: []const collector_mod.FeedbackEntry) SatisfactionReport {
        const personas = [_]collector_mod.PersonaRef{ .abbey, .aviva, .abi, .ralph, .other };
        var persona_stats: [5]?PersonaStats = .{null} ** 5;
        var overall_total: f32 = 0;
        var overall_count: usize = 0;

        for (personas, 0..) |persona, i| {
            const stats = self.analyzePersona(entries, persona);
            if (stats.total_entries > 0) {
                persona_stats[i] = stats;
                overall_total += stats.average_rating * @as(f32, @floatFromInt(stats.total_entries));
                overall_count += stats.total_entries;
            }
        }

        return .{
            .overall_average = if (overall_count > 0) overall_total / @as(f32, @floatFromInt(overall_count)) else 0.0,
            .total_entries = overall_count,
            .persona_stats = persona_stats,
            .overall_trend = if (overall_count < self.min_threshold) .insufficient_data else .stable,
            .generated_at = std.time.timestamp(),
        };
    }

    fn computeTrend(self: *const Self, entries: []const collector_mod.FeedbackEntry, persona: collector_mod.PersonaRef) Trend {
        // Split relevant entries into halves and compare averages
        var relevant_count: usize = 0;
        for (entries) |e| {
            if (e.persona == persona) relevant_count += 1;
        }

        if (relevant_count < self.min_threshold) return .insufficient_data;

        const midpoint = relevant_count / 2;
        var first_half_total: f32 = 0;
        var first_half_count: f32 = 0;
        var second_half_total: f32 = 0;
        var second_half_count: f32 = 0;
        var seen: usize = 0;

        for (entries) |e| {
            if (e.persona == persona) {
                if (seen < midpoint) {
                    first_half_total += @floatFromInt(e.rating);
                    first_half_count += 1;
                } else {
                    second_half_total += @floatFromInt(e.rating);
                    second_half_count += 1;
                }
                seen += 1;
            }
        }

        if (first_half_count == 0 or second_half_count == 0) return .insufficient_data;

        const first_avg = first_half_total / first_half_count;
        const second_avg = second_half_total / second_half_count;
        const diff = second_avg - first_avg;

        if (diff > 0.3) return .improving;
        if (diff < -0.3) return .declining;
        return .stable;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "FeedbackAnalyzer persona stats" {
    const analyzer = FeedbackAnalyzer.init(2);

    var entries: [4]collector_mod.FeedbackEntry = undefined;
    for (&entries, 0..) |*e, i| {
        e.* = .{
            .id = i + 1,
            .rating = if (i < 2) 5 else 3,
            .rating_type = .stars,
            .category = .quality,
            .persona = .abbey,
            .session_id = undefined,
            .session_id_len = 0,
            .text = undefined,
            .text_len = 0,
            .timestamp = @intCast(i),
        };
    }

    const stats = analyzer.analyzePersona(&entries, .abbey);
    try std.testing.expect(stats.total_entries == 4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), stats.average_rating, 0.01);
}

test "FeedbackAnalyzer report" {
    const analyzer = FeedbackAnalyzer.init(1);

    var entries: [2]collector_mod.FeedbackEntry = undefined;
    entries[0] = .{
        .id = 1,
        .rating = 5,
        .rating_type = .stars,
        .category = .helpfulness,
        .persona = .abbey,
        .session_id = undefined,
        .session_id_len = 0,
        .text = undefined,
        .text_len = 0,
        .timestamp = 100,
    };
    entries[1] = .{
        .id = 2,
        .rating = 4,
        .rating_type = .stars,
        .category = .accuracy,
        .persona = .aviva,
        .session_id = undefined,
        .session_id_len = 0,
        .text = undefined,
        .text_len = 0,
        .timestamp = 200,
    };

    const report = analyzer.generateReport(&entries);
    try std.testing.expect(report.total_entries == 2);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), report.overall_average, 0.01);
}

test {
    std.testing.refAllDecls(@This());
}
