//! Feedback Module — User feedback collection and analysis for ABI.
//!
//! Provides mechanisms for collecting, storing, and analyzing user feedback
//! on persona interactions. Supports star ratings, thumbs up/down, text
//! feedback with category tagging, and per-persona satisfaction tracking.
//!
//! Integration points:
//! - Post-interaction: `submit()` → record user feedback
//! - Dashboard: `generateReport()` → persona performance summary
//! - Tuning: `analyzePersona()` → per-persona insights for model refinement

const std = @import("std");
pub const config_mod = @import("config.zig");
pub const collector = @import("collector.zig");
pub const analyzer = @import("analyzer.zig");
pub const storage = @import("storage.zig");

// Re-export core types
pub const FeedbackConfig = config_mod.FeedbackConfig;
pub const FeedbackCollector = collector.FeedbackCollector;
pub const FeedbackEntry = collector.FeedbackEntry;
pub const FeedbackCategory = collector.FeedbackCategory;
pub const RatingType = collector.RatingType;
pub const PersonaRef = collector.PersonaRef;
pub const FeedbackAnalyzer = analyzer.FeedbackAnalyzer;
pub const PersonaStats = analyzer.PersonaStats;
pub const SatisfactionReport = analyzer.SatisfactionReport;
pub const Trend = analyzer.Trend;
pub const FeedbackStorage = storage.FeedbackStorage;
pub const FeedbackQuery = storage.FeedbackQuery;
pub const FeedbackExport = storage.FeedbackExport;

/// High-level feedback system integrating collection, storage, and analysis.
pub const FeedbackSystem = struct {
    allocator: std.mem.Allocator,
    store: *FeedbackStorage,
    analyzer_engine: FeedbackAnalyzer,
    cfg: FeedbackConfig,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, feedback_cfg: FeedbackConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .store = try FeedbackStorage.init(allocator, feedback_cfg),
            .analyzer_engine = FeedbackAnalyzer.init(feedback_cfg.min_analysis_threshold),
            .cfg = feedback_cfg,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.store.deinit();
        self.allocator.destroy(self);
    }

    /// Submit a star rating (1-5).
    pub fn submitStars(
        self: *Self,
        rating: u8,
        persona: PersonaRef,
        category: FeedbackCategory,
        session_id: []const u8,
        text: ?[]const u8,
    ) u64 {
        return self.store.submitStars(rating, persona, category, session_id, text);
    }

    /// Submit a thumbs up/down rating.
    pub fn submitThumbs(
        self: *Self,
        thumbs_up: bool,
        persona: PersonaRef,
        category: FeedbackCategory,
        session_id: []const u8,
        text: ?[]const u8,
    ) u64 {
        return self.store.submitThumbs(thumbs_up, persona, category, session_id, text);
    }

    /// Query stored feedback.
    pub fn query(self: *Self, q: FeedbackQuery) ![]const FeedbackEntry {
        return self.store.query(q);
    }

    /// Get a satisfaction report across all personas.
    pub fn generateReport(self: *Self) !SatisfactionReport {
        // Get all entries via a broad query
        const entries = try self.store.query(.{ .limit = self.cfg.max_entries });
        defer self.allocator.free(entries);
        return self.analyzer_engine.generateReport(entries);
    }

    /// Get statistics for a specific persona.
    pub fn analyzePersona(self: *Self, persona: PersonaRef) !PersonaStats {
        const entries = try self.store.query(.{ .persona = persona, .limit = self.cfg.max_entries });
        defer self.allocator.free(entries);
        return self.analyzer_engine.analyzePersona(entries, persona);
    }

    /// Get overall entry count.
    pub fn entryCount(self: *Self) usize {
        return self.store.entryCount();
    }

    /// Export a summary.
    pub fn exportSummary(self: *Self) FeedbackExport {
        return self.store.exportSummary();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "FeedbackSystem end-to-end" {
    const allocator = std.testing.allocator;
    var system = try FeedbackSystem.init(allocator, .{ .min_analysis_threshold = 1 });
    defer system.deinit();

    _ = system.submitStars(5, .abbey, .helpfulness, "s1", "Very helpful!");
    _ = system.submitStars(4, .aviva, .accuracy, "s2", null);
    _ = system.submitThumbs(true, .abbey, .quality, "s3", null);

    try std.testing.expect(system.entryCount() == 3);

    const report = try system.generateReport();
    try std.testing.expect(report.total_entries == 3);
    try std.testing.expect(report.overall_average > 0.0);

    const abbey_stats = try system.analyzePersona(.abbey);
    try std.testing.expect(abbey_stats.total_entries == 2);
}

test "FeedbackSystem export" {
    const allocator = std.testing.allocator;
    var system = try FeedbackSystem.init(allocator, .{});
    defer system.deinit();

    _ = system.submitStars(5, .abbey, .quality, "s1", null);
    _ = system.submitStars(1, .aviva, .quality, "s2", null);

    const summary = system.exportSummary();
    try std.testing.expect(summary.total_count == 2);
    try std.testing.expect(summary.positive_count == 1);
}

test {
    std.testing.refAllDecls(@This());
}
