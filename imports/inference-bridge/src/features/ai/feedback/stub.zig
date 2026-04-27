//! Feedback Module stub — disabled at compile time.

const std = @import("std");

// ── Top-level types ──────────────────────────────────────────────────────────

/// Configuration for the feedback system.
pub const FeedbackConfig = struct {
    max_entries: u32 = 10_000,
    enable_analysis: bool = true,
    min_analysis_threshold: u32 = 10,
    track_profile_scores: bool = true,
    max_text_length: u32 = 2048,

    pub fn defaults() FeedbackConfig {
        return .{};
    }
};

/// Rating type for feedback.
pub const RatingType = enum { stars, thumbs };

/// Feedback category tags.
pub const FeedbackCategory = enum {
    accuracy,
    helpfulness,
    speed,
    quality,
    tone,
    relevance,
    code_quality,
    general,
};

/// Profile type reference.
pub const ProfileRef = enum { abbey, aviva, abi, ralph, other };

/// A single feedback entry.
pub const FeedbackEntry = struct {
    id: u64,
    rating: u8,
    rating_type: RatingType,
    category: FeedbackCategory,
    profile: ProfileRef,
    session_id: [64]u8,
    session_id_len: u8,
    text: [256]u8,
    text_len: u16,
    timestamp: i64,

    pub fn getSessionId(self: *const FeedbackEntry) []const u8 {
        return self.session_id[0..self.session_id_len];
    }

    pub fn getText(self: *const FeedbackEntry) []const u8 {
        return self.text[0..self.text_len];
    }

    pub fn isPositive(self: *const FeedbackEntry) bool {
        return switch (self.rating_type) {
            .stars => self.rating >= 4,
            .thumbs => self.rating >= 5,
        };
    }
};

/// Feedback collector stub.
pub const FeedbackCollector = struct {
    entries: []FeedbackEntry,
    head: usize,
    count: usize,
    next_id: u64,
    max_text_length: u32,
    mutex: StubMutex,

    const StubMutex = struct {
        pub fn lock(_: *@This()) void {}
        pub fn unlock(_: *@This()) void {}
    };

    pub fn init(_: std.mem.Allocator, _: FeedbackConfig) !*FeedbackCollector {
        return error.FeatureDisabled;
    }

    pub fn deinit(_: *FeedbackCollector, _: std.mem.Allocator) void {}

    pub fn submitStarRating(_: *FeedbackCollector, _: u8, _: ProfileRef, _: FeedbackCategory, _: []const u8, _: ?[]const u8) u64 {
        return 0;
    }

    pub fn submitThumbsRating(_: *FeedbackCollector, _: bool, _: ProfileRef, _: FeedbackCategory, _: []const u8, _: ?[]const u8) u64 {
        return 0;
    }

    pub fn entryCount(_: *FeedbackCollector) usize {
        return 0;
    }

    pub fn getByProfile(_: *FeedbackCollector, _: std.mem.Allocator, _: ProfileRef) ![]const FeedbackEntry {
        return error.FeatureDisabled;
    }

    pub fn averageRating(_: *FeedbackCollector, _: ProfileRef) f32 {
        return 0.0;
    }
};

/// Satisfaction trend direction.
pub const Trend = enum { improving, stable, declining, insufficient_data };

/// Aggregate statistics for a profile.
pub const ProfileStats = struct {
    profile: ProfileRef,
    total_entries: usize,
    average_rating: f32,
    positive_ratio: f32,
    trend: Trend,
    category_breakdown: [8]struct {
        category: FeedbackCategory,
        count: usize,
        average_rating: f32,
    },
};

/// Overall system satisfaction report.
pub const SatisfactionReport = struct {
    overall_average: f32,
    total_entries: usize,
    profile_stats: [5]?ProfileStats,
    overall_trend: Trend,
    generated_at: i64,
};

/// Feedback analyzer stub.
pub const FeedbackAnalyzer = struct {
    min_threshold: u32,

    pub fn init(min_analysis_threshold: u32) FeedbackAnalyzer {
        return .{ .min_threshold = min_analysis_threshold };
    }

    pub fn analyzeProfile(_: *const FeedbackAnalyzer, _: []const FeedbackEntry, profile: ProfileRef) ProfileStats {
        return .{
            .profile = profile,
            .total_entries = 0,
            .average_rating = 0.0,
            .positive_ratio = 0.0,
            .trend = .insufficient_data,
            .category_breakdown = .{.{
                .category = .accuracy,
                .count = 0,
                .average_rating = 0.0,
            }} ** 8,
        };
    }

    pub fn generateReport(_: *const FeedbackAnalyzer, _: []const FeedbackEntry) SatisfactionReport {
        return .{
            .overall_average = 0.0,
            .total_entries = 0,
            .profile_stats = .{null} ** 5,
            .overall_trend = .insufficient_data,
            .generated_at = 0,
        };
    }
};

/// Query filter for retrieving feedback.
pub const FeedbackQuery = struct {
    profile: ?ProfileRef = null,
    category: ?FeedbackCategory = null,
    min_rating: ?u8 = null,
    max_rating: ?u8 = null,
    limit: usize = 100,
};

/// Exported feedback summary.
pub const FeedbackExport = struct {
    total_count: usize,
    average_rating: f32,
    positive_count: usize,
    negative_count: usize,
    entries_exported: usize,
};

/// Feedback storage stub.
pub const FeedbackStorage = struct {
    collector_inst: ?*FeedbackCollector,
    allocator: std.mem.Allocator,

    pub fn init(_: std.mem.Allocator, _: FeedbackConfig) !*FeedbackStorage {
        return error.FeatureDisabled;
    }

    pub fn deinit(_: *FeedbackStorage) void {}

    pub fn submitStars(_: *FeedbackStorage, _: u8, _: ProfileRef, _: FeedbackCategory, _: []const u8, _: ?[]const u8) u64 {
        return 0;
    }

    pub fn submitThumbs(_: *FeedbackStorage, _: bool, _: ProfileRef, _: FeedbackCategory, _: []const u8, _: ?[]const u8) u64 {
        return 0;
    }

    pub fn query(_: *FeedbackStorage, _: FeedbackQuery) ![]const FeedbackEntry {
        return error.FeatureDisabled;
    }

    pub fn exportSummary(_: *FeedbackStorage) FeedbackExport {
        return .{
            .total_count = 0,
            .average_rating = 0.0,
            .positive_count = 0,
            .negative_count = 0,
            .entries_exported = 0,
        };
    }

    pub fn entryCount(_: *FeedbackStorage) usize {
        return 0;
    }
};

// ── Sub-module namespace stubs (match mod.zig's pub imports) ─────────────────

pub const config_mod = struct {
    pub const FeedbackConfig = @import("stub.zig").FeedbackConfig;
};

pub const collector = struct {
    pub const FeedbackCollector = @import("stub.zig").FeedbackCollector;
    pub const FeedbackEntry = @import("stub.zig").FeedbackEntry;
    pub const FeedbackCategory = @import("stub.zig").FeedbackCategory;
    pub const RatingType = @import("stub.zig").RatingType;
    pub const ProfileRef = @import("stub.zig").ProfileRef;
};

pub const analyzer = struct {
    pub const FeedbackAnalyzer = @import("stub.zig").FeedbackAnalyzer;
    pub const ProfileStats = @import("stub.zig").ProfileStats;
    pub const SatisfactionReport = @import("stub.zig").SatisfactionReport;
    pub const Trend = @import("stub.zig").Trend;
};

pub const storage = struct {
    pub const FeedbackStorage = @import("stub.zig").FeedbackStorage;
    pub const FeedbackQuery = @import("stub.zig").FeedbackQuery;
    pub const FeedbackExport = @import("stub.zig").FeedbackExport;
};

// ── FeedbackSystem (matches mod.zig) ─────────────────────────────────────────

pub const FeedbackSystem = struct {
    allocator: std.mem.Allocator,
    store: *FeedbackStorage,
    analyzer_engine: FeedbackAnalyzer,
    cfg: FeedbackConfig,

    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: FeedbackConfig) !*Self {
        return error.FeatureDisabled;
    }

    pub fn deinit(_: *Self) void {}

    pub fn submitStars(_: *Self, _: u8, _: ProfileRef, _: FeedbackCategory, _: []const u8, _: ?[]const u8) u64 {
        return 0;
    }

    pub fn submitThumbs(_: *Self, _: bool, _: ProfileRef, _: FeedbackCategory, _: []const u8, _: ?[]const u8) u64 {
        return 0;
    }

    pub fn query(_: *Self, _: FeedbackQuery) ![]const FeedbackEntry {
        return error.FeatureDisabled;
    }

    pub fn generateReport(_: *Self) !SatisfactionReport {
        return error.FeatureDisabled;
    }

    pub fn analyzeProfile(_: *Self, _: ProfileRef) !ProfileStats {
        return error.FeatureDisabled;
    }

    pub fn entryCount(_: *Self) usize {
        return 0;
    }

    pub fn exportSummary(_: *Self) FeedbackExport {
        return .{
            .total_count = 0,
            .average_rating = 0.0,
            .positive_count = 0,
            .negative_count = 0,
            .entries_exported = 0,
        };
    }
};

test {
    std.testing.refAllDecls(@This());
}
