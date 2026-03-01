//! Feedback Storage
//!
//! Provides persistence and retrieval for feedback data:
//! - In-memory ring buffer (wraps FeedbackCollector)
//! - Query by persona/session/time range
//! - Export to structured format

const std = @import("std");
const collector_mod = @import("collector.zig");
const cfg = @import("config.zig");

/// Query filter for retrieving feedback.
pub const FeedbackQuery = struct {
    /// Filter by persona (null = all).
    persona: ?collector_mod.PersonaRef = null,
    /// Filter by category (null = all).
    category: ?collector_mod.FeedbackCategory = null,
    /// Filter by minimum rating.
    min_rating: ?u8 = null,
    /// Filter by maximum rating.
    max_rating: ?u8 = null,
    /// Maximum number of results.
    limit: usize = 100,
};

/// Exported feedback summary in a structured format.
pub const FeedbackExport = struct {
    total_count: usize,
    average_rating: f32,
    positive_count: usize,
    negative_count: usize,
    entries_exported: usize,
};

/// Feedback storage with query capabilities.
pub const FeedbackStorage = struct {
    collector: *collector_mod.FeedbackCollector,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, feedback_cfg: cfg.FeedbackConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .collector = try collector_mod.FeedbackCollector.init(allocator, feedback_cfg),
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.collector.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Submit a star rating.
    pub fn submitStars(
        self: *Self,
        rating: u8,
        persona: collector_mod.PersonaRef,
        category: collector_mod.FeedbackCategory,
        session_id: []const u8,
        text: ?[]const u8,
    ) u64 {
        return self.collector.submitStarRating(rating, persona, category, session_id, text);
    }

    /// Submit a thumbs up/down.
    pub fn submitThumbs(
        self: *Self,
        thumbs_up: bool,
        persona: collector_mod.PersonaRef,
        category: collector_mod.FeedbackCategory,
        session_id: []const u8,
        text: ?[]const u8,
    ) u64 {
        return self.collector.submitThumbsRating(thumbs_up, persona, category, session_id, text);
    }

    /// Query feedback matching the given filter.
    pub fn query(self: *Self, q: FeedbackQuery) ![]const collector_mod.FeedbackEntry {
        self.collector.mutex.lock();
        defer self.collector.mutex.unlock();

        var matches = std.ArrayListUnmanaged(collector_mod.FeedbackEntry).empty;
        errdefer matches.deinit(self.allocator);

        const start = if (self.collector.count < self.collector.entries.len) 0 else self.collector.head;
        var i: usize = 0;
        while (i < self.collector.count and matches.items.len < q.limit) : (i += 1) {
            const pos = (start + i) % self.collector.entries.len;
            const entry = self.collector.entries[pos];

            // Apply filters
            if (q.persona) |p| {
                if (entry.persona != p) continue;
            }
            if (q.category) |c| {
                if (entry.category != c) continue;
            }
            if (q.min_rating) |min| {
                if (entry.rating < min) continue;
            }
            if (q.max_rating) |max| {
                if (entry.rating > max) continue;
            }

            try matches.append(self.allocator, entry);
        }

        return matches.toOwnedSlice(self.allocator);
    }

    /// Export a summary of all stored feedback.
    pub fn exportSummary(self: *Self) FeedbackExport {
        self.collector.mutex.lock();
        defer self.collector.mutex.unlock();

        var total_rating: f32 = 0;
        var positive: usize = 0;
        var negative: usize = 0;

        const start = if (self.collector.count < self.collector.entries.len) 0 else self.collector.head;
        var i: usize = 0;
        while (i < self.collector.count) : (i += 1) {
            const pos = (start + i) % self.collector.entries.len;
            const entry = self.collector.entries[pos];
            total_rating += @floatFromInt(entry.rating);
            if (entry.isPositive()) {
                positive += 1;
            } else {
                negative += 1;
            }
        }

        return .{
            .total_count = self.collector.count,
            .average_rating = if (self.collector.count > 0) total_rating / @as(f32, @floatFromInt(self.collector.count)) else 0.0,
            .positive_count = positive,
            .negative_count = negative,
            .entries_exported = self.collector.count,
        };
    }

    /// Get the total entry count.
    pub fn entryCount(self: *Self) usize {
        return self.collector.entryCount();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "FeedbackStorage submit and query" {
    const allocator = std.testing.allocator;
    var storage = try FeedbackStorage.init(allocator, .{});
    defer storage.deinit();

    _ = storage.submitStars(5, .abbey, .quality, "s1", null);
    _ = storage.submitStars(2, .aviva, .accuracy, "s2", null);
    _ = storage.submitStars(4, .abbey, .helpfulness, "s3", null);

    // Query for abbey only
    const abbey_results = try storage.query(.{ .persona = .abbey });
    defer allocator.free(abbey_results);
    try std.testing.expect(abbey_results.len == 2);

    // Query for high ratings
    const high_results = try storage.query(.{ .min_rating = 4 });
    defer allocator.free(high_results);
    try std.testing.expect(high_results.len == 2);
}

test "FeedbackStorage export summary" {
    const allocator = std.testing.allocator;
    var storage = try FeedbackStorage.init(allocator, .{});
    defer storage.deinit();

    _ = storage.submitStars(5, .abbey, .quality, "s1", null);
    _ = storage.submitStars(1, .aviva, .accuracy, "s2", null);

    const summary = storage.exportSummary();
    try std.testing.expect(summary.total_count == 2);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), summary.average_rating, 0.01);
    try std.testing.expect(summary.positive_count == 1);
    try std.testing.expect(summary.negative_count == 1);
}

test {
    std.testing.refAllDecls(@This());
}
