//! Feedback Collector
//!
//! Collects user feedback on persona interactions including:
//! - Star ratings (1-5)
//! - Thumbs up/down
//! - Text feedback with optional category tagging
//! - Session and persona association

const std = @import("std");
const cfg = @import("config.zig");

/// Rating type for feedback.
pub const RatingType = enum {
    /// 1-5 star scale.
    stars,
    /// Binary thumbs up/down.
    thumbs,
};

/// Feedback category tags.
pub const FeedbackCategory = enum {
    /// Response accuracy and correctness.
    accuracy,
    /// How helpful the response was.
    helpfulness,
    /// Response speed and latency.
    speed,
    /// Overall response quality.
    quality,
    /// Tone and communication style.
    tone,
    /// Relevance to the query.
    relevance,
    /// Code quality (for code responses).
    code_quality,
    /// General/uncategorized.
    general,
};

/// Persona type reference (mirrors the persona system).
pub const PersonaRef = enum {
    abbey,
    aviva,
    abi,
    ralph,
    other,
};

/// A single feedback entry.
pub const FeedbackEntry = struct {
    /// Monotonic entry ID.
    id: u64,
    /// Star rating (1-5) or thumbs (1=down, 5=up).
    rating: u8,
    /// Type of rating.
    rating_type: RatingType,
    /// Feedback category.
    category: FeedbackCategory,
    /// Associated persona.
    persona: PersonaRef,
    /// Session identifier (fixed buffer).
    session_id: [64]u8,
    session_id_len: u8,
    /// Optional text feedback (fixed buffer).
    text: [256]u8,
    text_len: u16,
    /// Timestamp (Unix seconds).
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

/// Feedback collector with ring-buffer storage.
pub const FeedbackCollector = struct {
    entries: []FeedbackEntry,
    head: usize,
    count: usize,
    next_id: u64,
    max_text_length: u32,
    mutex: std.Thread.Mutex,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, feedback_cfg: cfg.FeedbackConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const cap: usize = if (feedback_cfg.max_entries == 0) 1000 else feedback_cfg.max_entries;
        const entries = try allocator.alloc(FeedbackEntry, cap);

        self.* = .{
            .entries = entries,
            .head = 0,
            .count = 0,
            .next_id = 1,
            .max_text_length = feedback_cfg.max_text_length,
            .mutex = .{},
        };
        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
        allocator.destroy(self);
    }

    /// Submit a star rating (1-5).
    pub fn submitStarRating(
        self: *Self,
        rating: u8,
        persona: PersonaRef,
        category: FeedbackCategory,
        session_id: []const u8,
        text: ?[]const u8,
    ) u64 {
        const clamped_rating = std.math.clamp(rating, 1, 5);
        return self.addEntry(clamped_rating, .stars, persona, category, session_id, text);
    }

    /// Submit a thumbs up/down (true = up, false = down).
    pub fn submitThumbsRating(
        self: *Self,
        thumbs_up: bool,
        persona: PersonaRef,
        category: FeedbackCategory,
        session_id: []const u8,
        text: ?[]const u8,
    ) u64 {
        const rating: u8 = if (thumbs_up) 5 else 1;
        return self.addEntry(rating, .thumbs, persona, category, session_id, text);
    }

    /// Get total number of feedback entries.
    pub fn entryCount(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.count;
    }

    /// Get entries for a specific persona.
    pub fn getByPersona(self: *Self, allocator: std.mem.Allocator, persona: PersonaRef) ![]const FeedbackEntry {
        self.mutex.lock();
        defer self.mutex.unlock();

        var matches = std.ArrayListUnmanaged(FeedbackEntry).empty;
        errdefer matches.deinit(allocator);

        const start = if (self.count < self.entries.len) 0 else self.head;
        var i: usize = 0;
        while (i < self.count) : (i += 1) {
            const pos = (start + i) % self.entries.len;
            if (self.entries[pos].persona == persona) {
                try matches.append(allocator, self.entries[pos]);
            }
        }
        return matches.toOwnedSlice(allocator);
    }

    /// Get the average rating for a persona (returns 0.0 if no entries).
    pub fn averageRating(self: *Self, persona: PersonaRef) f32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var total: f32 = 0;
        var count: f32 = 0;
        const start = if (self.count < self.entries.len) 0 else self.head;
        var i: usize = 0;
        while (i < self.count) : (i += 1) {
            const pos = (start + i) % self.entries.len;
            if (self.entries[pos].persona == persona) {
                total += @floatFromInt(self.entries[pos].rating);
                count += 1;
            }
        }
        return if (count > 0) total / count else 0.0;
    }

    fn addEntry(
        self: *Self,
        rating: u8,
        rating_type: RatingType,
        persona: PersonaRef,
        category: FeedbackCategory,
        session_id: []const u8,
        text: ?[]const u8,
    ) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const id = self.next_id;
        self.next_id += 1;

        var entry = FeedbackEntry{
            .id = id,
            .rating = rating,
            .rating_type = rating_type,
            .category = category,
            .persona = persona,
            .session_id = undefined,
            .session_id_len = 0,
            .text = undefined,
            .text_len = 0,
            .timestamp = std.time.timestamp(),
        };

        const sid_len: u8 = @intCast(@min(session_id.len, 64));
        @memcpy(entry.session_id[0..sid_len], session_id[0..sid_len]);
        entry.session_id_len = sid_len;

        if (text) |t| {
            const max_len = @min(t.len, @min(@as(usize, self.max_text_length), 256));
            const text_len: u16 = @intCast(max_len);
            @memcpy(entry.text[0..text_len], t[0..text_len]);
            entry.text_len = text_len;
        }

        self.entries[self.head] = entry;
        self.head = (self.head + 1) % self.entries.len;
        if (self.count < self.entries.len) self.count += 1;

        return id;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "FeedbackCollector submit and retrieve" {
    const allocator = std.testing.allocator;
    const collector = try FeedbackCollector.init(allocator, .{});
    defer collector.deinit(allocator);

    const id1 = collector.submitStarRating(5, .abbey, .helpfulness, "sess-1", "Great response!");
    const id2 = collector.submitThumbsRating(false, .aviva, .accuracy, "sess-2", null);

    try std.testing.expect(id1 == 1);
    try std.testing.expect(id2 == 2);
    try std.testing.expect(collector.entryCount() == 2);
}

test "FeedbackCollector average rating" {
    const allocator = std.testing.allocator;
    const collector = try FeedbackCollector.init(allocator, .{});
    defer collector.deinit(allocator);

    _ = collector.submitStarRating(5, .abbey, .quality, "s1", null);
    _ = collector.submitStarRating(3, .abbey, .quality, "s2", null);
    _ = collector.submitStarRating(4, .aviva, .quality, "s3", null);

    const abbey_avg = collector.averageRating(.abbey);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), abbey_avg, 0.01);

    const aviva_avg = collector.averageRating(.aviva);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), aviva_avg, 0.01);
}

test "FeedbackCollector getByPersona" {
    const allocator = std.testing.allocator;
    const collector = try FeedbackCollector.init(allocator, .{});
    defer collector.deinit(allocator);

    _ = collector.submitStarRating(5, .abbey, .quality, "s1", null);
    _ = collector.submitStarRating(3, .aviva, .quality, "s2", null);
    _ = collector.submitStarRating(4, .abbey, .quality, "s3", null);

    const abbey_entries = try collector.getByPersona(allocator, .abbey);
    defer allocator.free(abbey_entries);
    try std.testing.expect(abbey_entries.len == 2);
}

test "FeedbackEntry positive check" {
    const entry = FeedbackEntry{
        .id = 1,
        .rating = 5,
        .rating_type = .stars,
        .category = .quality,
        .persona = .abbey,
        .session_id = undefined,
        .session_id_len = 0,
        .text = undefined,
        .text_len = 0,
        .timestamp = 0,
    };
    try std.testing.expect(entry.isPositive());
}

test {
    std.testing.refAllDecls(@This());
}
