//! Abbey Episodic Memory
//!
//! Event-based memory that stores and retrieves experiences.
//! Supports temporal organization, emotional tagging, and consolidation.

const std = @import("std");
const types = @import("../../core/types.zig");
const neural = @import("../neural/mod.zig");
const simd = @import("../../../../services/shared/simd.zig");

// ============================================================================
// Episode Types
// ============================================================================

/// A single episode (a coherent unit of experience)
pub const Episode = struct {
    id: u64,
    start_time: i64,
    end_time: i64,
    messages: std.ArrayListUnmanaged(types.Message),
    summary: ?[]const u8 = null,
    embedding: ?[]f32 = null,
    emotional_arc: EmotionalArc,
    topics: std.ArrayListUnmanaged([]const u8),
    importance: f32 = 0.5,
    access_count: usize = 0,
    last_accessed: i64 = 0,
    consolidated: bool = false,

    pub const EmotionalArc = struct {
        start_emotion: types.EmotionType = .neutral,
        end_emotion: types.EmotionType = .neutral,
        peak_emotion: types.EmotionType = .neutral,
        peak_intensity: f32 = 0.0,
        emotional_trajectory: []types.EmotionType = &.{},
    };

    pub fn deinit(self: *Episode, allocator: std.mem.Allocator) void {
        self.messages.deinit(allocator);
        self.topics.deinit(allocator);
        if (self.summary) |s| allocator.free(s);
        if (self.embedding) |e| allocator.free(e);
    }

    pub fn getDuration(self: *const Episode) i64 {
        return self.end_time - self.start_time;
    }

    pub fn messageCount(self: *const Episode) usize {
        return self.messages.items.len;
    }
};

// ============================================================================
// Episodic Memory Store
// ============================================================================

/// Manages episodic memories
pub const EpisodicMemory = struct {
    allocator: std.mem.Allocator,
    episodes: std.ArrayListUnmanaged(Episode),
    current_episode: ?*Episode = null,
    episode_counter: u64 = 0,
    max_episodes: usize,
    embedding_dim: usize,

    // Indexes for fast retrieval
    time_index: std.AutoHashMapUnmanaged(i64, usize), // timestamp -> episode index
    topic_index: std.StringHashMapUnmanaged(std.ArrayListUnmanaged(usize)), // topic -> episode indices
    emotion_index: std.AutoHashMapUnmanaged(u8, std.ArrayListUnmanaged(usize)), // emotion -> episode indices

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, max_episodes: usize, embedding_dim: usize) Self {
        return Self{
            .allocator = allocator,
            .episodes = .{},
            .max_episodes = max_episodes,
            .embedding_dim = embedding_dim,
            .time_index = .{},
            .topic_index = .{},
            .emotion_index = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.episodes.items) |*ep| {
            ep.deinit(self.allocator);
        }
        self.episodes.deinit(self.allocator);
        self.time_index.deinit(self.allocator);

        var topic_it = self.topic_index.iterator();
        while (topic_it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.topic_index.deinit(self.allocator);

        var emotion_it = self.emotion_index.iterator();
        while (emotion_it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.emotion_index.deinit(self.allocator);
    }

    /// Start a new episode
    pub fn beginEpisode(self: *Self) !*Episode {
        if (self.current_episode != null) {
            try self.endEpisode();
        }

        // Check capacity
        if (self.episodes.items.len >= self.max_episodes) {
            try self.evictOldest();
        }

        self.episode_counter += 1;
        const now = types.getTimestampSec();

        try self.episodes.append(self.allocator, .{
            .id = self.episode_counter,
            .start_time = now,
            .end_time = now,
            .messages = .{},
            .emotional_arc = .{},
            .topics = .{},
        });

        self.current_episode = &self.episodes.items[self.episodes.items.len - 1];
        return self.current_episode.?;
    }

    /// End the current episode
    pub fn endEpisode(self: *Self) !void {
        if (self.current_episode) |ep| {
            ep.end_time = types.getTimestampSec();

            // Update emotional arc
            if (ep.messages.items.len > 0) {
                var peak_intensity: f32 = 0;
                for (ep.messages.items) |msg| {
                    if (msg.metadata) |meta| {
                        if (meta.emotion_detected) |emotion| {
                            if (meta.confidence) |conf| {
                                if (conf.score > peak_intensity) {
                                    peak_intensity = conf.score;
                                    ep.emotional_arc.peak_emotion = emotion;
                                    ep.emotional_arc.peak_intensity = peak_intensity;
                                }
                            }
                        }
                    }
                }
            }

            // Index the episode
            try self.indexEpisode(ep);

            self.current_episode = null;
        }
    }

    /// Add a message to the current episode
    pub fn addMessage(self: *Self, message: types.Message) !void {
        if (self.current_episode == null) {
            _ = try self.beginEpisode();
        }

        try self.current_episode.?.messages.append(self.allocator, message);
        self.current_episode.?.end_time = types.getTimestampSec();
    }

    /// Index an episode for fast retrieval
    fn indexEpisode(self: *Self, episode: *Episode) !void {
        const idx = self.episodes.items.len - 1;

        // Time index
        try self.time_index.put(self.allocator, episode.start_time, idx);

        // Topic index
        for (episode.topics.items) |topic| {
            const result = try self.topic_index.getOrPut(self.allocator, topic);
            if (!result.found_existing) {
                result.value_ptr.* = .{};
            }
            try result.value_ptr.append(self.allocator, idx);
        }

        // Emotion index
        const emotion_key = @intFromEnum(episode.emotional_arc.peak_emotion);
        const emotion_result = try self.emotion_index.getOrPut(self.allocator, emotion_key);
        if (!emotion_result.found_existing) {
            emotion_result.value_ptr.* = .{};
        }
        try emotion_result.value_ptr.append(self.allocator, idx);
    }

    /// Evict the least important episode
    fn evictOldest(self: *Self) !void {
        if (self.episodes.items.len == 0) return;

        // Find least important non-consolidated episode
        var min_importance: f32 = std.math.floatMax(f32);
        var evict_idx: usize = 0;

        for (self.episodes.items, 0..) |ep, i| {
            if (!ep.consolidated and ep.importance < min_importance) {
                min_importance = ep.importance;
                evict_idx = i;
            }
        }

        // Remove from indexes and free
        var ep = self.episodes.orderedRemove(evict_idx);
        ep.deinit(self.allocator);
    }

    // ========================================================================
    // Retrieval Methods
    // ========================================================================

    /// Retrieve episodes by time range
    pub fn getByTimeRange(self: *Self, start: i64, end: i64) ![]const *Episode {
        var results = std.ArrayListUnmanaged(*Episode).empty;
        errdefer results.deinit(self.allocator);

        for (self.episodes.items) |*ep| {
            if (ep.start_time >= start and ep.end_time <= end) {
                try results.append(self.allocator, ep);
                ep.access_count += 1;
                ep.last_accessed = types.getTimestampSec();
            }
        }

        return results.toOwnedSlice(self.allocator);
    }

    /// Retrieve episodes by topic
    pub fn getByTopic(self: *Self, topic: []const u8) ![]const *Episode {
        var results = std.ArrayListUnmanaged(*Episode).empty;
        errdefer results.deinit(self.allocator);

        if (self.topic_index.get(topic)) |indices| {
            for (indices.items) |idx| {
                if (idx < self.episodes.items.len) {
                    var ep = &self.episodes.items[idx];
                    try results.append(self.allocator, ep);
                    ep.access_count += 1;
                    ep.last_accessed = types.getTimestampSec();
                }
            }
        }

        return results.toOwnedSlice(self.allocator);
    }

    /// Retrieve episodes by emotion
    pub fn getByEmotion(self: *Self, emotion: types.EmotionType) ![]const *Episode {
        var results = std.ArrayListUnmanaged(*Episode).empty;
        errdefer results.deinit(self.allocator);

        const emotion_key = @intFromEnum(emotion);
        if (self.emotion_index.get(emotion_key)) |indices| {
            for (indices.items) |idx| {
                if (idx < self.episodes.items.len) {
                    var ep = &self.episodes.items[idx];
                    try results.append(self.allocator, ep);
                    ep.access_count += 1;
                    ep.last_accessed = types.getTimestampSec();
                }
            }
        }

        return results.toOwnedSlice(self.allocator);
    }

    /// Retrieve most recent episodes
    pub fn getRecent(self: *Self, count: usize) []const *Episode {
        const start = if (self.episodes.items.len > count)
            self.episodes.items.len - count
        else
            0;

        var results: [64]*Episode = undefined;
        const actual_count = @min(count, self.episodes.items.len - start);

        for (0..actual_count) |i| {
            results[i] = &self.episodes.items[start + i];
            results[i].access_count += 1;
            results[i].last_accessed = types.getTimestampSec();
        }

        return results[0..actual_count];
    }

    /// Search episodes by embedding similarity
    pub fn searchBySimilarity(
        self: *Self,
        query_embedding: []const f32,
        top_k: usize,
    ) ![]const EpisodeMatch {
        var matches = std.ArrayListUnmanaged(EpisodeMatch).empty;
        errdefer matches.deinit(self.allocator);

        for (self.episodes.items, 0..) |*ep, idx| {
            if (ep.embedding) |emb| {
                const similarity = cosineSimilarity(query_embedding, emb);
                try matches.append(self.allocator, .{
                    .episode_idx = idx,
                    .similarity = similarity,
                });
            }
        }

        // Sort by similarity (descending)
        const items = matches.items;
        std.mem.sort(EpisodeMatch, items, {}, struct {
            fn lessThan(_: void, a: EpisodeMatch, b: EpisodeMatch) bool {
                return a.similarity > b.similarity;
            }
        }.lessThan);

        // Return top_k
        const result_count = @min(top_k, items.len);
        return matches.toOwnedSlice(self.allocator)[0..result_count];
    }

    pub const EpisodeMatch = struct {
        episode_idx: usize,
        similarity: f32,
    };

    // ========================================================================
    // Consolidation
    // ========================================================================

    /// Consolidate old episodes (summarize and compress)
    pub fn consolidate(self: *Self, threshold_hours: u32) !usize {
        const now = types.getTimestampSec();
        const threshold = now - @as(i64, threshold_hours) * 3600;
        var consolidated_count: usize = 0;

        for (self.episodes.items) |*ep| {
            if (!ep.consolidated and ep.end_time < threshold) {
                // Mark as consolidated (in real impl, would compress/summarize)
                ep.consolidated = true;
                consolidated_count += 1;
            }
        }

        return consolidated_count;
    }

    /// Get memory statistics
    pub fn getStats(self: *const Self) EpisodicStats {
        var total_messages: usize = 0;
        var consolidated: usize = 0;

        for (self.episodes.items) |ep| {
            total_messages += ep.messages.items.len;
            if (ep.consolidated) consolidated += 1;
        }

        return .{
            .episode_count = self.episodes.items.len,
            .total_messages = total_messages,
            .consolidated_count = consolidated,
            .has_active_episode = self.current_episode != null,
        };
    }

    pub const EpisodicStats = struct {
        episode_count: usize,
        total_messages: usize,
        consolidated_count: usize,
        has_active_episode: bool,
    };
};

// ============================================================================
// Utility Functions
// ============================================================================

fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    return simd.cosineSimilarity(a, b);
}

// ============================================================================
// Tests
// ============================================================================

test "episodic memory basic" {
    const allocator = std.testing.allocator;

    var memory = EpisodicMemory.init(allocator, 100, 128);
    defer memory.deinit();

    // Begin an episode
    _ = try memory.beginEpisode();

    // Add messages
    try memory.addMessage(types.Message.user("Hello!"));
    try memory.addMessage(types.Message.assistant("Hi there!"));

    // End episode
    try memory.endEpisode();

    const stats = memory.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.episode_count);
    try std.testing.expectEqual(@as(usize, 2), stats.total_messages);
}

test "episodic retrieval" {
    const allocator = std.testing.allocator;

    var memory = EpisodicMemory.init(allocator, 100, 128);
    defer memory.deinit();

    // Create multiple episodes
    for (0..3) |_| {
        _ = try memory.beginEpisode();
        try memory.addMessage(types.Message.user("Test"));
        try memory.endEpisode();
    }

    const recent = memory.getRecent(2);
    try std.testing.expectEqual(@as(usize, 2), recent.len);
}
