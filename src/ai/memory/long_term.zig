//! Long-term memory implementation using vector similarity.
//!
//! Stores messages with embeddings and retrieves relevant past context
//! based on semantic similarity to the current query.

const std = @import("std");
const time = @import("../../shared/utils.zig");
const simd = @import("../../shared/simd.zig");
const mod = @import("mod.zig");
const Message = mod.Message;
const MessageRole = mod.MessageRole;
const MemoryStats = mod.MemoryStats;
const MemoryType = @import("manager.zig").MemoryType;

/// Long-term memory configuration.
pub const LongTermConfig = struct {
    /// Maximum stored memories.
    max_memories: usize = 1000,
    /// Embedding dimension.
    embedding_dim: usize = 384,
    /// Retrieval top-k.
    top_k: usize = 5,
    /// Minimum similarity threshold for retrieval.
    min_similarity: f32 = 0.5,
    /// Custom embedding function (optional).
    embed_fn: ?*const fn ([]const u8, std.mem.Allocator) (std.mem.Allocator.Error || error{EmbeddingFailed})![]f32 = null,
    /// Recency half-life in seconds for eviction scoring.
    /// Memories older than this will have their recency factor halved.
    /// Default: 1 hour (3600 seconds).
    recency_half_life_secs: i64 = 3600,
};

/// A memory entry with embedding.
pub const MemoryEntry = struct {
    /// Original message.
    message: Message,
    /// Embedding vector.
    embedding: []f32,
    /// Importance score (0-1).
    importance: f32,
    /// Access count.
    access_count: u32,
    /// Last accessed timestamp.
    last_accessed: i64,

    pub fn deinit(self: *MemoryEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.embedding);
        if (@intFromPtr(self.message.content.ptr) != 0) {
            allocator.free(self.message.content);
        }
        if (self.message.name) |n| allocator.free(n);
        if (self.message.metadata) |m| allocator.free(m);
        self.* = undefined;
    }
};

/// Retrieval result from long-term memory.
pub const RetrievalResult = struct {
    /// Retrieved message.
    message: Message,
    /// Similarity score.
    similarity: f32,
    /// Memory importance.
    importance: f32,
};

/// Long-term memory with vector-based retrieval.
pub const LongTermMemory = struct {
    allocator: std.mem.Allocator,
    config: LongTermConfig,
    memories: std.ArrayListUnmanaged(MemoryEntry),
    total_access: u64,

    /// Initialize long-term memory.
    pub fn init(allocator: std.mem.Allocator, config: LongTermConfig) LongTermMemory {
        return .{
            .allocator = allocator,
            .config = config,
            .memories = .{},
            .total_access = 0,
        };
    }

    /// Deinitialize and free resources.
    pub fn deinit(self: *LongTermMemory) void {
        for (self.memories.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.memories.deinit(self.allocator);
        self.* = undefined;
    }

    /// Store a message with optional pre-computed embedding.
    pub fn store(
        self: *LongTermMemory,
        message: Message,
        embedding: ?[]const f32,
        importance: f32,
    ) !void {
        // Evict if at capacity (remove least important)
        if (self.memories.items.len >= self.config.max_memories) {
            try self.evictLeastImportant();
        }

        // Clone message content
        const cloned_content = try self.allocator.dupe(u8, message.content);
        errdefer self.allocator.free(cloned_content);

        const cloned_name = if (message.name) |n|
            try self.allocator.dupe(u8, n)
        else
            null;
        errdefer if (cloned_name) |n| self.allocator.free(n);

        const cloned_metadata = if (message.metadata) |m|
            try self.allocator.dupe(u8, m)
        else
            null;
        errdefer if (cloned_metadata) |m| self.allocator.free(m);

        // Get or compute embedding
        const emb = if (embedding) |e|
            try self.allocator.dupe(f32, e)
        else
            try self.computeEmbedding(message.content);
        errdefer self.allocator.free(emb);

        const entry = MemoryEntry{
            .message = .{
                .role = message.role,
                .content = cloned_content,
                .name = cloned_name,
                .timestamp = message.timestamp,
                .token_count = message.token_count,
                .metadata = cloned_metadata,
            },
            .embedding = emb,
            .importance = importance,
            .access_count = 0,
            .last_accessed = time.nowSeconds(),
        };

        try self.memories.append(self.allocator, entry);
    }

    /// Retrieve relevant memories based on query.
    pub fn retrieve(
        self: *LongTermMemory,
        query: []const u8,
        top_k: ?usize,
    ) ![]RetrievalResult {
        const k = top_k orelse self.config.top_k;

        if (self.memories.items.len == 0) {
            return &[_]RetrievalResult{};
        }

        // Compute query embedding
        const query_embedding = try self.computeEmbedding(query);
        defer self.allocator.free(query_embedding);

        return self.retrieveByEmbedding(query_embedding, k);
    }

    /// Retrieve using pre-computed query embedding.
    pub fn retrieveByEmbedding(
        self: *LongTermMemory,
        query_embedding: []const f32,
        top_k: usize,
    ) ![]RetrievalResult {
        if (self.memories.items.len == 0) {
            return &[_]RetrievalResult{};
        }

        // Score all memories
        var scored = std.ArrayListUnmanaged(struct { idx: usize, score: f32 }){};
        defer scored.deinit(self.allocator);

        for (self.memories.items, 0..) |entry, idx| {
            const similarity = cosineSimilarity(query_embedding, entry.embedding);
            if (similarity >= self.config.min_similarity) {
                // Combined score: similarity + importance bonus
                const score = similarity * 0.8 + entry.importance * 0.2;
                try scored.append(self.allocator, .{ .idx = idx, .score = score });
            }
        }

        // Sort by score descending
        std.mem.sort(
            @TypeOf(scored.items[0]),
            scored.items,
            {},
            struct {
                fn lessThan(_: void, a: @TypeOf(scored.items[0]), b: @TypeOf(scored.items[0])) bool {
                    return a.score > b.score;
                }
            }.lessThan,
        );

        // Build results
        const result_count = @min(top_k, scored.items.len);
        var results = try self.allocator.alloc(RetrievalResult, result_count);
        errdefer self.allocator.free(results);

        for (scored.items[0..result_count], 0..) |item, i| {
            const entry = &self.memories.items[item.idx];
            entry.access_count += 1;
            entry.last_accessed = time.nowSeconds();
            self.total_access += 1;

            results[i] = .{
                .message = entry.message,
                .similarity = cosineSimilarity(query_embedding, entry.embedding),
                .importance = entry.importance,
            };
        }

        return results;
    }

    /// Update importance of a memory.
    pub fn updateImportance(self: *LongTermMemory, index: usize, importance: f32) void {
        if (index < self.memories.items.len) {
            self.memories.items[index].importance = importance;
        }
    }

    /// Evict least important memory.
    fn evictLeastImportant(self: *LongTermMemory) !void {
        if (self.memories.items.len == 0) return;

        var min_idx: usize = 0;
        var min_score: f32 = std.math.floatMax(f32);

        const now = time.nowSeconds();
        for (self.memories.items, 0..) |entry, idx| {
            // Score based on importance, access frequency, and recency
            const age_seconds: f32 = @floatFromInt(@max(0, now - entry.last_accessed));
            const recency_factor: f32 = 1.0 / (1.0 + (age_seconds / 3600.0));
            const access_factor = @as(f32, @floatFromInt(entry.access_count + 1));
            const score = entry.importance * access_factor * recency_factor;

            if (score < min_score) {
                min_score = score;
                min_idx = idx;
            }
        }

        var removed = self.memories.orderedRemove(min_idx);
        removed.deinit(self.allocator);
    }

    /// Calculate recency factor using exponential decay.
    /// Returns a value between 0 and 1, where 1 means just accessed
    /// and the value halves for each half_life_secs that has elapsed.
    fn calculateRecencyFactor(last_accessed: i64, current_time: i64, half_life_secs: i64) f32 {
        if (half_life_secs <= 0) return 1.0;

        const elapsed = current_time - last_accessed;
        if (elapsed <= 0) return 1.0;

        // Exponential decay: factor = 2^(-elapsed / half_life)
        // Using exp for better numerical stability: 2^x = e^(x * ln(2))
        const ln_2: f32 = 0.693147180559945;
        const decay_exponent = -@as(f32, @floatFromInt(elapsed)) / @as(f32, @floatFromInt(half_life_secs));
        const recency = @exp(decay_exponent * ln_2);

        // Clamp to reasonable bounds
        return @max(0.001, @min(1.0, recency));
    }

    /// Compute embedding for text.
    fn computeEmbedding(self: *LongTermMemory, text: []const u8) ![]f32 {
        if (self.config.embed_fn) |embed_fn| {
            return embed_fn(text, self.allocator);
        }

        // Default: simple hash-based pseudo-embedding
        return try defaultEmbedding(self.allocator, text, self.config.embedding_dim);
    }

    /// Clear all memories.
    pub fn clear(self: *LongTermMemory) void {
        for (self.memories.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.memories.clearRetainingCapacity();
        self.total_access = 0;
    }

    /// Get memory statistics.
    pub fn getStats(self: *const LongTermMemory) MemoryStats {
        return .{
            .message_count = self.memories.items.len,
            .total_tokens = 0, // Not tracked for long-term
            .memory_type = .long_term,
            .capacity = self.config.max_memories,
            .utilization = if (self.config.max_memories > 0)
                @as(f64, @floatFromInt(self.memories.items.len)) /
                    @as(f64, @floatFromInt(self.config.max_memories))
            else
                0,
        };
    }

    /// Get memory count.
    pub fn count(self: *const LongTermMemory) usize {
        return self.memories.items.len;
    }
};

/// Compute cosine similarity between two vectors (SIMD-optimized via shared module).
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    return simd.cosineSimilarity(a, b);
}

/// Default pseudo-embedding based on character hash.
fn defaultEmbedding(
    allocator: std.mem.Allocator,
    text: []const u8,
    dim: usize,
) ![]f32 {
    var embedding = try allocator.alloc(f32, dim);
    @memset(embedding, 0);

    // Simple bag-of-characters with position weighting
    for (text, 0..) |c, i| {
        const idx = @as(usize, c) % dim;
        const pos_weight: f32 = 1.0 / @as(f32, @floatFromInt(i + 1));
        embedding[idx] += pos_weight;
    }

    // Normalize
    var norm: f32 = 0;
    for (embedding) |v| {
        norm += v * v;
    }
    norm = @sqrt(norm);
    if (norm > 0) {
        for (embedding) |*v| {
            v.* /= norm;
        }
    }

    return embedding;
}

test "long-term memory store and retrieve" {
    const allocator = std.testing.allocator;
    var memory = LongTermMemory.init(allocator, .{
        .max_memories = 100,
        .embedding_dim = 64,
        .top_k = 3,
        .min_similarity = 0,
    });
    defer memory.deinit();

    try memory.store(Message.user("Hello world"), null, 0.8);
    try memory.store(Message.user("Goodbye world"), null, 0.5);
    try memory.store(Message.assistant("Hi there!"), null, 0.7);

    try std.testing.expectEqual(@as(usize, 3), memory.count());

    const results = try memory.retrieve("Hello", 2);
    defer allocator.free(results);

    try std.testing.expect(results.len > 0);
}

test "long-term memory eviction" {
    const allocator = std.testing.allocator;
    var memory = LongTermMemory.init(allocator, .{
        .max_memories = 3,
        .embedding_dim = 32,
    });
    defer memory.deinit();

    try memory.store(Message.user("First"), null, 0.3);
    try memory.store(Message.user("Second"), null, 0.8);
    try memory.store(Message.user("Third"), null, 0.5);
    try memory.store(Message.user("Fourth"), null, 0.9);

    // Should have evicted least important
    try std.testing.expectEqual(@as(usize, 3), memory.count());
}

test "cosine similarity" {
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 0.0, 0.0 };
    const c = [_]f32{ 0.0, 1.0, 0.0 };

    // Same vector = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cosineSimilarity(&a, &b), 0.001);

    // Orthogonal = 0.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosineSimilarity(&a, &c), 0.001);
}

test "recency factor calculation" {
    const current_time: i64 = 1000000;
    const half_life: i64 = 3600; // 1 hour

    // Just accessed - should be 1.0
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0),
        LongTermMemory.calculateRecencyFactor(current_time, current_time, half_life),
        0.001,
    );

    // Future timestamp (edge case) - should be 1.0
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0),
        LongTermMemory.calculateRecencyFactor(current_time + 100, current_time, half_life),
        0.001,
    );

    // One half-life elapsed - should be ~0.5
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        LongTermMemory.calculateRecencyFactor(current_time - half_life, current_time, half_life),
        0.001,
    );

    // Two half-lives elapsed - should be ~0.25
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        LongTermMemory.calculateRecencyFactor(current_time - 2 * half_life, current_time, half_life),
        0.001,
    );

    // Very old entry - should be clamped to minimum (0.001)
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.001),
        LongTermMemory.calculateRecencyFactor(0, current_time, half_life),
        0.0001,
    );

    // Zero half-life (disabled) - should return 1.0
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0),
        LongTermMemory.calculateRecencyFactor(current_time - 10000, current_time, 0),
        0.001,
    );
}

test "long-term memory with custom recency half-life" {
    const allocator = std.testing.allocator;
    var memory = LongTermMemory.init(allocator, .{
        .max_memories = 100,
        .embedding_dim = 32,
        .recency_half_life_secs = 7200, // 2 hours
    });
    defer memory.deinit();

    try memory.store(Message.user("Test message"), null, 0.5);
    try std.testing.expectEqual(@as(usize, 1), memory.count());

    // Verify the entry has a valid last_accessed timestamp
    const entry = memory.memories.items[0];
    try std.testing.expect(entry.last_accessed > 0);
}
