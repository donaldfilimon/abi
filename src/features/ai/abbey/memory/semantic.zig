//! Abbey Semantic Memory
//!
//! Long-term knowledge storage with vector-based retrieval.
//! Stores facts, concepts, and learned associations.

const std = @import("std");
const types = @import("../../core/types.zig");
const simd = @import("../../../../services/shared/simd/mod.zig");

// ============================================================================
// Knowledge Types
// ============================================================================

/// A piece of knowledge
pub const Knowledge = struct {
    id: u64,
    content: []const u8,
    category: KnowledgeCategory,
    embedding: ?[]f32 = null,
    source: KnowledgeSource,
    confidence: f32 = 0.5,
    created_at: i64,
    accessed_at: i64,
    access_count: usize = 0,
    associations: std.ArrayListUnmanaged(u64), // IDs of related knowledge

    pub const KnowledgeCategory = enum {
        fact,
        concept,
        procedure,
        preference,
        relationship,
        context,
        skill,
    };

    pub const KnowledgeSource = enum {
        user_stated,
        inferred,
        retrieved,
        system,
        learned,
    };

    pub fn deinit(self: *Knowledge, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
        if (self.embedding) |e| allocator.free(e);
        self.associations.deinit(allocator);
    }
};

// ============================================================================
// Semantic Memory Store
// ============================================================================

/// Manages semantic/knowledge memory
pub const SemanticMemory = struct {
    allocator: std.mem.Allocator,
    knowledge: std.ArrayListUnmanaged(Knowledge),
    knowledge_counter: u64 = 0,
    embedding_dim: usize,

    // Indexes
    category_index: std.AutoHashMapUnmanaged(u8, std.ArrayListUnmanaged(usize)),
    id_to_index: std.AutoHashMapUnmanaged(u64, usize),

    // HNSW-like approximate nearest neighbor index (simplified)
    embedding_index: ?EmbeddingIndex = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, embedding_dim: usize) Self {
        return Self{
            .allocator = allocator,
            .knowledge = .{},
            .embedding_dim = embedding_dim,
            .category_index = .{},
            .id_to_index = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.knowledge.items) |*k| {
            k.deinit(self.allocator);
        }
        self.knowledge.deinit(self.allocator);

        var cat_it = self.category_index.iterator();
        while (cat_it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.category_index.deinit(self.allocator);
        self.id_to_index.deinit(self.allocator);

        if (self.embedding_index) |*idx| {
            idx.deinit();
        }
    }

    /// Store new knowledge
    pub fn store(self: *Self, content: []const u8, category: Knowledge.KnowledgeCategory, source: Knowledge.KnowledgeSource) !u64 {
        self.knowledge_counter += 1;
        const now = types.getTimestampSec();

        const content_copy = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(content_copy);

        const idx = self.knowledge.items.len;

        try self.knowledge.append(self.allocator, .{
            .id = self.knowledge_counter,
            .content = content_copy,
            .category = category,
            .source = source,
            .created_at = now,
            .accessed_at = now,
            .associations = .{},
        });

        // Index by category
        const cat_key = @intFromEnum(category);
        const result = try self.category_index.getOrPut(self.allocator, cat_key);
        if (!result.found_existing) {
            result.value_ptr.* = .{};
        }
        try result.value_ptr.append(self.allocator, idx);

        // Index by ID
        try self.id_to_index.put(self.allocator, self.knowledge_counter, idx);

        return self.knowledge_counter;
    }

    /// Store knowledge with embedding
    pub fn storeWithEmbedding(
        self: *Self,
        content: []const u8,
        embedding: []const f32,
        category: Knowledge.KnowledgeCategory,
        source: Knowledge.KnowledgeSource,
    ) !u64 {
        const id = try self.store(content, category, source);

        // Add embedding
        const idx = self.id_to_index.get(id) orelse return error.NotFound;
        self.knowledge.items[idx].embedding = try self.allocator.dupe(f32, embedding);

        // Update embedding index
        if (self.embedding_index) |*emb_idx| {
            try emb_idx.add(id, embedding);
        }

        return id;
    }

    /// Retrieve knowledge by ID
    pub fn get(self: *Self, id: u64) ?*Knowledge {
        const idx = self.id_to_index.get(id) orelse return null;
        if (idx >= self.knowledge.items.len) return null;

        var k = &self.knowledge.items[idx];
        k.accessed_at = types.getTimestampSec();
        k.access_count += 1;
        return k;
    }

    /// Retrieve knowledge by category (caller must free returned slice)
    pub fn getByCategory(self: *Self, category: Knowledge.KnowledgeCategory) ![]*Knowledge {
        const cat_key = @intFromEnum(category);
        if (self.category_index.get(cat_key)) |indices| {
            var results = std.ArrayListUnmanaged(*Knowledge).empty;
            errdefer results.deinit(self.allocator);

            for (indices.items) |idx| {
                if (idx < self.knowledge.items.len) {
                    var k = &self.knowledge.items[idx];
                    k.accessed_at = types.getTimestampSec();
                    k.access_count += 1;
                    try results.append(self.allocator, k);
                }
            }

            return try results.toOwnedSlice(self.allocator);
        }
        return &.{};
    }

    /// Search by embedding similarity
    pub fn search(self: *Self, query_embedding: []const f32, top_k: usize) ![]const KnowledgeMatch {
        var matches = std.ArrayListUnmanaged(KnowledgeMatch).empty;
        errdefer matches.deinit(self.allocator);

        for (self.knowledge.items, 0..) |*k, idx| {
            if (k.embedding) |emb| {
                const similarity = cosineSimilarity(query_embedding, emb);
                try matches.append(self.allocator, .{
                    .knowledge_idx = idx,
                    .id = k.id,
                    .similarity = similarity,
                });
            }
        }

        // Sort by similarity
        const items = matches.items;
        std.mem.sort(KnowledgeMatch, items, {}, struct {
            fn lessThan(_: void, a: KnowledgeMatch, b: KnowledgeMatch) bool {
                return a.similarity > b.similarity;
            }
        }.lessThan);

        const result_count = @min(top_k, items.len);
        const all = try matches.toOwnedSlice(self.allocator);
        return all[0..result_count];
    }

    pub const KnowledgeMatch = struct {
        knowledge_idx: usize,
        id: u64,
        similarity: f32,
    };

    /// Create association between knowledge items
    pub fn associate(self: *Self, id1: u64, id2: u64) !void {
        if (self.get(id1)) |k1| {
            try k1.associations.append(self.allocator, id2);
        }
        if (self.get(id2)) |k2| {
            try k2.associations.append(self.allocator, id1);
        }
    }

    /// Get associated knowledge
    pub fn getAssociations(self: *Self, id: u64) []const *Knowledge {
        const k = self.get(id) orelse return &.{};

        var results: [64]*Knowledge = undefined;
        var count: usize = 0;

        for (k.associations.items) |assoc_id| {
            if (self.get(assoc_id)) |assoc| {
                if (count < 64) {
                    results[count] = assoc;
                    count += 1;
                }
            }
        }

        return results[0..count];
    }

    /// Update confidence of knowledge
    pub fn updateConfidence(self: *Self, id: u64, new_confidence: f32) void {
        if (self.get(id)) |k| {
            k.confidence = std.math.clamp(new_confidence, 0.0, 1.0);
        }
    }

    /// Prune low-confidence, rarely accessed knowledge
    pub fn prune(self: *Self, min_confidence: f32, min_accesses: usize) usize {
        var pruned: usize = 0;
        var i: usize = 0;

        while (i < self.knowledge.items.len) {
            const k = &self.knowledge.items[i];
            if (k.confidence < min_confidence and k.access_count < min_accesses) {
                // Remove from indexes
                _ = self.id_to_index.remove(k.id);

                // Remove and free
                var removed = self.knowledge.orderedRemove(i);
                removed.deinit(self.allocator);
                pruned += 1;
            } else {
                i += 1;
            }
        }

        return pruned;
    }

    /// Get memory statistics
    pub fn getStats(self: *const Self) SemanticStats {
        var with_embeddings: usize = 0;
        var total_associations: usize = 0;
        var avg_confidence: f32 = 0;

        for (self.knowledge.items) |k| {
            if (k.embedding != null) with_embeddings += 1;
            total_associations += k.associations.items.len;
            avg_confidence += k.confidence;
        }

        if (self.knowledge.items.len > 0) {
            avg_confidence /= @as(f32, @floatFromInt(self.knowledge.items.len));
        }

        return .{
            .knowledge_count = self.knowledge.items.len,
            .with_embeddings = with_embeddings,
            .total_associations = total_associations,
            .avg_confidence = avg_confidence,
        };
    }

    pub const SemanticStats = struct {
        knowledge_count: usize,
        with_embeddings: usize,
        total_associations: usize,
        avg_confidence: f32,
    };
};

// ============================================================================
// Embedding Index (Simplified HNSW-like structure)
// ============================================================================

pub const EmbeddingIndex = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayListUnmanaged(IndexEntry),
    dim: usize,

    const IndexEntry = struct {
        id: u64,
        embedding: []f32,
    };

    pub fn init(allocator: std.mem.Allocator, dim: usize) EmbeddingIndex {
        return .{
            .allocator = allocator,
            .entries = .{},
            .dim = dim,
        };
    }

    pub fn deinit(self: *EmbeddingIndex) void {
        for (self.entries.items) |entry| {
            self.allocator.free(entry.embedding);
        }
        self.entries.deinit(self.allocator);
    }

    pub fn add(self: *EmbeddingIndex, id: u64, embedding: []const f32) !void {
        const emb_copy = try self.allocator.dupe(f32, embedding);
        try self.entries.append(self.allocator, .{
            .id = id,
            .embedding = emb_copy,
        });
    }

    pub fn search(self: *EmbeddingIndex, query: []const f32, top_k: usize) ![]const struct { id: u64, score: f32 } {
        var results = std.ArrayListUnmanaged(struct { id: u64, score: f32 }).empty;
        errdefer results.deinit(self.allocator);

        for (self.entries.items) |entry| {
            const score = cosineSimilarity(query, entry.embedding);
            try results.append(self.allocator, .{ .id = entry.id, .score = score });
        }

        const items = results.items;
        std.mem.sort(@TypeOf(items[0]), items, {}, struct {
            fn lessThan(_: void, a: anytype, b: anytype) bool {
                return a.score > b.score;
            }
        }.lessThan);

        return results.toOwnedSlice(self.allocator)[0..@min(top_k, items.len)];
    }
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

test "semantic memory store and retrieve" {
    const allocator = std.testing.allocator;

    var memory = SemanticMemory.init(allocator, 128);
    defer memory.deinit();

    const id = try memory.store("Zig is a systems programming language", .fact, .user_stated);
    try std.testing.expect(id > 0);

    const k = memory.get(id);
    try std.testing.expect(k != null);
    try std.testing.expectEqualStrings("Zig is a systems programming language", k.?.content);
}

test "semantic memory associations" {
    const allocator = std.testing.allocator;

    var memory = SemanticMemory.init(allocator, 128);
    defer memory.deinit();

    const id1 = try memory.store("Zig", .concept, .user_stated);
    const id2 = try memory.store("Systems programming", .concept, .user_stated);

    try memory.associate(id1, id2);

    const assocs = memory.getAssociations(id1);
    try std.testing.expectEqual(@as(usize, 1), assocs.len);
}

test "semantic memory category retrieval" {
    const allocator = std.testing.allocator;

    var memory = SemanticMemory.init(allocator, 128);
    defer memory.deinit();

    _ = try memory.store("Fact 1", .fact, .user_stated);
    _ = try memory.store("Fact 2", .fact, .user_stated);
    _ = try memory.store("Preference 1", .preference, .user_stated);

    const facts = try memory.getByCategory(.fact);
    defer if (facts.len > 0) allocator.free(facts);
    try std.testing.expectEqual(@as(usize, 2), facts.len);
}

test {
    std.testing.refAllDecls(@This());
}
