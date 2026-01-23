//! Persona Embedding Index
//!
//! Manages semantic search for AI personas using WDBX and vector embeddings.
//! Allows mapping user intent to the most appropriate persona based on
//! characteristic embeddings.
//!
//! Features:
//! - HNSW-based approximate nearest neighbor search
//! - Persona characteristic embedding storage
//! - Conversation history embedding for adaptive learning
//! - Domain-aware persona matching

const std = @import("std");
const types = @import("../types.zig");
const embeddings = @import("../../embeddings/mod.zig");
const database = @import("../../../database/mod.zig");
const seed_data = @import("seed_data.zig");

/// Result of a persona matching operation.
pub const PersonaMatch = struct {
    /// The matched persona type.
    persona: types.PersonaType,
    /// Similarity score (usually cosine similarity, 0.0 to 1.0).
    similarity: f32,
    /// Optional metadata associated with the persona embedding.
    metadata: ?[]const u8 = null,
    /// Whether this match was boosted by domain preference.
    domain_boosted: bool = false,
};

/// Configuration for the persona embedding index.
pub const IndexConfig = struct {
    /// HNSW M parameter (number of neighbors to connect).
    hnsw_m: u16 = 16,
    /// HNSW ef_construction parameter (size of dynamic candidate list).
    ef_construction: u16 = 200,
    /// Embedding dimension (depends on model used).
    embedding_dim: usize = 384,
    /// Whether to auto-seed on initialization.
    auto_seed: bool = true,
    /// Maximum conversation embeddings to cache.
    max_conversation_cache: usize = 10000,
};

/// Namespace IDs for different embedding collections in WDBX.
const NAMESPACE_PERSONAS: u32 = 1;
const NAMESPACE_CONVERSATIONS: u32 = 2;
const NAMESPACE_FEEDBACK: u32 = 3;

/// Semantic index for persona selection and behavioral learning.
pub const PersonaEmbeddingIndex = struct {
    allocator: std.mem.Allocator,
    /// Database context for vector storage (WDBX).
    db: *database.Context,
    /// Embedding context for vector generation.
    embeddings_ctx: *embeddings.Context,
    /// Cache of persona vectors for fast similarity checks.
    persona_vectors: std.AutoHashMapUnmanaged(types.PersonaType, []const f32),
    /// Configuration.
    config: IndexConfig,
    /// Whether the index has been seeded with persona characteristics.
    is_seeded: bool = false,
    /// Statistics tracking.
    stats: IndexStats = .{},

    const Self = @This();

    /// Statistics for the persona index.
    pub const IndexStats = struct {
        queries_total: u64 = 0,
        cache_hits: u64 = 0,
        embeddings_stored: u64 = 0,
        conversations_stored: u64 = 0,
    };

    /// Initialize the persona embedding index.
    pub fn init(allocator: std.mem.Allocator, db: *database.Context, emb: *embeddings.Context) !*Self {
        return initWithConfig(allocator, db, emb, .{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(
        allocator: std.mem.Allocator,
        db: *database.Context,
        emb: *embeddings.Context,
        config: IndexConfig,
    ) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .db = db,
            .embeddings_ctx = emb,
            .persona_vectors = .{},
            .config = config,
        };

        // Auto-seed if configured
        if (config.auto_seed) {
            try self.seedPersonaCharacteristics();
        }

        return self;
    }

    /// Shutdown the index and free resources.
    pub fn deinit(self: *Self) void {
        var it = self.persona_vectors.valueIterator();
        while (it.next()) |vec| {
            self.allocator.free(vec.*);
        }
        self.persona_vectors.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Seed the index with persona characteristic embeddings.
    pub fn seedPersonaCharacteristics(self: *Self) !void {
        if (self.is_seeded) return;

        // Seed each persona
        inline for (std.meta.fields(types.PersonaType)) |field| {
            const persona: types.PersonaType = @enumFromInt(field.value);

            // Skip unknown/system personas
            if (persona == .assistant) continue;

            const combined_text = try seed_data.getCombinedCharacteristics(self.allocator, persona);
            defer self.allocator.free(combined_text);

            try self.storePersonaEmbedding(persona, combined_text);
        }

        self.is_seeded = true;
    }

    /// Store a persona's characteristic embedding in the index.
    pub fn storePersonaEmbedding(
        self: *Self,
        persona: types.PersonaType,
        characteristics: []const u8,
    ) !void {
        // Generate embedding for characteristics text
        const vector = try self.embeddings_ctx.embed(characteristics);

        // dupe vector for local cache
        const vector_owned = try self.allocator.dupe(f32, vector);
        errdefer self.allocator.free(vector_owned);

        // Remove old vector if exists
        if (self.persona_vectors.fetchRemove(persona)) |old| {
            self.allocator.free(old.value);
        }

        try self.persona_vectors.put(self.allocator, persona, vector_owned);

        // Store in persistent WDBX database with namespace
        const id = makePersonaId(persona);
        try self.db.insertVectorWithNamespace(NAMESPACE_PERSONAS, id, vector, @tagName(persona));

        self.stats.embeddings_stored += 1;
    }

    /// Find the best matching personas for a given query string.
    pub fn findBestPersona(
        self: *Self,
        allocator: std.mem.Allocator,
        query: []const u8,
        top_k: usize,
    ) ![]PersonaMatch {
        self.stats.queries_total += 1;

        const query_vector = try self.embeddings_ctx.embed(query);

        // First try fast in-memory search if we have cached vectors
        if (self.persona_vectors.count() > 0) {
            return self.searchCachedVectors(allocator, query_vector, query, top_k);
        }

        // Fall back to WDBX search
        const search_results = try self.db.searchVectorsWithNamespace(NAMESPACE_PERSONAS, query_vector, top_k);

        var matches = try allocator.alloc(PersonaMatch, search_results.len);
        errdefer allocator.free(matches);

        for (search_results, 0..) |res, i| {
            matches[i] = .{
                .persona = personaFromId(res.id),
                .similarity = res.score,
                .metadata = res.metadata,
            };
        }

        return matches;
    }

    /// Search using cached persona vectors (faster for small number of personas).
    fn searchCachedVectors(
        self: *Self,
        allocator: std.mem.Allocator,
        query_vector: []const f32,
        query_text: []const u8,
        top_k: usize,
    ) ![]PersonaMatch {
        self.stats.cache_hits += 1;

        // Calculate similarities for all cached personas
        var scored = std.ArrayList(struct { persona: types.PersonaType, score: f32 }).init(allocator);
        defer scored.deinit();

        var it = self.persona_vectors.iterator();
        while (it.next()) |entry| {
            const similarity = cosineSimilarity(query_vector, entry.value_ptr.*);
            try scored.append(.{ .persona = entry.key_ptr.*, .score = similarity });
        }

        // Sort by score descending
        std.mem.sort(@TypeOf(scored.items[0]), scored.items, {}, struct {
            fn lessThan(_: void, a: @TypeOf(scored.items[0]), b: @TypeOf(scored.items[0])) bool {
                return a.score > b.score; // Descending
            }
        }.lessThan);

        // Take top_k results
        const result_count = @min(top_k, scored.items.len);
        var matches = try allocator.alloc(PersonaMatch, result_count);
        errdefer allocator.free(matches);

        for (scored.items[0..result_count], 0..) |item, i| {
            // Apply domain boost if applicable
            var score = item.score;
            var domain_boosted = false;

            const chars = seed_data.getCharacteristics(item.persona);
            for (chars.keywords) |keyword| {
                if (std.mem.indexOf(u8, query_text, keyword) != null) {
                    score = @min(score * 1.1, 1.0); // 10% boost, capped at 1.0
                    domain_boosted = true;
                    break;
                }
            }

            matches[i] = .{
                .persona = item.persona,
                .similarity = score,
                .domain_boosted = domain_boosted,
            };
        }

        return matches;
    }

    /// Store a conversation interaction for adaptive learning.
    pub fn storeConversationEmbedding(
        self: *Self,
        conversation_id: u64,
        content: []const u8,
        persona_used: types.PersonaType,
        success_score: f32,
    ) !void {
        const vector = try self.embeddings_ctx.embed(content);

        // Build metadata JSON
        var metadata_buf = std.ArrayList(u8).init(self.allocator);
        defer metadata_buf.deinit();

        try std.json.stringify(.{
            .persona = @tagName(persona_used),
            .score = success_score,
            .timestamp = std.time.timestamp(),
        }, .{}, metadata_buf.writer());

        try self.db.insertVectorWithNamespace(NAMESPACE_CONVERSATIONS, conversation_id, vector, metadata_buf.items);

        self.stats.conversations_stored += 1;
    }

    /// Store feedback embedding to learn from user corrections.
    pub fn storeFeedbackEmbedding(
        self: *Self,
        feedback_id: u64,
        original_query: []const u8,
        correct_persona: types.PersonaType,
        was_routed_to: types.PersonaType,
    ) !void {
        const vector = try self.embeddings_ctx.embed(original_query);

        var metadata_buf = std.ArrayList(u8).init(self.allocator);
        defer metadata_buf.deinit();

        try std.json.stringify(.{
            .correct = @tagName(correct_persona),
            .was_routed = @tagName(was_routed_to),
            .timestamp = std.time.timestamp(),
        }, .{}, metadata_buf.writer());

        try self.db.insertVectorWithNamespace(NAMESPACE_FEEDBACK, feedback_id, vector, metadata_buf.items);
    }

    /// Get index statistics.
    pub fn getStats(self: *const Self) IndexStats {
        return self.stats;
    }

    /// Get the number of cached persona vectors.
    pub fn getCachedCount(self: *const Self) usize {
        return self.persona_vectors.count();
    }

    /// Check if a specific persona has been seeded.
    pub fn hasPersona(self: *const Self, persona: types.PersonaType) bool {
        return self.persona_vectors.contains(persona);
    }
};

// Helper functions

/// Create a unique ID for a persona within the WDBX namespace.
fn makePersonaId(persona: types.PersonaType) u64 {
    return @intFromEnum(persona);
}

/// Convert a WDBX ID back to a PersonaType.
fn personaFromId(id: u64) types.PersonaType {
    const persona_id: u32 = @intCast(id);
    if (persona_id < std.meta.fields(types.PersonaType).len) {
        return @enumFromInt(persona_id);
    }
    return .assistant;
}

/// Calculate cosine similarity between two vectors.
fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len or a.len == 0) return 0.0;

    var dot_product: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;

    for (a, b) |va, vb| {
        dot_product += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    const norm = @sqrt(norm_a) * @sqrt(norm_b);
    if (norm == 0.0) return 0.0;

    return dot_product / norm;
}

// Tests

test "cosineSimilarity" {
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 0.0, 0.0 };
    const c = [_]f32{ 0.0, 1.0, 0.0 };

    // Identical vectors
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cosineSimilarity(&a, &b), 0.001);

    // Orthogonal vectors
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosineSimilarity(&a, &c), 0.001);
}

test "makePersonaId and personaFromId roundtrip" {
    const personas = [_]types.PersonaType{ .abi, .abbey, .aviva };

    for (personas) |persona| {
        const id = makePersonaId(persona);
        const recovered = personaFromId(id);
        try std.testing.expect(persona == recovered);
    }
}
