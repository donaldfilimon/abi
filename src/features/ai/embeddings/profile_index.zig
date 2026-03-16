//! Profile Embedding Index
//!
//! Manages semantic search for AI profiles using WDBX and vector embeddings.
//! Allows mapping user intent to the most appropriate profile based on
//! characteristic embeddings.
//!
//! Features:
//! - HNSW-based approximate nearest neighbor search
//! - Profile characteristic embedding storage
//! - Conversation history embedding for adaptive learning
//! - Domain-aware profile matching

const std = @import("std");
const build_options = @import("build_options");
const types = @import("../types.zig");
const embeddings = @import("mod.zig");
const database = if (build_options.feat_database)
    @import("../../database/mod.zig")
else
    @import("../../database/stub.zig");
const seed_data = @import("seed_data.zig");
const time = @import("../../../services/shared/mod.zig").time;
const simd = @import("../../../services/shared/simd/mod.zig");

/// Result of a profile matching operation.
pub const ProfileMatch = struct {
    /// The matched profile type.
    profile: types.ProfileType,
    /// Similarity score (usually cosine similarity, 0.0 to 1.0).
    similarity: f32,
    /// Optional metadata associated with the profile embedding.
    metadata: ?[]const u8 = null,
    /// Whether this match was boosted by domain preference.
    domain_boosted: bool = false,
};

/// Configuration for the profile embedding index.
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
const NAMESPACE_PROFILES: u32 = 1;
const NAMESPACE_CONVERSATIONS: u32 = 2;
const NAMESPACE_FEEDBACK: u32 = 3;

const NAMESPACE_SHIFT: u6 = 60;
const NAMESPACE_MASK: u64 = 0xF000_0000_0000_0000;
const ID_MASK: u64 = 0x0FFF_FFFF_FFFF_FFFF;

/// Semantic index for profile selection and behavioral learning.
pub const ProfileEmbeddingIndex = struct {
    allocator: std.mem.Allocator,
    /// Database context for vector storage (WDBX).
    db: *database.Context,
    /// Embedding context for vector generation.
    embeddings_ctx: *embeddings.Context,
    /// Cache of profile vectors for fast similarity checks.
    profile_vectors: std.AutoHashMapUnmanaged(types.ProfileType, []const f32),
    /// Configuration.
    config: IndexConfig,
    /// Whether the index has been seeded with profile characteristics.
    is_seeded: bool = false,
    /// Statistics tracking.
    stats: IndexStats = .{},

    const Self = @This();

    /// Statistics for the profile index.
    pub const IndexStats = struct {
        queries_total: u64 = 0,
        cache_hits: u64 = 0,
        embeddings_stored: u64 = 0,
        conversations_stored: u64 = 0,
    };

    /// Initialize the profile embedding index.
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
            .profile_vectors = .empty,
            .config = config,
        };

        // Auto-seed if configured
        if (config.auto_seed) {
            try self.seedProfileCharacteristics();
        }

        return self;
    }

    /// Shutdown the index and free resources.
    pub fn deinit(self: *Self) void {
        var it = self.profile_vectors.valueIterator();
        while (it.next()) |vec| {
            self.allocator.free(vec.*);
        }
        self.profile_vectors.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Seed the index with profile characteristic embeddings.
    pub fn seedProfileCharacteristics(self: *Self) !void {
        if (self.is_seeded) return;

        // Seed each profile
        inline for (std.meta.fields(types.ProfileType)) |field| {
            const profile: types.ProfileType = @enumFromInt(field.value);

            const combined_text = try seed_data.getCombinedCharacteristics(self.allocator, profile);
            defer self.allocator.free(combined_text);

            try self.storeProfileEmbedding(profile, combined_text);
        }

        self.is_seeded = true;
    }

    /// Store a profile's characteristic embedding in the index.
    pub fn storeProfileEmbedding(
        self: *Self,
        profile: types.ProfileType,
        characteristics: []const u8,
    ) !void {
        // Generate embedding for characteristics text
        const vector = try self.embeddings_ctx.embed(characteristics);

        // dupe vector for local cache
        const vector_owned = try self.allocator.dupe(f32, vector);
        errdefer self.allocator.free(vector_owned);

        // Remove old vector if exists
        if (self.profile_vectors.fetchRemove(profile)) |old| {
            self.allocator.free(old.value);
        }

        try self.profile_vectors.put(self.allocator, profile, vector_owned);

        // Store in persistent WDBX database with namespace
        const id = makeProfileId(profile);
        try self.db.insertVector(id, vector, @tagName(profile));

        self.stats.embeddings_stored += 1;
    }

    /// Find the best matching profiles for a given query string.
    pub fn findBestProfile(
        self: *Self,
        allocator: std.mem.Allocator,
        query: []const u8,
        top_k: usize,
    ) ![]ProfileMatch {
        self.stats.queries_total += 1;

        const query_vector = try self.embeddings_ctx.embed(query);

        // First try fast in-memory search if we have cached vectors
        if (self.profile_vectors.count() > 0) {
            return self.searchCachedVectors(allocator, query_vector, query, top_k);
        }

        // Fall back to WDBX search (over-fetch, then filter by namespace)
        const search_k = top_k * 4 + 8;
        const search_results = try self.db.searchVectors(query_vector, search_k);
        defer allocator.free(search_results);

        var matches: std.ArrayListUnmanaged(ProfileMatch) = .empty;
        defer matches.deinit(allocator);

        for (search_results) |res| {
            if (namespaceFromId(res.id) != NAMESPACE_PROFILES) continue;
            const profile = profileFromId(res.id);
            try matches.append(allocator, .{
                .profile = profile,
                .similarity = res.score,
                .metadata = @tagName(profile),
            });
            if (matches.items.len >= top_k) break;
        }

        return matches.toOwnedSlice(allocator);
    }

    /// Search using cached profile vectors (faster for small number of profiles).
    fn searchCachedVectors(
        self: *Self,
        allocator: std.mem.Allocator,
        query_vector: []const f32,
        query_text: []const u8,
        top_k: usize,
    ) ![]ProfileMatch {
        self.stats.cache_hits += 1;

        // Calculate similarities for all cached profiles
        var scored: std.ArrayListUnmanaged(struct { profile: types.ProfileType, score: f32 }) = .empty;
        defer scored.deinit(allocator);

        var it = self.profile_vectors.iterator();
        while (it.next()) |entry| {
            const similarity = cosineSimilarity(query_vector, entry.value_ptr.*);
            try scored.append(allocator, .{ .profile = entry.key_ptr.*, .score = similarity });
        }

        // Sort by score descending
        std.mem.sort(@TypeOf(scored.items[0]), scored.items, {}, struct {
            fn lessThan(_: void, a: @TypeOf(scored.items[0]), b: @TypeOf(scored.items[0])) bool {
                return a.score > b.score; // Descending
            }
        }.lessThan);

        // Take top_k results
        const result_count = @min(top_k, scored.items.len);
        var matches = try allocator.alloc(ProfileMatch, result_count);
        errdefer allocator.free(matches);

        for (scored.items[0..result_count], 0..) |item, i| {
            // Apply domain boost if applicable
            var score = item.score;
            var domain_boosted = false;

            const chars = seed_data.getCharacteristics(item.profile);
            for (chars.keywords) |keyword| {
                if (std.mem.indexOf(u8, query_text, keyword) != null) {
                    score = @min(score * 1.1, 1.0); // 10% boost, capped at 1.0
                    domain_boosted = true;
                    break;
                }
            }

            matches[i] = .{
                .profile = item.profile,
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
        profile_used: types.ProfileType,
        success_score: f32,
    ) !void {
        const vector = try self.embeddings_ctx.embed(content);

        // Build metadata JSON
        var metadata_buf: std.ArrayListUnmanaged(u8) = .empty;
        defer metadata_buf.deinit(self.allocator);

        try std.json.stringify(.{
            .profile = @tagName(profile_used),
            .score = success_score,
            .timestamp = time.unixSeconds(),
        }, .{}, metadata_buf.writer(self.allocator));

        const id = makeNamespacedId(NAMESPACE_CONVERSATIONS, conversation_id);
        try self.db.insertVector(id, vector, metadata_buf.items);

        self.stats.conversations_stored += 1;
    }

    /// Store feedback embedding to learn from user corrections.
    pub fn storeFeedbackEmbedding(
        self: *Self,
        feedback_id: u64,
        original_query: []const u8,
        correct_profile: types.ProfileType,
        was_routed_to: types.ProfileType,
    ) !void {
        const vector = try self.embeddings_ctx.embed(original_query);

        var metadata_buf: std.ArrayListUnmanaged(u8) = .empty;
        defer metadata_buf.deinit(self.allocator);

        try std.json.stringify(.{
            .correct = @tagName(correct_profile),
            .was_routed = @tagName(was_routed_to),
            .timestamp = time.unixSeconds(),
        }, .{}, metadata_buf.writer(self.allocator));

        const id = makeNamespacedId(NAMESPACE_FEEDBACK, feedback_id);
        try self.db.insertVector(id, vector, metadata_buf.items);
    }

    /// Get index statistics.
    pub fn getStats(self: *const Self) IndexStats {
        return self.stats;
    }

    /// Get the number of cached profile vectors.
    pub fn getCachedCount(self: *const Self) usize {
        return self.profile_vectors.count();
    }

    /// Check if a specific profile has been seeded.
    pub fn hasProfile(self: *const Self, profile: types.ProfileType) bool {
        return self.profile_vectors.contains(profile);
    }
};

// Helper functions

fn makeNamespacedId(namespace: u32, id: u64) u64 {
    return (@as(u64, namespace) << NAMESPACE_SHIFT) | (id & ID_MASK);
}

fn namespaceFromId(id: u64) u32 {
    return @intCast((id & NAMESPACE_MASK) >> NAMESPACE_SHIFT);
}

fn stripNamespace(id: u64) u64 {
    return id & ID_MASK;
}

/// Create a unique ID for a profile within the WDBX namespace.
fn makeProfileId(profile: types.ProfileType) u64 {
    return makeNamespacedId(NAMESPACE_PROFILES, @intFromEnum(profile));
}

/// Convert a WDBX ID back to a ProfileType.
fn profileFromId(id: u64) types.ProfileType {
    if (namespaceFromId(id) != NAMESPACE_PROFILES) {
        return .assistant;
    }
    const profile_id: u32 = @intCast(stripNamespace(id));
    if (profile_id < std.meta.fields(types.ProfileType).len) {
        return @enumFromInt(profile_id);
    }
    return .assistant;
}

/// Calculate cosine similarity between two vectors.
fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    return simd.cosineSimilarity(a, b);
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

test "makeProfileId and profileFromId roundtrip" {
    const profiles = [_]types.ProfileType{ .abi, .abbey, .aviva };

    for (profiles) |profile| {
        const id = makeProfileId(profile);
        const recovered = profileFromId(id);
        try std.testing.expect(profile == recovered);
    }
}

test {
    std.testing.refAllDecls(@This());
}
