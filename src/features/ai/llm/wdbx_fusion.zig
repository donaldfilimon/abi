//! WDBX Fusion — RAG Context Retrieval + Embedding Cache
//!
//! Integrates the WDBX vector database with LLM inference to provide:
//!   1. **Embedding cache**: Cache-through pattern storing text→vector mappings
//!   2. **RAG context**: Retrieve relevant document chunks to augment prompts
//!
//! ## Usage
//! ```zig
//! var fusion = try WdbxFusion.init(allocator, .{});
//! defer fusion.deinit();
//!
//! // Add documents for RAG
//! try fusion.addDocument("Zig uses comptime for metaprogramming", "zig-docs");
//!
//! // Augment a prompt with relevant context
//! const augmented = try fusion.augmentPrompt(allocator, "How does Zig do metaprogramming?");
//! defer allocator.free(augmented);
//! ```
//!
//! Gated by `-Denable-llm` + `-Denable-database`. When database is disabled,
//! all operations return gracefully (no-op cache, empty RAG results).

const std = @import("std");
const build_options = @import("build_options");

/// WDBX types — conditionally imported based on build flags.
const db_available = build_options.enable_database;

/// Configuration for WDBX fusion.
pub const FusionConfig = struct {
    /// Maximum context chunks to retrieve for RAG.
    max_context_chunks: u32 = 5,
    /// Minimum cosine similarity to include a chunk.
    min_similarity: f32 = 0.3,
    /// Embedding dimension (must match the embedding model).
    embedding_dim: u32 = 384,
    /// Maximum cached embeddings (LRU eviction when exceeded).
    max_cache_entries: u32 = 10000,
    /// Prefix prepended to RAG context in augmented prompts.
    context_prefix: []const u8 = "Relevant context:\n",
    /// Separator between context chunks.
    chunk_separator: []const u8 = "\n---\n",
    /// Separator between context block and user prompt.
    prompt_separator: []const u8 = "\n\nUser query:\n",
    /// Name for the embedding cache WDBX collection.
    cache_collection: []const u8 = "abi_embedding_cache",
    /// Name for the RAG documents WDBX collection.
    documents_collection: []const u8 = "abi_rag_documents",
};

/// A retrieved context chunk from the RAG store.
pub const ContextChunk = struct {
    /// The document text content.
    content: []const u8,
    /// Cosine similarity score to the query.
    similarity: f32,
    /// Document title/source (from metadata).
    title: []const u8,
};

/// Cached embedding entry.
pub const CacheEntry = struct {
    text_hash: u64,
    vector: []f32,
};

/// WDBX Fusion engine for RAG and embedding caching.
pub const WdbxFusion = struct {
    allocator: std.mem.Allocator,
    config: FusionConfig,

    // In-memory embedding cache (text hash → vector)
    cache: std.AutoHashMapUnmanaged(u64, []f32),
    cache_count: u32,

    // Document store (id → content + title)
    documents: std.AutoHashMapUnmanaged(u64, Document),
    next_doc_id: u64,

    // Document vectors for similarity search (parallel arrays)
    doc_vectors: std.ArrayListUnmanaged(DocVector),

    const Document = struct {
        content: []u8,
        title: []u8,
    };

    const DocVector = struct {
        doc_id: u64,
        vector: []f32,
    };

    pub fn init(allocator: std.mem.Allocator, config: FusionConfig) !WdbxFusion {
        return .{
            .allocator = allocator,
            .config = config,
            .cache = .{},
            .cache_count = 0,
            .documents = .{},
            .next_doc_id = 1,
            .doc_vectors = .{},
        };
    }

    pub fn deinit(self: *WdbxFusion) void {
        // Free cached vectors
        var cache_iter = self.cache.iterator();
        while (cache_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.cache.deinit(self.allocator);

        // Free documents
        var doc_iter = self.documents.iterator();
        while (doc_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.content);
            self.allocator.free(entry.value_ptr.title);
        }
        self.documents.deinit(self.allocator);

        // Free document vectors
        for (self.doc_vectors.items) |dv| {
            self.allocator.free(dv.vector);
        }
        self.doc_vectors.deinit(self.allocator);
    }

    // ========================================================================
    // Embedding Cache
    // ========================================================================

    /// Cache an embedding vector for the given text.
    pub fn cacheEmbedding(self: *WdbxFusion, text: []const u8, vector: []const f32) !void {
        const hash = hashText(text);

        // Evict oldest entry if at capacity (simple replacement)
        if (self.cache_count >= self.config.max_cache_entries) {
            var iter = self.cache.iterator();
            if (iter.next()) |entry| {
                self.allocator.free(entry.value_ptr.*);
                self.cache.removeByPtr(entry.key_ptr);
                self.cache_count -= 1;
            }
        }

        const vec_copy = try self.allocator.dupe(f32, vector);
        errdefer self.allocator.free(vec_copy);

        const result = try self.cache.getOrPut(self.allocator, hash);
        if (result.found_existing) {
            self.allocator.free(result.value_ptr.*);
        } else {
            self.cache_count += 1;
        }
        result.value_ptr.* = vec_copy;
    }

    /// Retrieve a cached embedding, or null if not cached.
    pub fn getCachedEmbedding(self: *const WdbxFusion, text: []const u8) ?[]const f32 {
        const hash = hashText(text);
        if (self.cache.get(hash)) |vec| return vec;
        return null;
    }

    /// Get a cached embedding, or compute and cache it using the provided function.
    pub fn getOrComputeEmbedding(
        self: *WdbxFusion,
        text: []const u8,
        compute_fn: *const fn ([]const u8, std.mem.Allocator) anyerror![]f32,
    ) ![]const f32 {
        if (self.getCachedEmbedding(text)) |cached| return cached;

        const vector = try compute_fn(text, self.allocator);
        errdefer self.allocator.free(vector);

        try self.cacheEmbedding(text, vector);
        // The cacheEmbedding made its own copy, so free the compute result
        self.allocator.free(vector);

        return self.getCachedEmbedding(text) orelse unreachable;
    }

    // ========================================================================
    // RAG Document Store
    // ========================================================================

    /// Add a document to the RAG store with a pre-computed embedding vector.
    pub fn addDocument(self: *WdbxFusion, content: []const u8, title: []const u8, vector: []const f32) !u64 {
        const doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        const content_copy = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(content_copy);
        const title_copy = try self.allocator.dupe(u8, title);
        errdefer self.allocator.free(title_copy);
        const vec_copy = try self.allocator.dupe(f32, vector);
        errdefer self.allocator.free(vec_copy);

        try self.documents.put(self.allocator, doc_id, .{
            .content = content_copy,
            .title = title_copy,
        });
        errdefer _ = self.documents.remove(doc_id);

        try self.doc_vectors.append(self.allocator, .{
            .doc_id = doc_id,
            .vector = vec_copy,
        });

        return doc_id;
    }

    /// Retrieve the most relevant context chunks for a query vector.
    pub fn retrieveContext(
        self: *const WdbxFusion,
        allocator: std.mem.Allocator,
        query_vector: []const f32,
    ) ![]ContextChunk {
        if (self.doc_vectors.items.len == 0) return allocator.alloc(ContextChunk, 0);

        // Compute similarities
        const Scored = struct { doc_id: u64, similarity: f32 };
        var scored = std.ArrayListUnmanaged(Scored).empty;
        defer scored.deinit(allocator);

        for (self.doc_vectors.items) |dv| {
            const sim = cosineSimilarity(query_vector, dv.vector);
            if (sim >= self.config.min_similarity) {
                try scored.append(allocator, .{ .doc_id = dv.doc_id, .similarity = sim });
            }
        }

        // Sort by similarity descending
        std.mem.sort(Scored, scored.items, {}, struct {
            fn cmp(_: void, a: Scored, b: Scored) bool {
                return a.similarity > b.similarity;
            }
        }.cmp);

        // Take top-k
        const limit = @min(scored.items.len, self.config.max_context_chunks);
        const chunks = try allocator.alloc(ContextChunk, limit);
        for (0..limit) |i| {
            const doc = self.documents.get(scored.items[i].doc_id) orelse continue;
            chunks[i] = .{
                .content = doc.content,
                .similarity = scored.items[i].similarity,
                .title = doc.title,
            };
        }

        return chunks;
    }

    /// Augment a prompt with RAG context from retrieved chunks.
    /// Returns a new string: [context_prefix][chunks][prompt_separator][original_prompt]
    pub fn augmentPrompt(
        self: *const WdbxFusion,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        query_vector: []const f32,
    ) ![]u8 {
        const chunks = try self.retrieveContext(allocator, query_vector);
        defer allocator.free(chunks);

        if (chunks.len == 0) {
            return allocator.dupe(u8, prompt);
        }

        // Build augmented prompt
        var buf = std.ArrayListUnmanaged(u8).empty;
        defer buf.deinit(allocator);

        try buf.appendSlice(allocator, self.config.context_prefix);

        for (chunks, 0..) |chunk, i| {
            if (i > 0) {
                try buf.appendSlice(allocator, self.config.chunk_separator);
            }
            if (chunk.title.len > 0) {
                try buf.appendSlice(allocator, "[");
                try buf.appendSlice(allocator, chunk.title);
                try buf.appendSlice(allocator, "] ");
            }
            try buf.appendSlice(allocator, chunk.content);
        }

        try buf.appendSlice(allocator, self.config.prompt_separator);
        try buf.appendSlice(allocator, prompt);

        return buf.toOwnedSlice(allocator);
    }

    /// Get document count in the RAG store.
    pub fn documentCount(self: *const WdbxFusion) usize {
        return self.documents.count();
    }

    /// Get cached embedding count.
    pub fn cacheSize(self: *const WdbxFusion) u32 {
        return self.cache_count;
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

/// FNV-1a hash of text for cache keys.
fn hashText(text: []const u8) u64 {
    return std.hash.Fnv1a_64.hash(text);
}

/// Cosine similarity between two vectors.
fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    const len = @min(a.len, b.len);
    if (len == 0) return 0;

    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    for (0..len) |i| {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 0;
    return dot / denom;
}

// ============================================================================
// Tests
// ============================================================================

test "WdbxFusion init and deinit" {
    const allocator = std.testing.allocator;
    var fusion = try WdbxFusion.init(allocator, .{});
    defer fusion.deinit();
    try std.testing.expectEqual(@as(u32, 0), fusion.cacheSize());
    try std.testing.expectEqual(@as(usize, 0), fusion.documentCount());
}

test "embedding cache round-trip" {
    const allocator = std.testing.allocator;
    var fusion = try WdbxFusion.init(allocator, .{});
    defer fusion.deinit();

    const vec = [_]f32{ 0.1, 0.2, 0.3 };
    try fusion.cacheEmbedding("hello world", &vec);
    try std.testing.expectEqual(@as(u32, 1), fusion.cacheSize());

    const cached = fusion.getCachedEmbedding("hello world") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(f32, 0.1), cached[0]);
    try std.testing.expectEqual(@as(f32, 0.3), cached[2]);
}

test "embedding cache miss" {
    const allocator = std.testing.allocator;
    var fusion = try WdbxFusion.init(allocator, .{});
    defer fusion.deinit();
    try std.testing.expect(fusion.getCachedEmbedding("nonexistent") == null);
}

test "addDocument and retrieveContext" {
    const allocator = std.testing.allocator;
    var fusion = try WdbxFusion.init(allocator, .{ .min_similarity = 0.0 });
    defer fusion.deinit();

    const vec1 = [_]f32{ 1.0, 0.0, 0.0 };
    const vec2 = [_]f32{ 0.0, 1.0, 0.0 };
    _ = try fusion.addDocument("Doc about X", "title-x", &vec1);
    _ = try fusion.addDocument("Doc about Y", "title-y", &vec2);
    try std.testing.expectEqual(@as(usize, 2), fusion.documentCount());

    // Query aligned with doc1
    const query = [_]f32{ 0.9, 0.1, 0.0 };
    const chunks = try fusion.retrieveContext(allocator, &query);
    defer allocator.free(chunks);
    try std.testing.expect(chunks.len >= 1);
    // First result should be most similar to query (doc about X)
    try std.testing.expectEqualStrings("Doc about X", chunks[0].content);
}

test "augmentPrompt with no documents" {
    const allocator = std.testing.allocator;
    var fusion = try WdbxFusion.init(allocator, .{});
    defer fusion.deinit();

    const query = [_]f32{ 1.0, 0.0, 0.0 };
    const result = try fusion.augmentPrompt(allocator, "test prompt", &query);
    defer allocator.free(result);
    // With no documents, should return original prompt
    try std.testing.expectEqualStrings("test prompt", result);
}

test "augmentPrompt with documents" {
    const allocator = std.testing.allocator;
    var fusion = try WdbxFusion.init(allocator, .{ .min_similarity = 0.0 });
    defer fusion.deinit();

    const vec = [_]f32{ 1.0, 0.0, 0.0 };
    _ = try fusion.addDocument("Relevant info here", "source", &vec);

    const query = [_]f32{ 0.9, 0.1, 0.0 };
    const result = try fusion.augmentPrompt(allocator, "my question", &query);
    defer allocator.free(result);
    // Should contain context prefix, document content, and original prompt
    try std.testing.expect(std.mem.indexOf(u8, result, "Relevant info here") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "my question") != null);
}

test "cosineSimilarity correctness" {
    // Identical vectors → similarity = 1.0
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    try std.testing.expect(@abs(cosineSimilarity(&a, &a) - 1.0) < 1e-5);

    // Orthogonal vectors → similarity = 0.0
    const b = [_]f32{ 0.0, 1.0, 0.0 };
    try std.testing.expect(@abs(cosineSimilarity(&a, &b)) < 1e-5);
}

test {
    std.testing.refAllDecls(@This());
}
