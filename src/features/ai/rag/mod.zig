//! Retrieval-Augmented Generation (RAG) pipeline.
//!
//! Provides a complete RAG implementation for enhancing LLM responses
//! with relevant context from a document corpus. Includes:
//! - Document chunking with multiple strategies
//! - Vector-based retrieval
//! - Context injection and prompt building

const std = @import("std");
const document = @import("document.zig");
const chunker = @import("chunker.zig");
const retriever = @import("retriever.zig");
const context = @import("context.zig");

// Document types
pub const Document = document.Document;
pub const DocumentMetadata = document.DocumentMetadata;
pub const DocumentType = document.DocumentType;

// Chunking types
pub const Chunk = chunker.Chunk;
pub const ChunkingStrategy = chunker.ChunkingStrategy;
pub const ChunkerConfig = chunker.ChunkerConfig;
pub const Chunker = chunker.Chunker;

// Retrieval types
pub const RetrievalResult = retriever.RetrievalResult;
pub const RetrieverConfig = retriever.RetrieverConfig;
pub const Retriever = retriever.Retriever;

// Context types
pub const ContextConfig = context.ContextConfig;
pub const ContextBuilder = context.ContextBuilder;
pub const RagContext = context.RagContext;

/// RAG pipeline configuration.
pub const RagConfig = struct {
    /// Chunking configuration.
    chunking: ChunkerConfig = .{},
    /// Retrieval configuration.
    retrieval: RetrieverConfig = .{},
    /// Context building configuration.
    context: ContextConfig = .{},
    /// Maximum total context tokens.
    max_context_tokens: usize = 2000,
    /// Enable deduplication of similar chunks.
    deduplicate: bool = true,
    /// Similarity threshold for deduplication.
    dedup_threshold: f32 = 0.95,
};

/// RAG pipeline for retrieval-augmented generation.
pub const RagPipeline = struct {
    allocator: std.mem.Allocator,
    config: RagConfig,
    chunker_inst: Chunker,
    retriever_inst: Retriever,
    context_builder: ContextBuilder,
    documents: std.ArrayListUnmanaged(Document),
    chunks: std.ArrayListUnmanaged(IndexedChunk),

    const IndexedChunk = struct {
        chunk: Chunk,
        doc_id: []const u8,
        embedding: ?[]f32,
    };

    /// Initialize RAG pipeline.
    pub fn init(allocator: std.mem.Allocator, config: RagConfig) RagPipeline {
        return .{
            .allocator = allocator,
            .config = config,
            .chunker_inst = Chunker.init(allocator, config.chunking),
            .retriever_inst = Retriever.init(allocator, config.retrieval),
            .context_builder = ContextBuilder.init(allocator, config.context),
            .documents = .{},
            .chunks = .{},
        };
    }

    /// Deinitialize and free resources.
    pub fn deinit(self: *RagPipeline) void {
        for (self.documents.items) |*doc| {
            doc.deinit(self.allocator);
        }
        self.documents.deinit(self.allocator);

        for (self.chunks.items) |*indexed| {
            indexed.chunk.deinit(self.allocator);
            self.allocator.free(indexed.doc_id);
            if (indexed.embedding) |emb| {
                self.allocator.free(emb);
            }
        }
        self.chunks.deinit(self.allocator);

        self.chunker_inst.deinit();
        self.retriever_inst.deinit();
        self.context_builder.deinit();
        self.* = undefined;
    }

    /// Add a document to the corpus.
    pub fn addDocument(self: *RagPipeline, doc: Document) !void {
        // Clone document
        var cloned = try doc.clone(self.allocator);
        errdefer cloned.deinit(self.allocator);

        // Chunk the document
        const doc_chunks = try self.chunker_inst.chunk(doc.content);
        defer self.allocator.free(doc_chunks);

        // Index chunks
        for (doc_chunks) |chunk| {
            var cloned_chunk = try chunk.clone(self.allocator);
            errdefer cloned_chunk.deinit(self.allocator);

            const doc_id = try self.allocator.dupe(u8, doc.id);
            errdefer self.allocator.free(doc_id);

            // Compute embedding if retriever supports it
            const embedding = try self.retriever_inst.computeEmbedding(chunk.content);

            try self.chunks.append(self.allocator, .{
                .chunk = cloned_chunk,
                .doc_id = doc_id,
                .embedding = embedding,
            });
        }

        try self.documents.append(self.allocator, cloned);
    }

    /// Add a text document with auto-generated ID.
    pub fn addText(self: *RagPipeline, content: []const u8, title: ?[]const u8) !void {
        var id_buf: [32]u8 = undefined;
        const id = std.fmt.bufPrint(&id_buf, "doc_{d}", .{self.documents.items.len}) catch "doc_unknown";

        const doc = Document{
            .id = id,
            .content = content,
            .title = title,
            .doc_type = .text,
            .metadata = null,
        };

        try self.addDocument(doc);
    }

    /// Query the RAG pipeline and get augmented context.
    pub fn query(
        self: *RagPipeline,
        query_text: []const u8,
        top_k: ?usize,
    ) !RagResponse {
        const k = top_k orelse self.config.retrieval.top_k;

        // Retrieve relevant chunks
        var results = try self.retrieve(query_text, k);
        defer self.allocator.free(results);

        // Deduplicate if enabled
        if (self.config.deduplicate) {
            results = try self.deduplicateResults(results);
        }

        // Build context
        const rag_context = try self.context_builder.build(
            query_text,
            results,
            self.config.max_context_tokens,
        );

        return RagResponse{
            .query = query_text,
            .context = rag_context,
            .num_chunks = results.len,
            .sources = try self.extractSources(results),
        };
    }

    /// Retrieve relevant chunks for a query.
    pub fn retrieve(
        self: *RagPipeline,
        query_text: []const u8,
        top_k: usize,
    ) ![]RetrievalResult {
        // Compute query embedding
        const query_embedding = try self.retriever_inst.computeEmbedding(query_text);
        defer if (query_embedding) |emb| self.allocator.free(emb);

        // Score all chunks
        var scored = std.ArrayListUnmanaged(ScoredChunk){};
        defer scored.deinit(self.allocator);

        for (self.chunks.items) |indexed| {
            const score = if (query_embedding != null and indexed.embedding != null)
                self.retriever_inst.computeSimilarity(query_embedding.?, indexed.embedding.?)
            else
                self.retriever_inst.textSimilarity(query_text, indexed.chunk.content);

            if (score >= self.config.retrieval.min_score) {
                try scored.append(self.allocator, .{
                    .chunk = indexed.chunk,
                    .doc_id = indexed.doc_id,
                    .score = score,
                });
            }
        }

        // Sort by score descending
        std.mem.sort(ScoredChunk, scored.items, {}, struct {
            fn lessThan(_: void, a: ScoredChunk, b: ScoredChunk) bool {
                return a.score > b.score;
            }
        }.lessThan);

        // Build results
        const result_count = @min(top_k, scored.items.len);
        var results = try self.allocator.alloc(RetrievalResult, result_count);
        errdefer self.allocator.free(results);

        for (scored.items[0..result_count], 0..) |item, i| {
            results[i] = .{
                .chunk = item.chunk,
                .doc_id = item.doc_id,
                .score = item.score,
                .rank = i + 1,
            };
        }

        return results;
    }

    const ScoredChunk = struct {
        chunk: Chunk,
        doc_id: []const u8,
        score: f32,
    };

    /// Deduplicate similar results.
    fn deduplicateResults(
        self: *RagPipeline,
        results: []RetrievalResult,
    ) ![]RetrievalResult {
        if (results.len <= 1) return results;

        var unique = std.ArrayListUnmanaged(RetrievalResult){};
        errdefer unique.deinit(self.allocator);

        outer: for (results) |result| {
            for (unique.items) |existing| {
                const sim = self.retriever_inst.textSimilarity(
                    result.chunk.content,
                    existing.chunk.content,
                );
                if (sim >= self.config.dedup_threshold) {
                    continue :outer;
                }
            }
            try unique.append(self.allocator, result);
        }

        self.allocator.free(results);
        return unique.toOwnedSlice(self.allocator);
    }

    /// Extract source document IDs from results.
    fn extractSources(
        self: *RagPipeline,
        results: []const RetrievalResult,
    ) ![][]const u8 {
        var sources = std.StringHashMapUnmanaged(void){};
        defer sources.deinit(self.allocator);

        for (results) |result| {
            try sources.put(self.allocator, result.doc_id, {});
        }

        var source_list = try self.allocator.alloc([]const u8, sources.count());
        var i: usize = 0;
        var iter = sources.keyIterator();
        while (iter.next()) |key| {
            source_list[i] = key.*;
            i += 1;
        }

        return source_list;
    }

    /// Get document by ID.
    pub fn getDocument(self: *const RagPipeline, id: []const u8) ?*const Document {
        for (self.documents.items) |*doc| {
            if (std.mem.eql(u8, doc.id, id)) {
                return doc;
            }
        }
        return null;
    }

    /// Get number of indexed chunks.
    pub fn chunkCount(self: *const RagPipeline) usize {
        return self.chunks.items.len;
    }

    /// Get number of documents.
    pub fn documentCount(self: *const RagPipeline) usize {
        return self.documents.items.len;
    }

    /// Clear all documents and chunks.
    pub fn clear(self: *RagPipeline) void {
        for (self.documents.items) |*doc| {
            doc.deinit(self.allocator);
        }
        self.documents.clearRetainingCapacity();

        for (self.chunks.items) |*indexed| {
            indexed.chunk.deinit(self.allocator);
            self.allocator.free(indexed.doc_id);
            if (indexed.embedding) |emb| {
                self.allocator.free(emb);
            }
        }
        self.chunks.clearRetainingCapacity();
    }
};

/// RAG query response.
pub const RagResponse = struct {
    /// Original query.
    query: []const u8,
    /// Built context for LLM.
    context: RagContext,
    /// Number of chunks used.
    num_chunks: usize,
    /// Source document IDs.
    sources: [][]const u8,

    pub fn deinit(self: *RagResponse, allocator: std.mem.Allocator) void {
        self.context.deinit(allocator);
        allocator.free(self.sources);
        self.* = undefined;
    }

    /// Get the augmented prompt text.
    pub fn getPrompt(self: *const RagResponse) []const u8 {
        return self.context.prompt;
    }
};

/// Create a new RAG pipeline with default configuration.
pub fn createPipeline(allocator: std.mem.Allocator) RagPipeline {
    return RagPipeline.init(allocator, .{});
}

/// Create a RAG pipeline with custom configuration.
pub fn createPipelineWithConfig(
    allocator: std.mem.Allocator,
    config: RagConfig,
) RagPipeline {
    return RagPipeline.init(allocator, config);
}

test "rag pipeline basic operations" {
    const allocator = std.testing.allocator;
    var pipeline = RagPipeline.init(allocator, .{});
    defer pipeline.deinit();

    try pipeline.addText("Machine learning is a subset of AI.", "ML Intro");
    try pipeline.addText("Deep learning uses neural networks.", "DL Intro");

    try std.testing.expectEqual(@as(usize, 2), pipeline.documentCount());
    try std.testing.expect(pipeline.chunkCount() >= 2);
}

test "rag pipeline query" {
    const allocator = std.testing.allocator;
    var pipeline = RagPipeline.init(allocator, .{
        .retrieval = .{ .min_score = 0 },
    });
    defer pipeline.deinit();

    try pipeline.addText("Python is a programming language.", null);
    try pipeline.addText("JavaScript runs in browsers.", null);

    var response = try pipeline.query("programming language", 2);
    defer response.deinit(allocator);

    try std.testing.expect(response.num_chunks > 0);
}
