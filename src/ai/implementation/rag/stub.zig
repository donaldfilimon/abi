//! Stub implementation for RAG pipeline when AI features are disabled.

const std = @import("std");

/// Stub document type.
pub const DocumentType = enum {
    text,
    markdown,
    html,
    code,
    pdf,
    json,
    other,
};

/// Stub document metadata.
pub const DocumentMetadata = struct {
    source: ?[]const u8 = null,
    author: ?[]const u8 = null,
    created_at: ?i64 = null,
    modified_at: ?i64 = null,
    language: ?[]const u8 = null,
    tags: ?[]const []const u8 = null,
    custom: ?[]const u8 = null,
};

/// Stub document.
pub const Document = struct {
    id: []const u8 = "",
    title: ?[]const u8 = null,
    content: []const u8 = "",
    doc_type: DocumentType = .text,
    metadata: ?DocumentMetadata = null,

    pub fn text(id: []const u8, content: []const u8) Document {
        return .{ .id = id, .content = content };
    }

    pub fn withTitle(id: []const u8, title: []const u8, content: []const u8) Document {
        return .{ .id = id, .title = title, .content = content };
    }

    pub fn clone(self: Document, allocator: std.mem.Allocator) !Document {
        _ = allocator;
        return self;
    }

    pub fn deinit(self: *Document, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.* = undefined;
    }
};

/// Stub chunking strategy.
pub const ChunkingStrategy = enum {
    fixed,
    sentence,
    paragraph,
    recursive,
    semantic,
};

/// Stub chunker config.
pub const ChunkerConfig = struct {
    strategy: ChunkingStrategy = .recursive,
    chunk_size: usize = 500,
    chunk_overlap: usize = 50,
    min_chunk_size: usize = 50,
    max_chunk_size: usize = 2000,
};

/// Stub chunk.
pub const Chunk = struct {
    content: []const u8 = "",
    start_offset: usize = 0,
    end_offset: usize = 0,
    index: usize = 0,
    metadata: ?[]const u8 = null,

    pub fn clone(self: Chunk, allocator: std.mem.Allocator) !Chunk {
        _ = allocator;
        return self;
    }

    pub fn deinit(self: *Chunk, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.* = undefined;
    }
};

/// Stub chunker.
pub const Chunker = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: ChunkerConfig) Chunker {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Chunker) void {
        self.* = undefined;
    }

    pub fn chunk(self: *Chunker, content: []const u8) ![]Chunk {
        _ = self;
        _ = content;
        return error.RagDisabled;
    }
};

/// Stub retriever config.
pub const RetrieverConfig = struct {
    top_k: usize = 5,
    min_score: f32 = 0.3,
    embedding_dim: usize = 384,
};

/// Stub retrieval result.
pub const RetrievalResult = struct {
    chunk: Chunk = .{},
    doc_id: []const u8 = "",
    score: f32 = 0,
    rank: usize = 0,
};

/// Stub retriever.
pub const Retriever = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: RetrieverConfig) Retriever {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Retriever) void {
        self.* = undefined;
    }

    pub fn computeEmbedding(self: *Retriever, text: []const u8) !?[]f32 {
        _ = self;
        _ = text;
        return error.RagDisabled;
    }

    pub fn computeSimilarity(self: *const Retriever, a: []const f32, b: []const f32) f32 {
        _ = self;
        _ = a;
        _ = b;
        return 0;
    }

    pub fn textSimilarity(self: *const Retriever, query: []const u8, text: []const u8) f32 {
        _ = self;
        _ = query;
        _ = text;
        return 0;
    }
};

/// Stub context config.
pub const ContextConfig = struct {
    context_template: []const u8 = "",
    chunk_template: []const u8 = "",
    chunk_separator: []const u8 = "\n\n",
    include_sources: bool = true,
    include_scores: bool = false,
    max_context_length: usize = 10000,
};

/// Stub RAG context.
pub const RagContext = struct {
    prompt: []const u8 = "",
    context_only: []const u8 = "",
    chunks_used: usize = 0,
    total_tokens: usize = 0,
    truncated: bool = false,
    sources: []const SourceRef = &[_]SourceRef{},

    pub const SourceRef = struct {
        doc_id: []const u8 = "",
        chunk_index: usize = 0,
        score: f32 = 0,
    };

    pub fn deinit(self: *RagContext, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.* = undefined;
    }
};

/// Stub context builder.
pub const ContextBuilder = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: ContextConfig) ContextBuilder {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ContextBuilder) void {
        self.* = undefined;
    }

    pub fn build(
        self: *ContextBuilder,
        query: []const u8,
        results: []const RetrievalResult,
        max_tokens: usize,
    ) !RagContext {
        _ = self;
        _ = query;
        _ = results;
        _ = max_tokens;
        return error.RagDisabled;
    }
};

/// Stub RAG config.
pub const RagConfig = struct {
    chunking: ChunkerConfig = .{},
    retrieval: RetrieverConfig = .{},
    context: ContextConfig = .{},
    max_context_tokens: usize = 2000,
    deduplicate: bool = true,
    dedup_threshold: f32 = 0.95,
};

/// Stub RAG response.
pub const RagResponse = struct {
    query: []const u8 = "",
    context: RagContext = .{},
    num_chunks: usize = 0,
    sources: [][]const u8 = &[_][]const u8{},

    pub fn deinit(self: *RagResponse, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.* = undefined;
    }

    pub fn getPrompt(self: *const RagResponse) []const u8 {
        return self.context.prompt;
    }
};

/// Stub RAG pipeline.
pub const RagPipeline = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: RagConfig) RagPipeline {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *RagPipeline) void {
        self.* = undefined;
    }

    pub fn addDocument(self: *RagPipeline, doc: Document) !void {
        _ = self;
        _ = doc;
        return error.RagDisabled;
    }

    pub fn addText(self: *RagPipeline, content: []const u8, title: ?[]const u8) !void {
        _ = self;
        _ = content;
        _ = title;
        return error.RagDisabled;
    }

    pub fn query(self: *RagPipeline, query_text: []const u8, top_k: ?usize) !RagResponse {
        _ = self;
        _ = query_text;
        _ = top_k;
        return error.RagDisabled;
    }

    pub fn chunkCount(self: *const RagPipeline) usize {
        _ = self;
        return 0;
    }

    pub fn documentCount(self: *const RagPipeline) usize {
        _ = self;
        return 0;
    }

    pub fn clear(self: *RagPipeline) void {
        _ = self;
    }
};

/// Stub factory functions.
pub fn createPipeline(allocator: std.mem.Allocator) RagPipeline {
    return RagPipeline.init(allocator, .{});
}

pub fn createPipelineWithConfig(allocator: std.mem.Allocator, config: RagConfig) RagPipeline {
    return RagPipeline.init(allocator, config);
}
