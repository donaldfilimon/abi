//! RAG pipeline stub — disabled at compile time.

const std = @import("std");

// --- Types ---

pub const DocumentType = enum { text, markdown, html, code, pdf, json, other };

pub const DocumentMetadata = struct {
    source: ?[]const u8 = null,
    author: ?[]const u8 = null,
    created_at: ?i64 = null,
    modified_at: ?i64 = null,
    language: ?[]const u8 = null,
    tags: ?[]const []const u8 = null,
    custom: ?[]const u8 = null,
};

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
    pub fn clone(self: Document, _: std.mem.Allocator) !Document {
        return self;
    }
    pub fn deinit(self: *Document, _: std.mem.Allocator) void {
        self.* = undefined;
    }
};

pub const ChunkingStrategy = enum { fixed, sentence, paragraph, recursive, semantic };

pub const ChunkerConfig = struct {
    strategy: ChunkingStrategy = .recursive,
    chunk_size: usize = 500,
    chunk_overlap: usize = 50,
    min_chunk_size: usize = 50,
    max_chunk_size: usize = 2000,
};

pub const Chunk = struct {
    content: []const u8 = "",
    start_offset: usize = 0,
    end_offset: usize = 0,
    index: usize = 0,
    metadata: ?[]const u8 = null,
    pub fn clone(self: Chunk, _: std.mem.Allocator) !Chunk {
        return self;
    }
    pub fn deinit(self: *Chunk, _: std.mem.Allocator) void {
        self.* = undefined;
    }
};

pub const Chunker = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: ChunkerConfig) Chunker {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *Chunker) void {
        self.* = undefined;
    }
    pub fn chunk(_: *Chunker, _: []const u8) ![]Chunk {
        return error.FeatureDisabled;
    }
};

// --- Retriever ---

pub const RetrieverConfig = struct { top_k: usize = 5, min_score: f32 = 0.3, embedding_dim: usize = 384 };

pub const RetrievalResult = struct { chunk: Chunk = .{}, doc_id: []const u8 = "", score: f32 = 0, rank: usize = 0 };

pub const Retriever = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: RetrieverConfig) Retriever {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *Retriever) void {
        self.* = undefined;
    }
    pub fn computeEmbedding(_: *Retriever, _: []const u8) !?[]f32 {
        return error.FeatureDisabled;
    }
    pub fn computeSimilarity(_: *const Retriever, _: []const f32, _: []const f32) f32 {
        return 0;
    }
    pub fn textSimilarity(_: *const Retriever, _: []const u8, _: []const u8) f32 {
        return 0;
    }
};

// --- Context Builder ---

pub const ContextConfig = struct {
    context_template: []const u8 = "",
    chunk_template: []const u8 = "",
    chunk_separator: []const u8 = "\n\n",
    include_sources: bool = true,
    include_scores: bool = false,
    max_context_length: usize = 10000,
};

pub const RagContext = struct {
    prompt: []const u8 = "",
    context_only: []const u8 = "",
    chunks_used: usize = 0,
    total_tokens: usize = 0,
    truncated: bool = false,
    sources: []const SourceRef = &[_]SourceRef{},
    pub const SourceRef = struct { doc_id: []const u8 = "", chunk_index: usize = 0, score: f32 = 0 };
    pub fn deinit(self: *RagContext, _: std.mem.Allocator) void {
        self.* = undefined;
    }
};

pub const ContextBuilder = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: ContextConfig) ContextBuilder {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *ContextBuilder) void {
        self.* = undefined;
    }
    pub fn build(_: *ContextBuilder, _: []const u8, _: []const RetrievalResult, _: usize) !RagContext {
        return error.FeatureDisabled;
    }
};

// --- Pipeline ---

pub const RagConfig = struct {
    chunking: ChunkerConfig = .{},
    retrieval: RetrieverConfig = .{},
    context: ContextConfig = .{},
    max_context_tokens: usize = 2000,
    deduplicate: bool = true,
    dedup_threshold: f32 = 0.95,
};

pub const RagResponse = struct {
    query: []const u8 = "",
    context: RagContext = .{},
    num_chunks: usize = 0,
    sources: [][]const u8 = &[_][]const u8{},
    pub fn deinit(self: *RagResponse, _: std.mem.Allocator) void {
        self.* = undefined;
    }
    pub fn getPrompt(self: *const RagResponse) []const u8 {
        return self.context.prompt;
    }
};

pub const RagPipeline = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: RagConfig) RagPipeline {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *RagPipeline) void {
        self.* = undefined;
    }
    pub fn addDocument(_: *RagPipeline, _: Document) !void {
        return error.FeatureDisabled;
    }
    pub fn addText(_: *RagPipeline, _: []const u8, _: ?[]const u8) !void {
        return error.FeatureDisabled;
    }
    pub fn query(_: *RagPipeline, _: []const u8, _: ?usize) !RagResponse {
        return error.FeatureDisabled;
    }
    pub fn chunkCount(_: *const RagPipeline) usize {
        return 0;
    }
    pub fn documentCount(_: *const RagPipeline) usize {
        return 0;
    }
    pub fn clear(_: *RagPipeline) void {}
};

// --- Factory ---

pub fn createPipeline(allocator: std.mem.Allocator) RagPipeline {
    return RagPipeline.init(allocator, .{});
}
pub fn createPipelineWithConfig(allocator: std.mem.Allocator, config: RagConfig) RagPipeline {
    return RagPipeline.init(allocator, config);
}
