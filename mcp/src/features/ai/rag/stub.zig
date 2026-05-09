//! RAG pipeline stub — disabled at compile time.

const std = @import("std");
const types = @import("types.zig");

// Re-export types
pub const DocumentType = types.DocumentType;
pub const DocumentMetadata = types.DocumentMetadata;
pub const Document = types.Document;
pub const ChunkingStrategy = types.ChunkingStrategy;
pub const ChunkerConfig = types.ChunkerConfig;
pub const Chunk = types.Chunk;
pub const RetrieverConfig = types.RetrieverConfig;
pub const RetrievalResult = types.RetrievalResult;
pub const ContextConfig = types.ContextConfig;
pub const RagContext = types.RagContext;
pub const RagConfig = types.RagConfig;
pub const RagResponse = types.RagResponse;

// --- Chunker ---
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

test {
    std.testing.refAllDecls(@This());
}
