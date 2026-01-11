//! Stub implementation for embeddings when feature is disabled.
//!
//! Provides API-compatible stubs that return EmbeddingsDisabled error.

const std = @import("std");

pub const EmbeddingsError = error{
    EmbeddingsDisabled,
};

pub const EmbedderConfig = struct {
    dimension: u32 = 384,
    max_seq_length: u32 = 512,
    normalize: bool = true,
    model_id: []const u8 = "default",
    enable_cache: bool = true,
    cache_max_entries: usize = 10000,
    batch_size: usize = 32,
};

pub const EmbeddingResult = struct {
    vector: []f32,
    text_hash: u64,
    from_cache: bool,

    pub fn deinit(_: *EmbeddingResult, _: std.mem.Allocator) void {}
};

pub const BatchEmbeddingResponse = struct {
    embeddings: []EmbeddingResult,
    total_processed: usize,
    cache_hits: usize,
    processing_time_ns: u64,

    pub fn deinit(_: *BatchEmbeddingResponse, _: std.mem.Allocator) void {}
};

pub const SimilarityResult = struct {
    index: usize,
    score: f32,
    text: ?[]const u8,
};

pub const Embedder = struct {
    allocator: std.mem.Allocator,
    config: EmbedderConfig,

    pub fn init(_: std.mem.Allocator, _: EmbedderConfig) !Embedder {
        return EmbeddingsError.EmbeddingsDisabled;
    }

    pub fn deinit(_: *Embedder) void {}

    pub fn embed(_: *Embedder, _: []const u8) !EmbeddingResult {
        return EmbeddingsError.EmbeddingsDisabled;
    }

    pub fn embedBatch(_: *Embedder, _: []const []const u8) !BatchEmbeddingResponse {
        return EmbeddingsError.EmbeddingsDisabled;
    }

    pub fn findSimilar(
        _: *Embedder,
        _: []const u8,
        _: []const []const u8,
        _: usize,
    ) ![]SimilarityResult {
        return EmbeddingsError.EmbeddingsDisabled;
    }

    pub fn clearCache(_: *Embedder) void {}
    pub fn getCacheStats(_: *const Embedder) ?CacheStats {
        return null;
    }
};

pub const BatchConfig = struct {
    batch_size: usize = 32,
    dimension: u32 = 384,
    normalize: bool = true,
    max_seq_length: usize = 512,
    pad_to_max_length: bool = false,
};

pub const BatchResult = struct {
    batch_index: usize,
    batch_size: usize,
    embeddings: [][]f32,
    processing_time_ns: u64,
};

pub const BatchProcessor = struct {
    allocator: std.mem.Allocator,
    config: BatchConfig,

    pub fn init(allocator: std.mem.Allocator, config: BatchConfig) BatchProcessor {
        return .{ .allocator = allocator, .config = config };
    }

    pub fn deinit(_: *BatchProcessor) void {}

    pub fn process(_: *BatchProcessor, _: []const []const u8) ![][]f32 {
        return EmbeddingsError.EmbeddingsDisabled;
    }
};

pub const CacheConfig = struct {
    max_entries: usize = 10000,
    dimension: u32 = 384,
    collect_stats: bool = true,
};

pub const CacheStats = struct {
    total_lookups: u64,
    hits: u64,
    misses: u64,
    current_entries: usize,
    evictions: u64,
    memory_bytes: usize,

    pub fn hitRatio(_: CacheStats) f64 {
        return 0;
    }
};

pub const EmbeddingCache = struct {
    allocator: std.mem.Allocator,
    config: CacheConfig,

    pub fn init(allocator: std.mem.Allocator, config: CacheConfig) EmbeddingCache {
        return .{ .allocator = allocator, .config = config };
    }

    pub fn deinit(_: *EmbeddingCache) void {}

    pub fn get(_: *EmbeddingCache, _: u64) ?[]const f32 {
        return null;
    }

    pub fn put(_: *EmbeddingCache, _: u64, _: []const f32) !void {
        return EmbeddingsError.EmbeddingsDisabled;
    }

    pub fn remove(_: *EmbeddingCache, _: u64) bool {
        return false;
    }

    pub fn clear(_: *EmbeddingCache) void {}

    pub fn getStats(_: *const EmbeddingCache) CacheStats {
        return .{
            .total_lookups = 0,
            .hits = 0,
            .misses = 0,
            .current_entries = 0,
            .evictions = 0,
            .memory_bytes = 0,
        };
    }

    pub fn contains(_: *const EmbeddingCache, _: u64) bool {
        return false;
    }

    pub fn count(_: *const EmbeddingCache) usize {
        return 0;
    }
};

pub fn computeTextHash(text: []const u8) u64 {
    return std.hash.Wyhash.hash(0, text);
}

pub fn cosineSimilarity(_: []const f32, _: []const f32) f32 {
    return 0;
}

pub fn euclideanDistance(_: []const f32, _: []const f32) f32 {
    return 0;
}

pub fn normalize(_: []f32) void {}
