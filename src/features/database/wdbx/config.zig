//! Configuration for the WDBX Neural Vector Search Platform.
//!
//! Exposes all tuning parameters for dimensional structures, max capacities,
//! HNSW network depths, quantization, cache dimensions, and distance metrics.

const std = @import("std");

pub const DistanceMetric = enum {
    cosine,
    euclidean,
    dot_product,
    manhattan,

    pub fn requiresNormalization(self: DistanceMetric) bool {
        return self == .cosine;
    }
};

pub const Config = struct {
    // ═══════════════════════════════════════════════════════════
    // Vector Configuration
    // ═══════════════════════════════════════════════════════════

    /// Vector dimensionality
    /// Must match embedding model output
    /// Common values: 384, 768, 1536, 3072
    /// Range: [1, 16384]
    dimensions: comptime_int = 768,

    /// Maximum vectors the index can hold
    /// Preallocates index structure. Can be grown dynamically with reindexing
    max_vectors: u32 = 10_000_000,

    // ═══════════════════════════════════════════════════════════
    // Distance Metric
    // ═══════════════════════════════════════════════════════════

    /// Similarity metric for search
    /// - .cosine: Best for text (requires normalization)
    /// - .euclidean: Best for unnormalized vectors
    /// - .dot_product: Maximum inner product search
    /// - .manhattan: L1 distance
    metric: DistanceMetric = .cosine,

    // ═══════════════════════════════════════════════════════════
    // Index Configuration
    // ═══════════════════════════════════════════════════════════

    index: IndexConfig = .{},

    pub const IndexAlgorithm = enum { hnsw, ivf, flat };

    pub const IndexConfig = struct {
        algorithm: IndexAlgorithm = .hnsw,

        /// HNSW: Max connections per node (except layer 0)
        /// Higher = more memory, better recall
        /// Typical: 16-64
        hnsw_m: u16 = 32,

        /// HNSW: Build-time search depth
        /// Higher = better graph quality, slower build
        /// Typical: 100-500
        hnsw_ef_construction: u16 = 200,

        /// HNSW: Default query-time search depth
        /// Dynamically overridable per query
        /// Higher = better recall, higher latency
        /// Typical: 32-256
        hnsw_ef_search: u16 = 64,

        /// Enable connection pruning during build
        /// Improves graph quality but slower insertion
        enable_pruning: bool = true,

        /// Enable graph compression (delta encoding)
        /// Reduces memory ~20% with minimal performance impact
        enable_compression: bool = true,
    };

    // ═══════════════════════════════════════════════════════════
    // Cache Configuration
    // ═══════════════════════════════════════════════════════════

    cache: CacheConfig = .{},

    pub const CacheConfig = struct {
        /// Total cache capacity (number of embeddings)
        capacity: usize = 100_000,

        /// Number of cache segments (reduces lock contention)
        /// Must be power of 2
        /// Typical: 8-64
        segments: u8 = 32,

        /// Time-to-live for cache entries (seconds)
        ttl_seconds: u64 = 3600,

        /// Enable Bloom filter for negative lookup optimization
        /// Reduces cache misses by ~50% with 1% memory overhead
        enable_bloom_filter: bool = true,

        /// Bloom filter false positive rate
        /// Lower = more memory, fewer false positives
        bloom_false_positive_rate: f32 = 0.01,
    };

    // ═══════════════════════════════════════════════════════════
    // Memory Configuration
    // ═══════════════════════════════════════════════════════════

    memory: MemoryConfig = .{},

    pub const MemoryConfig = struct {
        /// Enable memory pooling for common vector sizes
        /// Reduces allocation overhead ~30%
        enable_pooling: bool = true,

        /// Pool sizes (dimensions)
        /// Preallocates buffers for these sizes
        pool_sizes: []const usize = &.{ 256, 512, 768, 1024, 1536, 2048 },

        /// Arena allocator block size
        /// Larger = fewer system allocations, more waste
        arena_block_size: usize = 4 * 1024 * 1024, // 4MB

        /// Enable vector quantization
        /// Compresses vectors 4-32x with 1-3% accuracy loss
        enable_quantization: bool = false,

        /// Quantization precision (bits per dimension)
        /// 8-bit: 4× compression, 1% accuracy loss
        /// 4-bit: 8× compression, 2% accuracy loss
        quantization_bits: u8 = 8,
    };

    // ═══════════════════════════════════════════════════════════
    // Network Configuration
    // ═══════════════════════════════════════════════════════════

    network: NetworkConfig = .{},

    pub const NetworkConfig = struct {
        /// Maximum concurrent network connections
        max_connections: u16 = 10_000,

        /// Connection timeout (milliseconds)
        connection_timeout_ms: u32 = 5_000,

        /// Request timeout (milliseconds)
        request_timeout_ms: u32 = 30_000,

        /// Enable HTTP compression (gzip)
        enable_compression: bool = true,

        /// Enable TLS for HTTPS
        enable_tls: bool = true,
    };

    // ═══════════════════════════════════════════════════════════
    // Compile-Time Validation
    // ═══════════════════════════════════════════════════════════

    /// Validates configuration at compile time
    pub fn validate(comptime self: Config) void {
        if (self.dimensions < 1 or self.dimensions > 16384) {
            @compileError("Dimensions must be between 1 and 16384");
        }
        if (self.cache.segments == 0 or
            (self.cache.segments & (self.cache.segments - 1)) != 0) {
            @compileError("Cache segments must be power of 2");
        }
        if (self.memory.quantization_bits != 1 and
            self.memory.quantization_bits != 2 and
            self.memory.quantization_bits != 4 and
            self.memory.quantization_bits != 8) {
            @compileError("Quantization bits must be 1, 2, 4, or 8");
        }
    }
};

test "Config Defaults Compile" {
    const default = Config{};
    default.validate();
}
