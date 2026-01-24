//! Database Module - Vector Database API
//!
//! This module provides the WDBX vector database for high-performance similarity search.
//! It supports HNSW (Hierarchical Navigable Small World) and IVF-PQ (Inverted File with
//! Product Quantization) indexing algorithms.
//!
//! ## Features
//!
//! - **Vector Storage**: Store and retrieve high-dimensional vectors with metadata
//! - **Similarity Search**: Find similar vectors using cosine, euclidean, or dot product
//! - **Full-text Search**: BM25-based text search with inverted index
//! - **Hybrid Search**: Combine vector and text search with configurable fusion
//! - **Metadata Filtering**: Filter results by metadata attributes
//! - **Batch Operations**: Efficient bulk insert/update/delete
//! - **Clustering**: K-means clustering for data analysis
//! - **Quantization**: Scalar and product quantization for compression
//! - **GPU Acceleration**: Optional GPU-accelerated distance calculations
//!
//! ## Quick Start
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Initialize with database enabled
//! var fw = try abi.Framework.init(allocator, .{
//!     .database = .{ .path = "./vectors.db" },
//! });
//! defer fw.deinit();
//!
//! // Get database context
//! const db_ctx = try fw.getDatabase();
//!
//! // Insert a vector
//! try db_ctx.insertVector(1, &[_]f32{ 0.1, 0.2, 0.3, 0.4 }, "metadata");
//!
//! // Search for similar vectors
//! const results = try db_ctx.searchVectors(&query_vector, 10);
//! defer allocator.free(results);
//! ```
//!
//! ## Standalone Usage
//!
//! ```zig
//! const db = abi.database;
//!
//! // Open or create a database
//! var handle = try db.open(allocator, "vectors.db");
//! defer db.close(&handle);
//!
//! // Insert vectors
//! try db.insert(&handle, 1, &[_]f32{ 0.1, 0.2, 0.3 }, "doc1");
//! try db.insert(&handle, 2, &[_]f32{ 0.4, 0.5, 0.6 }, "doc2");
//!
//! // Search
//! const results = try db.search(&handle, allocator, &query, 5);
//! defer allocator.free(results);
//!
//! for (results) |result| {
//!     std.debug.print("ID: {}, Score: {d}\n", .{ result.id, result.score });
//! }
//! ```
//!
//! ## Advanced Features
//!
//! ### Hybrid Search
//!
//! ```zig
//! var engine = try db.HybridSearchEngine.init(allocator, .{
//!     .vector_weight = 0.7,
//!     .text_weight = 0.3,
//!     .fusion = .rrf,  // Reciprocal Rank Fusion
//! });
//! defer engine.deinit();
//!
//! const results = try engine.search(query_vector, "search text", 10);
//! ```
//!
//! ### Metadata Filtering
//!
//! ```zig
//! var filter = db.FilterBuilder.init()
//!     .eq("category", .{ .string = "tech" })
//!     .gte("year", .{ .int = 2020 })
//!     .build();
//!
//! const results = try db.FilteredSearch.search(&handle, query, filter, 10);
//! ```

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config/mod.zig");

// Re-export shared dependencies for child modules
pub const time = @import("../shared/time.zig");

pub const database = @import("database.zig");
pub const db_helpers = @import("db_helpers.zig");
pub const storage = @import("storage.zig");
pub const storage_v2 = @import("storage_v2.zig");
pub const wdbx = @import("wdbx.zig");
pub const cli = @import("cli.zig");
pub const http = @import("http.zig");
pub const fulltext = @import("fulltext.zig");
pub const hybrid = @import("hybrid.zig");
pub const filter = @import("filter.zig");
pub const batch = @import("batch.zig");
pub const clustering = @import("clustering.zig");
pub const formats = @import("formats/mod.zig");
pub const quantization = @import("quantization.zig");
pub const gpu_accel = @import("gpu_accel.zig");
pub const block_chain = @import("block_chain.zig");
pub const distributed = @import("distributed/mod.zig");

// Novel index structures (research: billion-scale ANN)
pub const diskann = @import("diskann.zig");
pub const scann = @import("scann.zig");

pub const Database = database.Database;
pub const DatabaseHandle = wdbx.DatabaseHandle;
pub const SearchResult = wdbx.SearchResult;
pub const VectorView = wdbx.VectorView;
pub const Stats = wdbx.Stats;

// Full-text search exports
pub const InvertedIndex = fulltext.InvertedIndex;
pub const Bm25Config = fulltext.Bm25Config;
pub const TokenizerConfig = fulltext.TokenizerConfig;
pub const TextSearchResult = fulltext.TextSearchResult;
pub const QueryParser = fulltext.QueryParser;

// Hybrid search exports
pub const HybridSearchEngine = hybrid.HybridSearchEngine;
pub const HybridConfig = hybrid.HybridConfig;
pub const HybridResult = hybrid.HybridResult;
pub const FusionMethod = hybrid.FusionMethod;

// Metadata filter exports
pub const FilterBuilder = filter.FilterBuilder;
pub const FilterExpression = filter.FilterExpression;
pub const FilterCondition = filter.FilterCondition;
pub const FilterOperator = filter.FilterOperator;
pub const MetadataValue = filter.MetadataValue;
pub const MetadataStore = filter.MetadataStore;
pub const FilteredSearch = filter.FilteredSearch;
pub const FilteredResult = filter.FilteredResult;

// Batch operation exports
pub const BatchProcessor = batch.BatchProcessor;
pub const BatchConfig = batch.BatchConfig;
pub const BatchRecord = batch.BatchRecord;
pub const BatchResult = batch.BatchResult;
pub const BatchWriter = batch.BatchWriter;
pub const BatchOperationBuilder = batch.BatchOperationBuilder;
pub const BatchImporter = batch.BatchImporter;

// Clustering exports
pub const KMeans = clustering.KMeans;
pub const ClusterStats = clustering.ClusterStats;
pub const FitOptions = clustering.FitOptions;
pub const FitResult = clustering.FitResult;
pub const euclideanDistance = clustering.euclideanDistance;
pub const cosineSimilarity = clustering.cosineSimilarity;
pub const silhouetteScore = clustering.silhouetteScore;
pub const elbowMethod = clustering.elbowMethod;

// Quantization exports (from academic research: PQ, scalar quantization)
pub const ScalarQuantizer = quantization.ScalarQuantizer;
pub const ProductQuantizer = quantization.ProductQuantizer;
pub const QuantizationError = quantization.QuantizationError;

// GPU acceleration exports
pub const GpuAccelerator = gpu_accel.GpuAccelerator;
pub const GpuAccelConfig = gpu_accel.GpuAccelConfig;
pub const GpuAccelStats = gpu_accel.GpuAccelStats;

// Unified storage format exports
pub const UnifiedFormat = formats.UnifiedFormat;
pub const UnifiedFormatBuilder = formats.unified.UnifiedFormatBuilder;
pub const FormatHeader = formats.FormatHeader;
pub const FormatFlags = formats.FormatFlags;
pub const TensorDescriptor = formats.TensorDescriptor;
pub const DataType = formats.DataType;
pub const Converter = formats.Converter;
pub const ConversionOptions = formats.ConversionOptions;
pub const TargetFormat = formats.TargetFormat;
pub const CompressionType = formats.CompressionType;

// Streaming and mmap exports
pub const StreamingWriter = formats.StreamingWriter;
pub const StreamingReader = formats.StreamingReader;
pub const MappedFile = formats.MappedFile;
pub const MemoryCursor = formats.MemoryCursor;

// Storage v2 exports (industry-standard format)
pub const FileHeader = storage_v2.FileHeader;
pub const FileFooter = storage_v2.FileFooter;
pub const BloomFilter = storage_v2.BloomFilter;
pub const Crc32 = storage_v2.Crc32;
pub const StorageV2Config = storage_v2.StorageV2Config;
pub const saveDatabaseV2 = storage_v2.saveDatabaseV2;
pub const loadDatabaseV2 = storage_v2.loadDatabaseV2;

// Vector database format exports
pub const FormatVectorDatabase = formats.VectorDatabase;
pub const FormatVectorRecord = formats.VectorRecord;
pub const FormatSearchResult = formats.SearchResult;

// GGUF converter exports
pub const fromGguf = formats.fromGguf;
pub const toGguf = formats.toGguf;
pub const GgufTensorType = formats.GgufTensorType;

// ZON format exports (Zig 0.16 native serialization for WDBX)
pub const ZonFormat = formats.ZonFormat;
pub const ZonDatabase = formats.ZonDatabase;
pub const ZonRecord = formats.ZonRecord;
pub const ZonDatabaseConfig = formats.ZonDatabaseConfig;
pub const ZonDistanceMetric = formats.ZonDistanceMetric;
pub const exportToZon = formats.exportToZon;
pub const importFromZon = formats.importFromZon;
pub const ImportFormat = batch.ImportFormat;

// Distributed WDBX exports (block chain, sharding, consensus)
pub const BlockChain = block_chain.BlockChain;
pub const ConversationBlock = block_chain.ConversationBlock;
pub const BlockChainConfig = block_chain.BlockChainConfig;
pub const BlockChainError = block_chain.BlockChainError;
pub const PersonaTag = block_chain.PersonaTag;
pub const RoutingWeights = block_chain.RoutingWeights;
pub const IntentCategory = block_chain.IntentCategory;
pub const PolicyFlags = block_chain.PolicyFlags;

pub const ShardManager = distributed.ShardManager;
pub const ShardConfig = distributed.ShardConfig;
pub const ShardKey = distributed.ShardKey;
pub const ShardManagerError = distributed.ShardManagerError;
pub const HashRing = distributed.HashRing;
pub const LoadStats = distributed.LoadStats;

// DiskANN exports (billion-scale disk-based ANN)
pub const DiskANNIndex = diskann.DiskANNIndex;
pub const DiskANNConfig = diskann.DiskANNConfig;
pub const PQCodebook = diskann.PQCodebook;
pub const DiskANNStats = diskann.IndexStats;

// ScaNN exports (learned quantization for ANN)
pub const ScaNNIndex = scann.ScaNNIndex;
pub const ScaNNConfig = scann.ScaNNConfig;
pub const QuantizationType = scann.QuantizationType;
pub const AVQCodebook = scann.AVQCodebook;
pub const ScaNNStats = scann.IndexStats;

pub const BlockExchangeManager = distributed.BlockExchangeManager;
pub const BlockExchangeError = distributed.BlockExchangeError;
pub const SyncState = distributed.SyncState;
pub const VersionVector = distributed.VersionVector;
pub const VersionComparison = distributed.VersionComparison;
pub const SyncRequest = distributed.SyncRequest;
pub const SyncResponse = distributed.SyncResponse;
pub const BlockConflict = distributed.BlockConflict;

pub const DistributedBlockChain = distributed.DistributedBlockChain;
pub const DistributedBlockChainConfig = distributed.DistributedBlockChainConfig;
pub const DistributedBlockChainError = distributed.DistributedBlockChainError;
pub const DistributedConfig = distributed.DistributedConfig;
pub const DistributedContext = distributed.Context;

pub const DatabaseFeatureError = error{
    DatabaseDisabled,
};

/// Database Context for Framework integration.
///
/// The Context struct provides a high-level interface for database operations,
/// managing the database handle and providing convenient methods for common
/// operations like inserting and searching vectors.
///
/// ## Thread Safety
///
/// The Context is not thread-safe. For concurrent access, use external
/// synchronization or create separate Context instances.
///
/// ## Auto-open Behavior
///
/// If a path is provided in the configuration, the database will be automatically
/// opened during initialization. If no path is provided, the database must be
/// explicitly opened using `openDatabase()`.
///
/// ## Example
///
/// ```zig
/// var ctx = try Context.init(allocator, .{ .path = "./vectors.db" });
/// defer ctx.deinit();
///
/// // Insert vectors
/// try ctx.insertVector(1, &[_]f32{ 0.1, 0.2, 0.3 }, "metadata");
///
/// // Search
/// const results = try ctx.searchVectors(&query, 10);
/// defer allocator.free(results);
/// ```
pub const Context = struct {
    /// Memory allocator for database operations.
    allocator: std.mem.Allocator,
    /// Database configuration.
    config: config_module.DatabaseConfig,
    /// Database handle, or null if not yet opened.
    handle: ?DatabaseHandle = null,

    /// Initialize the database context.
    ///
    /// ## Parameters
    ///
    /// - `allocator`: Memory allocator for database operations
    /// - `cfg`: Database configuration (path, index settings, etc.)
    ///
    /// ## Returns
    ///
    /// A pointer to the initialized Context.
    ///
    /// ## Errors
    ///
    /// - `error.DatabaseDisabled`: Database feature is disabled at compile time
    /// - `error.OutOfMemory`: Memory allocation failed
    pub fn init(allocator: std.mem.Allocator, cfg: config_module.DatabaseConfig) !*Context {
        if (!isEnabled()) return error.DatabaseDisabled;

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
            .handle = null,
        };

        // Auto-open database if path is provided.
        if (cfg.path.len > 0) {
            ctx.handle = try wdbx.createDatabase(allocator, cfg.path);
        }

        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.handle) |*h| {
            wdbx.closeDatabase(h);
        }
        self.allocator.destroy(self);
    }

    /// Get or create the database handle.
    pub fn getHandle(self: *Context) !*DatabaseHandle {
        if (self.handle) |*h| {
            return h;
        }
        self.handle = try wdbx.createDatabase(self.allocator, self.config.path);
        return &self.handle.?;
    }

    /// Open a database and attach it to this Context.
    /// If a database is already open, it is closed first.
    /// The returned handle is owned by the Context; do not close it directly.
    pub fn openDatabase(self: *Context, name: []const u8) !*DatabaseHandle {
        if (self.handle) |*h| {
            wdbx.closeDatabase(h);
            self.handle = null;
        }
        self.handle = try wdbx.createDatabase(self.allocator, name);
        return &self.handle.?;
    }

    /// Insert a vector into the database.
    pub fn insertVector(self: *Context, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
        const h = try self.getHandle();
        try wdbx.insertVector(h, id, vector, metadata);
    }

    /// Search for similar vectors.
    pub fn searchVectors(self: *Context, query: []const f32, top_k: usize) ![]SearchResult {
        const h = try self.getHandle();
        return wdbx.searchVectors(h, self.allocator, query, top_k);
    }

    /// Get database statistics.
    pub fn getStats(self: *Context) !Stats {
        const h = try self.getHandle();
        return wdbx.getStats(h);
    }

    /// Optimize the database index.
    pub fn optimize(self: *Context) !void {
        const h = try self.getHandle();
        try wdbx.optimize(h);
    }
};

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    if (!isEnabled()) return DatabaseFeatureError.DatabaseDisabled;
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return build_options.enable_database;
}

pub fn isInitialized() bool {
    return initialized;
}

pub fn open(allocator: std.mem.Allocator, name: []const u8) !DatabaseHandle {
    return wdbx.createDatabase(allocator, name);
}

pub fn connect(allocator: std.mem.Allocator, name: []const u8) !DatabaseHandle {
    return wdbx.connectDatabase(allocator, name);
}

pub fn close(handle: *DatabaseHandle) void {
    wdbx.closeDatabase(handle);
}

pub fn insert(handle: *DatabaseHandle, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
    try wdbx.insertVector(handle, id, vector, metadata);
}

pub fn search(
    handle: *DatabaseHandle,
    allocator: std.mem.Allocator,
    query: []const f32,
    top_k: usize,
) ![]SearchResult {
    return wdbx.searchVectors(handle, allocator, query, top_k);
}

pub fn remove(handle: *DatabaseHandle, id: u64) bool {
    return wdbx.deleteVector(handle, id);
}

pub fn update(handle: *DatabaseHandle, id: u64, vector: []const f32) !bool {
    return wdbx.updateVector(handle, id, vector);
}

pub fn get(handle: *DatabaseHandle, id: u64) ?VectorView {
    return wdbx.getVector(handle, id);
}

pub fn list(handle: *DatabaseHandle, allocator: std.mem.Allocator, limit: usize) ![]VectorView {
    return wdbx.listVectors(handle, allocator, limit);
}

pub fn stats(handle: *DatabaseHandle) Stats {
    return wdbx.getStats(handle);
}

pub fn optimize(handle: *DatabaseHandle) !void {
    try wdbx.optimize(handle);
}

pub fn backup(handle: *DatabaseHandle, path: []const u8) !void {
    try wdbx.backup(handle, path);
}

pub fn restore(handle: *DatabaseHandle, path: []const u8) !void {
    try wdbx.restore(handle, path);
}

pub fn openFromFile(allocator: std.mem.Allocator, path: []const u8) !DatabaseHandle {
    const db = try storage.loadDatabase(allocator, path);
    return .{ .db = db };
}

pub fn openOrCreate(allocator: std.mem.Allocator, path: []const u8) !DatabaseHandle {
    const loaded = storage.loadDatabase(allocator, path);
    if (loaded) |db| {
        return .{ .db = db };
    } else |err| switch (err) {
        error.FileNotFound => return wdbx.createDatabase(allocator, path),
        else => return err,
    }
}

test "database module init gating" {
    if (!isEnabled()) return;
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}
