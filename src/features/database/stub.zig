//! Database Stub Module
//!
//! This module provides API-compatible no-op implementations for all public
//! database functions when the database feature is disabled at compile time.
//! All functions return `error.DatabaseDisabled` or empty/default values as
//! appropriate. Types are kept minimal - only essential ones needed for
//! compile-time checking.
//!
//! The database module encompasses:
//! - WDBX vector database with HNSW/IVF-PQ indexing
//! - Parallel search and batch operations
//! - Full-text search and hybrid queries
//! - Clustering and quantization
//! - Backup, restore, and persistence
//! - HTTP server for remote access
//!
//! To enable the real implementation, build with `-Denable-database=true`.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");

// ============================================================================
// Error Types
// ============================================================================

pub const DatabaseFeatureError = error{DatabaseDisabled};
pub const DatabaseError = error{
    DuplicateId,
    VectorNotFound,
    InvalidDimension,
    PoolExhausted,
    PersistenceError,
    ConcurrencyError,
    DatabaseDisabled,
};

// ============================================================================
// Local Stubs Imports
// ============================================================================

const types = @import("stubs/types.zig");
const wdbx_mod = @import("stubs/wdbx.zig");
const parallel = @import("stubs/parallel.zig");
const misc = @import("stubs/misc.zig");

// ============================================================================
// Core Types Re-exports
// ============================================================================

pub const DatabaseHandle = types.DatabaseHandle;
pub const SearchResult = types.SearchResult;
pub const VectorView = types.VectorView;
pub const Stats = types.Stats;
pub const BatchItem = types.BatchItem;
pub const DiagnosticsInfo = types.DiagnosticsInfo;

// ============================================================================
// Context (Framework integration)
// ============================================================================

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.DatabaseConfig,
    handle: ?DatabaseHandle = null,

    pub fn init(_: std.mem.Allocator, _: config_module.DatabaseConfig) !*Context {
        return error.DatabaseDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn getHandle(_: *Context) !*DatabaseHandle {
        return error.DatabaseDisabled;
    }
    pub fn openDatabase(_: *Context, _: []const u8) !*DatabaseHandle {
        return error.DatabaseDisabled;
    }
    pub fn insertVector(_: *Context, _: u64, _: []const f32, _: ?[]const u8) !void {
        return error.DatabaseDisabled;
    }
    pub fn searchVectors(_: *Context, _: []const f32, _: usize) ![]SearchResult {
        return error.DatabaseDisabled;
    }
    pub fn getStats(_: *Context) !Stats {
        return error.DatabaseDisabled;
    }
    pub fn optimize(_: *Context) !void {
        return error.DatabaseDisabled;
    }
};

// ============================================================================
// Sub-module Namespace Stubs Re-exports
// ============================================================================

pub const wdbx = wdbx_mod.wdbx;
pub const cli = misc.cli;

// Parallel search stubs
pub const ParallelSearchConfig = parallel.ParallelSearchConfig;
pub const ParallelSearchExecutor = parallel.ParallelSearchExecutor;
pub const ParallelBeamState = parallel.ParallelBeamState;
pub const ParallelWorkQueue = parallel.ParallelWorkQueue;
pub const BatchSearchResult = parallel.BatchSearchResult;
pub const ParallelSearchStats = parallel.ParallelSearchStats;
pub const batchCosineDistances = parallel.batchCosineDistances;

// Empty namespace stubs for sub-modules
pub const parallel_search = misc.parallel_search;
pub const database = misc.database;
pub const db_helpers = misc.db_helpers;
pub const storage = misc.storage;
pub const http = misc.http;
pub const fulltext = misc.fulltext;
pub const hybrid = misc.hybrid;
pub const filter = misc.filter;
pub const batch = misc.batch;
pub const clustering = misc.clustering;
pub const quantization = misc.quantization;
pub const formats = misc.formats;
pub const gpu_accel = misc.gpu_accel;
pub const block_chain = misc.block_chain;
pub const distributed = misc.distributed;
pub const diskann = misc.diskann;
pub const scann = misc.scann;
pub const parallel_hnsw = misc.parallel_hnsw;
pub const hnsw = misc.hnsw;
pub const index = misc.index;
pub const search_state = misc.search_state;
pub const distance_cache = misc.distance_cache;
pub const storage_v2 = misc.storage_v2;
pub const time = misc.time;

// ============================================================================
// Database type re-export
// ============================================================================

pub const Database = misc.database.Database;

// ============================================================================
// Full-text search type re-exports
// ============================================================================

pub const InvertedIndex = misc.fulltext.InvertedIndex;
pub const Bm25Config = misc.fulltext.Bm25Config;
pub const TokenizerConfig = misc.fulltext.TokenizerConfig;
pub const TextSearchResult = misc.fulltext.TextSearchResult;
pub const QueryParser = misc.fulltext.QueryParser;

// ============================================================================
// Hybrid search type re-exports
// ============================================================================

pub const HybridSearchEngine = misc.hybrid.HybridSearchEngine;
pub const HybridConfig = misc.hybrid.HybridConfig;
pub const HybridResult = misc.hybrid.HybridResult;
pub const FusionMethod = misc.hybrid.FusionMethod;

// ============================================================================
// Filter type re-exports
// ============================================================================

pub const FilterBuilder = misc.filter.FilterBuilder;
pub const FilterExpression = misc.filter.FilterExpression;
pub const FilterCondition = misc.filter.FilterCondition;
pub const FilterOperator = misc.filter.FilterOperator;
pub const MetadataValue = misc.filter.MetadataValue;
pub const MetadataStore = misc.filter.MetadataStore;
pub const FilteredSearch = misc.filter.FilteredSearch;
pub const FilteredResult = misc.filter.FilteredResult;

// ============================================================================
// Batch type re-exports
// ============================================================================

pub const BatchProcessor = misc.batch.BatchProcessor;
pub const BatchConfig = misc.batch.BatchConfig;
pub const BatchRecord = misc.batch.BatchRecord;
pub const BatchResult = misc.batch.BatchResult;
pub const BatchWriter = misc.batch.BatchWriter;
pub const BatchOperationBuilder = misc.batch.BatchOperationBuilder;
pub const BatchImporter = misc.batch.BatchImporter;
pub const ImportFormat = misc.batch.ImportFormat;

// ============================================================================
// Clustering type re-exports
// ============================================================================

pub const KMeans = misc.clustering.KMeans;
pub const ClusterStats = misc.clustering.ClusterStats;
pub const FitOptions = misc.clustering.FitOptions;
pub const FitResult = misc.clustering.FitResult;
pub const euclideanDistance = misc.clustering.euclideanDistance;
pub const cosineSimilarity = misc.clustering.cosineSimilarity;
pub const silhouetteScore = misc.clustering.silhouetteScore;
pub const elbowMethod = misc.clustering.elbowMethod;

// ============================================================================
// Quantization type re-exports
// ============================================================================

pub const ScalarQuantizer = misc.quantization.ScalarQuantizer;
pub const ProductQuantizer = misc.quantization.ProductQuantizer;
pub const QuantizationError = misc.quantization.QuantizationError;

// ============================================================================
// GPU acceleration type re-exports
// ============================================================================

pub const GpuAccelerator = misc.gpu_accel.GpuAccelerator;
pub const GpuAccelConfig = misc.gpu_accel.GpuAccelConfig;
pub const GpuAccelStats = misc.gpu_accel.GpuAccelStats;

// ============================================================================
// Formats type re-exports
// ============================================================================

pub const UnifiedFormat = misc.formats.UnifiedFormat;
pub const UnifiedFormatBuilder = misc.formats.unified.UnifiedFormatBuilder;
pub const FormatHeader = misc.formats.FormatHeader;
pub const FormatFlags = misc.formats.FormatFlags;
pub const TensorDescriptor = misc.formats.TensorDescriptor;
pub const DataType = misc.formats.DataType;
pub const Converter = misc.formats.Converter;
pub const ConversionOptions = misc.formats.ConversionOptions;
pub const TargetFormat = misc.formats.TargetFormat;
pub const CompressionType = misc.formats.CompressionType;
pub const StreamingWriter = misc.formats.StreamingWriter;
pub const StreamingReader = misc.formats.StreamingReader;
pub const MappedFile = misc.formats.MappedFile;
pub const MemoryCursor = misc.formats.MemoryCursor;
pub const FormatVectorDatabase = misc.formats.VectorDatabase;
pub const FormatVectorRecord = misc.formats.VectorRecord;
pub const FormatSearchResult = misc.formats.SearchResult;
pub const fromGguf = misc.formats.fromGguf;
pub const toGguf = misc.formats.toGguf;
pub const GgufTensorType = misc.formats.GgufTensorType;

// ZON format re-exports
pub const ZonFormat = misc.formats.ZonFormat;
pub const ZonDatabase = misc.formats.ZonDatabase;
pub const ZonRecord = misc.formats.ZonRecord;
pub const ZonDatabaseConfig = misc.formats.ZonDatabaseConfig;
pub const ZonDistanceMetric = misc.formats.ZonDistanceMetric;
pub const exportToZon = misc.formats.exportToZon;
pub const importFromZon = misc.formats.importFromZon;

// ============================================================================
// Storage v2 type re-exports
// ============================================================================

pub const FileHeader = misc.storage_v2.FileHeader;
pub const FileFooter = misc.storage_v2.FileFooter;
pub const BloomFilter = misc.storage_v2.BloomFilter;
pub const Crc32 = misc.storage_v2.Crc32;
pub const StorageV2Config = misc.storage_v2.StorageV2Config;
pub const saveDatabaseV2 = misc.storage_v2.saveDatabaseV2;
pub const loadDatabaseV2 = misc.storage_v2.loadDatabaseV2;

// ============================================================================
// BlockChain type re-exports
// ============================================================================

pub const BlockChain = misc.block_chain.BlockChain;
pub const ConversationBlock = misc.block_chain.ConversationBlock;
pub const BlockChainConfig = misc.block_chain.BlockChainConfig;
pub const BlockChainError = misc.block_chain.BlockChainError;
pub const PersonaTag = misc.block_chain.PersonaTag;
pub const RoutingWeights = misc.block_chain.RoutingWeights;
pub const IntentCategory = misc.block_chain.IntentCategory;
pub const PolicyFlags = misc.block_chain.PolicyFlags;

// ============================================================================
// Distributed type re-exports
// ============================================================================

pub const ShardManager = misc.distributed.ShardManager;
pub const ShardConfig = misc.distributed.ShardConfig;
pub const ShardKey = misc.distributed.ShardKey;
pub const ShardManagerError = misc.distributed.ShardManagerError;
pub const HashRing = misc.distributed.HashRing;
pub const LoadStats = misc.distributed.LoadStats;
pub const BlockExchangeManager = misc.distributed.BlockExchangeManager;
pub const BlockExchangeError = misc.distributed.BlockExchangeError;
pub const SyncState = misc.distributed.SyncState;
pub const VersionVector = misc.distributed.VersionVector;
pub const VersionComparison = misc.distributed.VersionComparison;
pub const SyncRequest = misc.distributed.SyncRequest;
pub const SyncResponse = misc.distributed.SyncResponse;
pub const BlockConflict = misc.distributed.BlockConflict;
pub const DistributedBlockChain = misc.distributed.DistributedBlockChain;
pub const DistributedBlockChainConfig = misc.distributed.DistributedBlockChainConfig;
pub const DistributedBlockChainError = misc.distributed.DistributedBlockChainError;
pub const DistributedConfig = misc.distributed.DistributedConfig;
pub const DistributedContext = misc.distributed.Context;

// ============================================================================
// DiskANN type re-exports
// ============================================================================

pub const DiskANNIndex = misc.diskann.DiskANNIndex;
pub const DiskANNConfig = misc.diskann.DiskANNConfig;
pub const PQCodebook = misc.diskann.PQCodebook;
pub const DiskANNStats = misc.diskann.IndexStats;

// ============================================================================
// ScaNN type re-exports
// ============================================================================

pub const ScaNNIndex = misc.scann.ScaNNIndex;
pub const ScaNNConfig = misc.scann.ScaNNConfig;
pub const QuantizationType = misc.scann.QuantizationType;
pub const AVQCodebook = misc.scann.AVQCodebook;
pub const ScaNNStats = misc.scann.IndexStats;

// ============================================================================
// Parallel HNSW type re-exports
// ============================================================================

pub const ParallelHnswBuilder = misc.parallel_hnsw.ParallelHnswBuilder;
pub const ParallelBuildConfig = misc.parallel_hnsw.ParallelBuildConfig;
pub const ParallelBuildStats = misc.parallel_hnsw.ParallelBuildStats;

// ============================================================================
// Module Lifecycle
// ============================================================================

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    return error.DatabaseDisabled;
}
pub fn deinit() void {
    initialized = false;
}
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return initialized;
}

// ============================================================================
// Core Database Operations
// ============================================================================

pub fn open(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}
pub fn connect(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}
pub fn close(_: *DatabaseHandle) void {}
pub fn insert(_: *DatabaseHandle, _: u64, _: []const f32, _: ?[]const u8) !void {
    return error.DatabaseDisabled;
}
pub fn search(_: *DatabaseHandle, _: std.mem.Allocator, _: []const f32, _: usize) ![]SearchResult {
    return error.DatabaseDisabled;
}
pub fn remove(_: *DatabaseHandle, _: u64) bool {
    return false;
}
pub fn update(_: *DatabaseHandle, _: u64, _: []const f32) !bool {
    return error.DatabaseDisabled;
}
pub fn get(_: *DatabaseHandle, _: u64) ?VectorView {
    return null;
}
pub fn list(_: *DatabaseHandle, _: std.mem.Allocator, _: usize) ![]VectorView {
    return error.DatabaseDisabled;
}
pub fn stats(_: *DatabaseHandle) Stats {
    return .{};
}
pub fn diagnostics(_: *DatabaseHandle) DiagnosticsInfo {
    return .{};
}
pub fn optimize(_: *DatabaseHandle) !void {
    return error.DatabaseDisabled;
}
pub fn backup(_: *DatabaseHandle, _: []const u8) !void {
    return error.DatabaseDisabled;
}
pub fn restore(_: *DatabaseHandle, _: []const u8) !void {
    return error.DatabaseDisabled;
}
pub fn openFromFile(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}
pub fn openOrCreate(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}
