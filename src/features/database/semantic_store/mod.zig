//! Canonical semantic-store surface layered over the legacy WDBX implementation.
//!
//! Wave 1 keeps the existing WDBX APIs available while introducing neutral
//! terminology for storage, retrieval, provenance, and distributed lineage.

const std = @import("std");
const legacy_store = @import("../wdbx.zig");
const block_chain = @import("../block_chain.zig");
const distributed = @import("../distributed/mod.zig");

pub const StoreHandle = legacy_store.DatabaseHandle;
pub const SearchResult = legacy_store.SearchResult;
pub const VectorView = legacy_store.VectorView;
pub const Stats = legacy_store.Stats;
pub const DatabaseConfig = legacy_store.DatabaseConfig;
pub const BatchItem = legacy_store.BatchItem;

pub const MemoryBlock = block_chain.ConversationBlock;
pub const MemoryBlockConfig = block_chain.BlockConfig;

pub const WeightInputs = struct {
    retrieval_score: ?f32 = null,
    importance: ?f32 = null,
};

pub const Lineage = struct {
    parent_block_id: ?u64 = null,
    shard_hash: ?u64 = null,
    sync_state: ?distributed.SyncState = null,
};

pub const InfluenceTrace = struct {
    pub const Source = enum {
        semantic_store,
        long_term_memory,
        distributed_replica,
        routing,
    };

    source: Source = .semantic_store,
    block_id: ?u64 = null,
    weight_inputs: WeightInputs = .{},
    lineage: Lineage = .{},

    pub fn forRetrieval(block_id: ?u64, retrieval_score: f32, importance: f32) InfluenceTrace {
        return .{
            .source = .semantic_store,
            .block_id = block_id,
            .weight_inputs = .{
                .retrieval_score = retrieval_score,
                .importance = importance,
            },
        };
    }
};

pub const RetrievalHit = struct {
    block_id: ?u64 = null,
    distance: ?f32 = null,
    score: f32 = 0.0,
    importance: f32 = 0.0,
    trace: InfluenceTrace = .{},
};

pub const DistributedConfig = distributed.DistributedConfig;
pub const ShardManager = distributed.ShardManager;
pub const ShardConfig = distributed.ShardConfig;
pub const ShardKey = distributed.ShardKey;
pub const BlockExchangeManager = distributed.BlockExchangeManager;
pub const VersionVector = distributed.VersionVector;
pub const VersionComparison = distributed.VersionComparison;
pub const BlockConflict = distributed.BlockConflict;

pub usingnamespace legacy_store;

pub fn openStore(allocator: std.mem.Allocator, name: []const u8) !StoreHandle {
    return legacy_store.createDatabase(allocator, name);
}

pub fn openStoreWithConfig(
    allocator: std.mem.Allocator,
    name: []const u8,
    config: DatabaseConfig,
) !StoreHandle {
    return legacy_store.createDatabaseWithConfig(allocator, name, config);
}

pub fn closeStore(handle: *StoreHandle) void {
    legacy_store.closeDatabase(handle);
}

pub fn storeVector(
    handle: *StoreHandle,
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8,
) !void {
    try legacy_store.insertVector(handle, id, vector, metadata);
}

pub fn searchStore(
    handle: *StoreHandle,
    allocator: std.mem.Allocator,
    query: []const f32,
    top_k: usize,
) ![]SearchResult {
    return legacy_store.searchVectors(handle, allocator, query, top_k);
}

pub fn backupStore(handle: *StoreHandle, path: []const u8) !void {
    try legacy_store.backup(handle, path);
}

pub fn restoreStore(handle: *StoreHandle, path: []const u8) !void {
    try legacy_store.restore(handle, path);
}

test "canonical store aliases preserve legacy handle type" {
    try std.testing.expect(StoreHandle == legacy_store.DatabaseHandle);
}

test "influence trace helper preserves weight inputs" {
    const trace = InfluenceTrace.forRetrieval(42, 0.9, 0.7);
    try std.testing.expectEqual(@as(?u64, 42), trace.block_id);
    try std.testing.expectEqual(@as(?f32, 0.9), trace.weight_inputs.retrieval_score);
    try std.testing.expectEqual(@as(?f32, 0.7), trace.weight_inputs.importance);
}
