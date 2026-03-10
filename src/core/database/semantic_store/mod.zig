//! Canonical semantic-store surface for weighted memory, retrieval, and lineage.

const std = @import("std");
const legacy_wdbx = @import("../wdbx");

pub const StoreHandle = legacy_wdbx.DatabaseHandle;
pub const DatabaseHandle = StoreHandle;
pub const SearchResult = legacy_wdbx.SearchResult;
pub const VectorView = legacy_wdbx.VectorView;
pub const Stats = legacy_wdbx.Stats;
pub const DatabaseConfig = legacy_wdbx.DatabaseConfig;
pub const BatchItem = legacy_wdbx.BatchItem;

pub const MemoryBlock = struct {
    id: u64 = 0,
    text: []const u8 = "",
    metadata: ?[]const u8 = null,
};

pub const MemoryBlockConfig = struct {};

pub const DistributedConfig = struct {
    enabled: bool = false,
};

pub const WeightInputs = struct {
    similarity: f32 = 0.0,
    importance: f32 = 0.0,
    recency: f32 = 1.0,
    custom_boost: f32 = 0.0,

    pub fn combinedScore(self: WeightInputs) f32 {
        return self.similarity * 0.7 +
            self.importance * 0.2 +
            self.recency * 0.1 +
            self.custom_boost;
    }
};

pub const Lineage = struct {
    parent_block_id: ?u64 = null,
    shard_key_hash: ?u64 = null,
    replica_count: u32 = 0,
};

pub const InfluenceTrace = struct {
    source: Source = .semantic_store,
    block_id: ?u64 = null,
    weight_inputs: WeightInputs = .{},
    lineage: ?Lineage = null,

    pub const Source = enum {
        semantic_store,
        local_memory,
        distributed_replica,
    };

    pub fn forRetrieval(block_id: u64, similarity: f32, importance: f32) InfluenceTrace {
        return .{
            .source = .semantic_store,
            .block_id = block_id,
            .weight_inputs = .{
                .similarity = similarity,
                .importance = importance,
            },
        };
    }
};

pub const RetrievalHit = struct {
    block_id: u64,
    score: f32,
    similarity: f32,
    importance: f32 = 0.0,
    trace: InfluenceTrace = .{},
};

pub fn openStore(allocator: std.mem.Allocator, name: []const u8) !StoreHandle {
    return legacy_wdbx.createDatabase(allocator, name);
}

pub fn openStoreWithConfig(
    allocator: std.mem.Allocator,
    name: []const u8,
    config: DatabaseConfig,
) !StoreHandle {
    return legacy_wdbx.createDatabaseWithConfig(allocator, name, config);
}

pub fn connectStore(allocator: std.mem.Allocator, name: []const u8) !StoreHandle {
    return legacy_wdbx.connectDatabase(allocator, name);
}

pub fn closeStore(handle: *StoreHandle) void {
    legacy_wdbx.closeDatabase(handle);
}

pub fn storeVector(
    handle: *StoreHandle,
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8,
) !void {
    try legacy_wdbx.insertVector(handle, id, vector, metadata);
}

pub fn searchStore(
    handle: *StoreHandle,
    allocator: std.mem.Allocator,
    query: []const f32,
    top_k: usize,
) ![]SearchResult {
    return legacy_wdbx.searchVectors(handle, allocator, query, top_k);
}

pub fn backupStore(handle: *StoreHandle, path: []const u8) !void {
    try legacy_wdbx.backup(handle, path);
}

pub fn restoreStore(handle: *StoreHandle, path: []const u8) !void {
    try legacy_wdbx.restore(handle, path);
}

pub const createDatabase = openStore;
pub const createDatabaseWithConfig = openStoreWithConfig;
pub const connectDatabase = connectStore;
pub const closeDatabase = closeStore;
pub const insertVector = storeVector;
pub const insertBatch = legacy_wdbx.insertBatch;
pub const searchVectors = searchStore;
pub const searchVectorsInto = legacy_wdbx.searchVectorsInto;
pub const deleteVector = legacy_wdbx.deleteVector;
pub const updateVector = legacy_wdbx.updateVector;
pub const getVector = legacy_wdbx.getVector;
pub const listVectors = legacy_wdbx.listVectors;
pub const getStats = legacy_wdbx.getStats;
pub const optimize = legacy_wdbx.optimize;
pub const backupToPath = legacy_wdbx.backupToPath;
pub const restoreFromPath = legacy_wdbx.restoreFromPath;
pub const backup = backupStore;
pub const restore = restoreStore;

test "semantic_store aliases the legacy handle surface" {
    try std.testing.expect(StoreHandle == legacy_wdbx.DatabaseHandle);
}

test "influence trace captures retrieval metadata" {
    const trace = InfluenceTrace.forRetrieval(42, 0.8, 0.6);
    try std.testing.expectEqual(InfluenceTrace.Source.semantic_store, trace.source);
    try std.testing.expectEqual(@as(?u64, 42), trace.block_id);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), trace.weight_inputs.similarity, 0.0001);
}
