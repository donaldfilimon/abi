//! Semantic-store stub surface when database features are disabled.

const std = @import("std");
const types = @import("../stubs/types.zig");
const legacy_wdbx = @import("../stubs/wdbx.zig").wdbx;
const misc = @import("../stubs/misc.zig");

pub const StoreHandle = types.DatabaseHandle;
pub const DatabaseHandle = StoreHandle;
pub const SearchResult = types.SearchResult;
pub const VectorView = types.VectorView;
pub const Stats = types.Stats;
pub const DatabaseConfig = types.DatabaseConfig;
pub const BatchItem = types.BatchItem;

pub const MemoryBlock = misc.block_chain.ConversationBlock;
pub const MemoryBlockConfig = misc.block_chain.BlockChainConfig;

pub const DistributedConfig = misc.distributed.DistributedConfig;
pub const ShardManager = misc.distributed.ShardManager;
pub const ShardConfig = misc.distributed.ShardConfig;
pub const ShardKey = misc.distributed.ShardKey;
pub const BlockExchangeManager = misc.distributed.BlockExchangeManager;
pub const VersionVector = misc.distributed.VersionVector;
pub const VersionComparison = misc.distributed.VersionComparison;
pub const BlockConflict = misc.distributed.BlockConflict;

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
    shard_key: ?ShardKey = null,
    version_vector: ?VersionVector = null,
    conflict: ?BlockConflict = null,
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
    block_id: u64 = 0,
    score: f32 = 0.0,
    similarity: f32 = 0.0,
    importance: f32 = 0.0,
    trace: InfluenceTrace = .{},
};

pub fn openStore(_: std.mem.Allocator, _: []const u8) !StoreHandle {
    return error.DatabaseDisabled;
}

pub fn openStoreWithConfig(_: std.mem.Allocator, _: []const u8, _: DatabaseConfig) !StoreHandle {
    return error.DatabaseDisabled;
}

pub fn connectStore(_: std.mem.Allocator, _: []const u8) !StoreHandle {
    return error.DatabaseDisabled;
}

pub fn closeStore(handle: *StoreHandle) void {
    legacy_wdbx.closeDatabase(handle);
}

pub fn storeVector(
    _: *StoreHandle,
    _: u64,
    _: []const f32,
    _: ?[]const u8,
) !void {
    return error.DatabaseDisabled;
}

pub fn searchStore(
    _: *StoreHandle,
    _: std.mem.Allocator,
    _: []const f32,
    _: usize,
) ![]SearchResult {
    return error.DatabaseDisabled;
}

pub fn backupStore(_: *StoreHandle, _: []const u8) !void {
    return error.DatabaseDisabled;
}

pub fn restoreStore(_: *StoreHandle, _: []const u8) !void {
    return error.DatabaseDisabled;
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
pub const backupToPath = legacy_wdbx.backup;
pub const restoreFromPath = legacy_wdbx.restore;
pub const backup = backupStore;
pub const restore = restoreStore;

test {
    std.testing.refAllDecls(@This());
}
