//! Canonical semantic-store surface for weighted memory, retrieval, and lineage.

const std = @import("std");
const wdbx_engine = @import("../../../wdbx/wdbx.zig");
const block_chain = @import("../block_chain.zig");
const distributed = @import("../distributed/mod.zig");

pub const StoreHandle = struct {
    engine: wdbx_engine.Engine,
    allocator: std.mem.Allocator,
};

pub const DatabaseHandle = StoreHandle;

pub const SearchResult = struct {
    id: u64,
    score: f32,
    metadata: ?[]const u8 = null,
};

pub const VectorView = struct {
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8,
};

pub const Stats = struct {
    count: usize,
};

pub const DatabaseConfig = struct {
    path: []const u8 = "",
};

pub const BatchItem = struct {
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8 = null,
};

pub const MemoryBlock = block_chain.ConversationBlock;
pub const MemoryBlockConfig = block_chain.BlockConfig;

pub const DistributedConfig = distributed.DistributedConfig;
pub const ShardManager = distributed.ShardManager;
pub const ShardConfig = distributed.ShardConfig;
pub const ShardKey = distributed.ShardKey;
pub const BlockExchangeManager = distributed.BlockExchangeManager;
pub const VersionVector = distributed.VersionVector;
pub const VersionComparison = distributed.VersionComparison;
pub const BlockConflict = distributed.BlockConflict;

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
    block_id: u64,
    score: f32,
    similarity: f32,
    importance: f32 = 0.0,
    trace: InfluenceTrace = .{},
};

pub fn openStore(allocator: std.mem.Allocator, name: []const u8) !StoreHandle {
    _ = name;
    return .{
        .engine = try wdbx_engine.Engine.init(allocator, .{}),
        .allocator = allocator,
    };
}

pub fn openStoreWithConfig(
    allocator: std.mem.Allocator,
    name: []const u8,
    config: DatabaseConfig,
) !StoreHandle {
    _ = name;
    _ = config;
    return openStore(allocator, name);
}

pub fn connectStore(allocator: std.mem.Allocator, name: []const u8) !StoreHandle {
    return openStore(allocator, name);
}

pub fn closeStore(handle: *StoreHandle) void {
    handle.engine.deinit();
}

pub fn storeVector(
    handle: *StoreHandle,
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8,
) !void {
    var id_buf: [32]u8 = undefined;
    const id_str = try std.fmt.bufPrint(&id_buf, "{d}", .{id});
    
    const meta = wdbx_engine.Metadata{
        .text = metadata orelse "",
    };

    try handle.engine.indexByVector(id_str, vector, meta);
}

pub fn searchStore(
    handle: *StoreHandle,
    allocator: std.mem.Allocator,
    query: []const f32,
    top_k: usize,
) ![]SearchResult {
    const results = try handle.engine.searchByVector(query, .{ .k = top_k });
    defer allocator.free(results);

    const out = try allocator.alloc(SearchResult, results.len);
    for (results, 0..) |res, i| {
        out[i] = .{
            .id = std.fmt.parseInt(u64, res.id, 10) catch 0,
            .score = res.similarity,
        };
    }
    return out;
}

pub fn backupStore(handle: *StoreHandle, path: []const u8) !void {
    try wdbx_engine.save(&handle.engine, path);
}

pub fn restoreStore(handle: *StoreHandle, path: []const u8) !void {
    const restored = try wdbx_engine.load(handle.allocator, path);
    handle.engine.deinit();
    handle.engine = restored;
}

pub const createDatabase = openStore;
pub const createDatabaseWithConfig = openStoreWithConfig;
pub const connectDatabase = connectStore;
pub const closeDatabase = closeStore;
pub const insertVector = storeVector;

pub fn insertBatch(handle: *StoreHandle, items: []const BatchItem) !void {
    for (items) |item| {
        try storeVector(handle, item.id, item.vector, item.metadata);
    }
}

pub const searchVectors = searchStore;

pub fn searchVectorsInto(
    handle: *StoreHandle,
    query: []const f32,
    top_k: usize,
    results: []SearchResult,
) usize {
    const res = searchStore(handle, handle.allocator, query, top_k) catch return 0;
    defer handle.allocator.free(res);
    
    const count = @min(res.len, results.len);
    @memcpy(results[0..count], res[0..count]);
    return count;
}

pub fn deleteVector(handle: *StoreHandle, id: u64) bool {
    var id_buf: [32]u8 = undefined;
    const id_str = std.fmt.bufPrint(&id_buf, "{d}", .{id}) catch return false;
    return handle.engine.delete(id_str);
}

pub fn updateVector(handle: *StoreHandle, id: u64, vector: []const f32) !bool {
    _ = deleteVector(handle, id);
    try storeVector(handle, id, vector, null);
    return true;
}

pub fn getVector(handle: *StoreHandle, id: u64) ?VectorView {
    for (handle.engine.vectors_array.items) |item| {
        const vid = std.fmt.parseInt(u64, item.id, 10) catch continue;
        if (vid == id) {
            return VectorView{
                .id = vid,
                .vector = item.vec,
                .metadata = item.metadata.text,
            };
        }
    }
    return null;
}

pub fn listVectors(
    handle: *StoreHandle,
    allocator: std.mem.Allocator,
    limit: usize,
) ![]VectorView {
    const count = @min(handle.engine.count(), limit);
    const out = try allocator.alloc(VectorView, count);
    for (handle.engine.vectors_array.items[0..count], 0..) |item, i| {
        out[i] = .{
            .id = std.fmt.parseInt(u64, item.id, 10) catch 0,
            .vector = item.vec,
            .metadata = item.metadata.text,
        };
    }
    return out;
}

pub fn getStats(handle: *StoreHandle) Stats {
    return .{ .count = handle.engine.count() };
}

pub fn optimize(handle: *StoreHandle) !void {
    _ = handle;
}

pub fn backupToPath(handle: *StoreHandle, path: []const u8) !void {
    try backupStore(handle, path);
}

pub fn restoreFromPath(handle: *StoreHandle, path: []const u8) !void {
    try restoreStore(handle, path);
}

pub const backup = backupStore;
pub const restore = restoreStore;

test "influence trace captures retrieval metadata" {
    const trace = InfluenceTrace.forRetrieval(42, 0.8, 0.6);
    try std.testing.expectEqual(InfluenceTrace.Source.semantic_store, trace.source);
    try std.testing.expectEqual(@as(?u64, 42), trace.block_id);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), trace.weight_inputs.similarity, 0.0001);
}
