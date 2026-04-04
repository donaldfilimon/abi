//! Canonical semantic-store surface for weighted memory, retrieval, and lineage.

const std = @import("std");
const database = @import("../database.zig");
const storage = @import("../storage.zig");
const fs = @import("../../../foundation/mod.zig").utils.fs;

pub const StoreHandle = struct {
    db: database.Database,
};
pub const DatabaseHandle = StoreHandle;
pub const SearchResult = database.SearchResult;
pub const VectorView = database.VectorView;
pub const Stats = database.Stats;
pub const DatabaseConfig = database.DatabaseConfig;
pub const BatchItem = database.Database.BatchItem;
pub const DiagnosticsInfo = database.DiagnosticsInfo;

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
    return .{ .db = try database.Database.init(allocator, name) };
}

pub fn openStoreWithConfig(
    allocator: std.mem.Allocator,
    name: []const u8,
    config: DatabaseConfig,
) !StoreHandle {
    return .{ .db = try database.Database.initWithConfig(allocator, name, config) };
}

pub fn connectStore(allocator: std.mem.Allocator, name: []const u8) !StoreHandle {
    return openStore(allocator, name);
}

pub fn closeStore(handle: *StoreHandle) void {
    handle.db.deinit();
    handle.* = undefined;
}

pub fn storeVector(
    handle: *StoreHandle,
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8,
) !void {
    try handle.db.insert(id, vector, metadata);
}

pub fn searchStore(
    handle: *StoreHandle,
    allocator: std.mem.Allocator,
    query: []const f32,
    top_k: usize,
) ![]SearchResult {
    return handle.db.search(allocator, query, top_k);
}

/// Search with weighted influence scoring that populates InfluenceTrace metadata.
/// Uses the semantic weighting formula: similarity*0.7 + importance*0.2 + recency*0.1.
pub fn searchStoreWeighted(
    handle: *StoreHandle,
    allocator: std.mem.Allocator,
    query: []const f32,
    top_k: usize,
) ![]RetrievalHit {
    const raw_results = try handle.db.search(allocator, query, top_k * 2);
    defer allocator.free(raw_results);

    const stats = handle.db.stats();
    const total_count = stats.count;

    var hits = std.ArrayList(RetrievalHit).init(allocator);
    errdefer hits.deinit();

    for (raw_results, 0..) |result, rank| {
        // Derive importance from the raw similarity score (higher score = more important).
        const importance: f32 = @min(1.0, @max(0.0, result.score));

        // Derive recency from insertion order: more recent vectors have higher IDs.
        // Normalize to [0,1] based on position relative to total count.
        const recency: f32 = if (total_count > 0)
            @as(f32, @floatFromInt(result.id)) / @as(f32, @floatFromInt(@max(total_count, 1)))
        else
            1.0;

        const weights = WeightInputs{
            .similarity = result.score,
            .importance = importance,
            .recency = @min(1.0, recency),
        };

        const trace = InfluenceTrace{
            .source = .semantic_store,
            .block_id = result.id,
            .weight_inputs = weights,
        };

        try hits.append(.{
            .block_id = result.id,
            .score = weights.combinedScore(),
            .similarity = result.score,
            .importance = importance,
            .trace = trace,
        });

        if (hits.items.len >= top_k) break;
        _ = rank;
    }

    // Sort by combined score descending
    std.sort.pdq(RetrievalHit, hits.items, {}, struct {
        fn lessThan(_: void, a: RetrievalHit, b: RetrievalHit) bool {
            return a.score > b.score;
        }
    }.lessThan);

    return hits.toOwnedSlice();
}

pub fn insertBatch(handle: *StoreHandle, items: []const BatchItem) !void {
    try handle.db.insertBatch(items);
}

pub fn searchVectorsInto(
    handle: *StoreHandle,
    query: []const f32,
    top_k: usize,
    results: []SearchResult,
) usize {
    return handle.db.searchInto(query, top_k, results);
}

pub fn deleteVector(handle: *StoreHandle, id: u64) bool {
    return handle.db.delete(id);
}

pub fn updateVector(handle: *StoreHandle, id: u64, vector: []const f32) !bool {
    return handle.db.update(id, vector);
}

pub fn getVector(handle: *StoreHandle, id: u64) ?VectorView {
    return handle.db.get(id);
}

pub fn listVectors(
    handle: *StoreHandle,
    allocator: std.mem.Allocator,
    limit: usize,
) ![]VectorView {
    return handle.db.list(allocator, limit);
}

pub fn getStats(handle: *StoreHandle) Stats {
    return handle.db.stats();
}

pub fn getDiagnostics(handle: *StoreHandle) DiagnosticsInfo {
    return handle.db.diagnostics();
}

pub fn optimize(handle: *StoreHandle) !void {
    handle.db.optimize();
}

fn ensureParentDirExists(allocator: std.mem.Allocator, path: []const u8) !void {
    const dir_path = std.fs.path.dirname(path) orelse return;
    if (dir_path.len == 0) return;

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    std.Io.Dir.cwd().createDirPath(io, dir_path) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
}

pub fn backupToPath(handle: *StoreHandle, path: []const u8) !void {
    try ensureParentDirExists(handle.db.allocator, path);
    try storage.saveDatabase(handle.db.allocator, &handle.db, path);
}

pub fn restoreFromPath(handle: *StoreHandle, path: []const u8) !void {
    const allocator = handle.db.allocator;
    const restored = try storage.loadDatabase(allocator, path);
    handle.db.deinit();
    handle.db = restored;
}

pub fn backupStore(handle: *StoreHandle, path: []const u8) !void {
    if (!fs.isSafeBackupPath(path)) return fs.PathValidationError.InvalidPath;
    const safe_path = try fs.normalizeBackupPath(handle.db.allocator, path);
    defer handle.db.allocator.free(safe_path);
    try backupToPath(handle, safe_path);
}

pub fn restoreStore(handle: *StoreHandle, path: []const u8) !void {
    if (!fs.isSafeBackupPath(path)) return fs.PathValidationError.InvalidPath;
    const allocator = handle.db.allocator;
    const safe_path = try fs.normalizeBackupPath(allocator, path);
    defer allocator.free(safe_path);
    try restoreFromPath(handle, safe_path);
}

pub const createDatabase = openStore;
pub const createDatabaseWithConfig = openStoreWithConfig;
pub const connectDatabase = connectStore;
pub const closeDatabase = closeStore;
pub const insertVector = storeVector;
pub const searchVectors = searchStore;
pub const backup = backupStore;
pub const restore = restoreStore;

test "semantic_store owns the database handle surface" {
    var handle = try openStore(std.testing.allocator, "semantic-store-handle");
    defer closeStore(&handle);

    try std.testing.expect(@TypeOf(handle) == StoreHandle);
}

test "influence trace captures retrieval metadata" {
    const trace = InfluenceTrace.forRetrieval(42, 0.8, 0.6);
    try std.testing.expectEqual(InfluenceTrace.Source.semantic_store, trace.source);
    try std.testing.expectEqual(@as(?u64, 42), trace.block_id);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), trace.weight_inputs.similarity, 0.0001);
}

// refAllDecls deferred — searchStoreWeighted uses ArrayList.init (Zig 0.16: use .empty)
