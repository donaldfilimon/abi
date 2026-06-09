const std = @import("std");
const foundation_pool = @import("../../foundation/pool_allocator.zig");
const memory = @import("../../core/memory.zig");
const runtime = @import("runtime.zig");
const types = @import("types.zig");

pub const index = @import("hnsw.zig");
pub const storage = @import("chain.zig");
pub const spatial_3d = @import("spatial_3d.zig");
pub const persistence = @import("persistence.zig");
pub const wal = @import("wal.zig");
pub const temporal = @import("temporal.zig");
pub const cluster = @import("cluster.zig");
pub const compression = @import("compression.zig");
pub const crypto_he = @import("crypto_he.zig");
pub const compute = @import("compute.zig");
pub const rest = @import("rest.zig");
pub const recovery = @import("recovery.zig");
pub const retrieval = @import("retrieval.zig");
pub const segments = @import("segments.zig");
pub const durable_store = @import("durable_store.zig");

pub const MAX_LAYERS = types.MAX_LAYERS;
pub const HNSW_DIMENSIONS = types.HNSW_DIMENSIONS;
pub const VECTOR_PADDED_BYTES = types.VECTOR_PADDED_BYTES;
pub const VectorRecord = types.VectorRecord;
pub const SearchResult = types.SearchResult;
pub const ConversationBlock = types.ConversationBlock;
pub const AccelerationStatus = types.AccelerationStatus;
pub const StoreStats = types.StoreStats;
pub const StoreConfig = types.StoreConfig;

pub const Store = struct {
    /// Optional sidecar write-ahead-log binding. When set (via `attachWal`),
    /// supported mutations append a durable record per mutation. The path is
    /// borrowed; its owner (a `durable_store.Session`) outlives the Store.
    pub const WalBinding = struct {
        io: std.Io,
        path: []const u8,
    };

    allocator: std.mem.Allocator,
    entries: std.StringHashMap([]const u8),
    index: index.HnswIndex(HNSW_DIMENSIONS),
    chain: storage.BlockChain,
    spatial_index: spatial_3d.SpatialIndex3D,
    temporal_graph: temporal.TemporalCausalGraph,
    next_vector_id: u32 = 1,
    vector_dimensions: ?usize = null,
    acceleration: AccelerationStatus,
    tracker: ?*memory.MemoryTracker = null,
    pool_alloc: ?*foundation_pool.PoolAllocator = null,
    wal_binding: ?WalBinding = null,

    pub fn init(a: std.mem.Allocator) Store {
        return initWithConfig(a, .{});
    }

    pub fn initWithConfig(a: std.mem.Allocator, config: StoreConfig) Store {
        return .{
            .allocator = a,
            .entries = std.StringHashMap([]const u8).init(a),
            .index = index.HnswIndex(HNSW_DIMENSIONS).init(a),
            .chain = storage.BlockChain.init(a),
            .spatial_index = spatial_3d.SpatialIndex3D.initWithPool(a, config.pool_alloc),
            .temporal_graph = temporal.TemporalCausalGraph.init(a),
            .acceleration = runtime.defaultAcceleration(),
            .tracker = null,
            .pool_alloc = config.pool_alloc,
        };
    }

    fn paddedAlloc(self: *Store) ![]f32 {
        if (self.pool_alloc) |pool| {
            const block = try pool.alloc();
            return @ptrCast(@alignCast(block));
        }
        return try self.allocator.alloc(f32, HNSW_DIMENSIONS);
    }

    fn paddedFree(self: *Store, padded: []f32) void {
        if (self.pool_alloc) |pool| {
            const block: []u8 = @ptrCast(padded);
            pool.free(block);
        } else {
            self.allocator.free(padded);
        }
    }

    pub fn deinit(self: *Store) void {
        var it = self.entries.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.entries.deinit();

        self.index.deinit();
        self.chain.deinit();
        self.spatial_index.deinit();
        self.temporal_graph.deinit();
    }

    pub fn store(self: *Store, key: []const u8, val: []const u8) !void {
        if (key.len == 0) return error.InvalidKey;

        const owned_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(owned_key);

        const owned_val = try self.allocator.dupe(u8, val);
        errdefer self.allocator.free(owned_val);

        const result = try self.entries.getOrPut(owned_key);
        if (result.found_existing) {
            self.allocator.free(owned_key);
            self.allocator.free(result.value_ptr.*);
            result.value_ptr.* = owned_val;
        } else {
            result.key_ptr.* = owned_key;
            result.value_ptr.* = owned_val;
        }
        if (self.wal_binding) |w| try wal.appendKv(w.io, self.allocator, w.path, key, val);
    }

    pub fn get(self: *const Store, key: []const u8) ?[]const u8 {
        return self.entries.get(key);
    }

    pub fn count(self: *const Store) usize {
        return self.entries.count();
    }

    pub fn vectorCount(self: *const Store) usize {
        return self.index.count();
    }

    pub fn setTracker(self: *Store, t: *memory.MemoryTracker) void {
        self.tracker = t;
        self.index.setTracker(t);
        self.temporal_graph.setTracker(t);
    }

    /// Bind a sidecar WAL so supported mutations (kv, block, temporal node/edge)
    /// are logged per mutation. `path` is borrowed and must outlive the Store.
    pub fn attachWal(self: *Store, io: std.Io, path: []const u8) void {
        self.wal_binding = .{ .io = io, .path = path };
    }

    pub fn stats(self: *const Store) StoreStats {
        return .{
            .kv_entries = self.entries.count(),
            .vectors = self.index.count(),
            .blocks = self.chain.len(),
            .spatial_records = self.spatial_index.count(),
            .temporal_nodes = self.temporal_graph.nodeCount(),
            .temporal_edges = self.temporal_graph.edgeCount(),
            .vector_dimensions = self.vector_dimensions,
            .next_vector_id = self.next_vector_id,
            .acceleration = self.acceleration,
        };
    }

    pub fn putVector(self: *Store, values: []const f32) !u32 {
        if (values.len == 0) return error.InvalidVector;
        if (values.len > HNSW_DIMENSIONS) return error.DimensionMismatch;
        if (self.vector_dimensions) |dims| {
            if (dims != values.len) return error.DimensionMismatch;
        } else {
            self.vector_dimensions = values.len;
        }

        const padded_size = VECTOR_PADDED_BYTES;

        var padded_values = try self.paddedAlloc();
        errdefer {
            self.paddedFree(padded_values);
            if (self.tracker) |t| t.trackFreeNoTag(padded_size);
        }
        if (self.tracker) |t| t.trackAllocNoTag(padded_size);
        @memset(padded_values, 0);
        @memcpy(padded_values[0..values.len], values);

        const id = self.next_vector_id;
        self.next_vector_id += 1;
        try self.index.insert(id, padded_values);
        self.paddedFree(padded_values);
        if (self.tracker) |t| t.trackFreeNoTag(padded_size);
        self.acceleration = try runtime.runAccelerationKernel("wdbx.putVector", values.len);
        return id;
    }

    pub fn search(self: *Store, query: []const f32, limit: usize) ![]SearchResult {
        if (query.len == 0) return error.InvalidVector;
        if (query.len > HNSW_DIMENSIONS) return error.DimensionMismatch;
        if (self.vector_dimensions) |dims| {
            if (dims != query.len) return error.DimensionMismatch;
        }

        const padded_size = VECTOR_PADDED_BYTES;

        var padded_query = try self.paddedAlloc();
        errdefer {
            self.paddedFree(padded_query);
            if (self.tracker) |t| t.trackFreeNoTag(padded_size);
        }
        if (self.tracker) |t| t.trackAllocNoTag(padded_size);
        @memset(padded_query, 0);
        @memcpy(padded_query[0..query.len], query);

        const results = try self.index.search(padded_query, limit);
        self.paddedFree(padded_query);
        if (self.tracker) |t| t.trackFreeNoTag(padded_size);
        self.acceleration = try runtime.runAccelerationKernel("wdbx.search", query.len * self.index.count());
        return results;
    }

    pub fn addTemporalNode(self: *Store, id: u32, timestamp_ms: i64) !void {
        try self.temporal_graph.addNode(id, timestamp_ms);
        if (self.wal_binding) |w| try wal.appendTemporalNode(w.io, self.allocator, w.path, id, timestamp_ms);
    }

    pub fn addTemporalEdge(self: *Store, cause: u32, effect: u32) !void {
        try self.temporal_graph.addCausalEdge(cause, effect);
        if (self.wal_binding) |w| try wal.appendTemporalEdge(w.io, self.allocator, w.path, cause, effect);
    }

    pub fn temporalNodeCount(self: *const Store) usize {
        return self.temporal_graph.nodeCount();
    }

    pub fn temporalEdgeCount(self: *const Store) usize {
        return self.temporal_graph.edgeCount();
    }

    pub fn temporalTimestamp(self: *const Store, id: u32) ?i64 {
        return self.temporal_graph.timestampFor(id);
    }

    pub fn getVector(self: *const Store, id: u32) ?[]const f32 {
        if (!self.index.storage.contains(id)) return null;
        return self.index.storage.get(id);
    }

    pub fn appendBlock(self: *Store, profile: []const u8, query_id: u32, response_id: u32, metadata: []const u8) ![32]u8 {
        const hash = try self.chain.append(profile, query_id, response_id, metadata);
        if (self.wal_binding) |w| {
            // Log with the timestamp the chain assigned so a replayed block
            // reproduces the same SHA-256 hash.
            const last = self.lastBlock().?;
            try wal.appendBlock(w.io, self.allocator, w.path, profile, query_id, response_id, metadata, last.timestamp_ms);
        }
        return hash;
    }

    /// Append a block preserving an explicit `timestamp_ms`. Used by snapshot
    /// restore so reconstructed block hashes match the originals exactly.
    pub fn restoreBlock(self: *Store, profile: []const u8, query_id: u32, response_id: u32, metadata: []const u8, timestamp_ms: i64) ![32]u8 {
        return try self.chain.appendAt(profile, query_id, response_id, metadata, timestamp_ms);
    }

    pub fn blockCount(self: *const Store) usize {
        return self.chain.len();
    }

    pub fn lastBlock(self: *const Store) ?ConversationBlock {
        var it = self.chain.iterator();
        defer self.chain.releaseIterator();
        var last: ?ConversationBlock = null;
        while (it.next()) |node| {
            last = node.data;
        }
        return last;
    }

    pub fn verifyBlocks(self: *const Store) bool {
        const self_mut = @constCast(self);
        return self_mut.chain.verifyChain();
    }

    pub fn putSpatial3D(self: *Store, id: u32, point: spatial_3d.Point3D, payload: []const u8) !void {
        try self.spatial_index.insert(id, point, payload);
        self.acceleration = try runtime.runAccelerationKernel("wdbx.putSpatial3D", 3);
    }

    pub fn searchSpatial3D(self: *const Store, center: spatial_3d.Point3D, k: usize, metric: spatial_3d.DistanceMetric) ![]spatial_3d.SpatialSearchResult {
        const results = try self.spatial_index.nearestNeighbors(center, k, metric);
        const self_mut = @constCast(self);
        self_mut.acceleration = try runtime.runAccelerationKernel("wdbx.searchSpatial3D", 3 * self.spatial_index.count());
        return results;
    }

    pub fn searchSpatialRadius3D(self: *const Store, center: spatial_3d.Point3D, radius: f32, metric: spatial_3d.DistanceMetric) ![]spatial_3d.SpatialSearchResult {
        const results = try self.spatial_index.radiusSearch(center, radius, metric);
        const self_mut = @constCast(self);
        self_mut.acceleration = try runtime.runAccelerationKernel("wdbx.searchSpatialRadius3D", 3 * self.spatial_index.count());
        return results;
    }

    pub fn accelerationStatus(self: *const Store) AccelerationStatus {
        return self.acceleration;
    }

    pub fn exportManifest(self: *const Store, allocator: std.mem.Allocator) ![]u8 {
        const s = self.stats();
        var out: std.ArrayListUnmanaged(u8) = .empty;
        errdefer out.deinit(allocator);
        try out.print(allocator, "{{\"kv_entries\":{d},\"vectors\":{d},\"blocks\":{d},\"spatial_records\":{d},\"temporal_nodes\":{d},\"temporal_edges\":{d},\"vector_dimensions\":", .{ s.kv_entries, s.vectors, s.blocks, s.spatial_records, s.temporal_nodes, s.temporal_edges });
        if (s.vector_dimensions) |dims| {
            try out.print(allocator, "{d}", .{dims});
        } else {
            try out.appendSlice(allocator, "null");
        }
        try out.print(
            allocator,
            ",\"next_vector_id\":{d},\"backend\":\"{s}\",\"mode\":\"{s}\"}}",
            .{ s.next_vector_id, runtime.backendName(s.acceleration.backend), runtime.executionModeName(s.acceleration.mode) },
        );
        return out.toOwnedSlice(allocator);
    }
};

test "Store owns and replaces entries" {
    var store_obj = Store.init(std.testing.allocator);
    defer store_obj.deinit();

    try store_obj.store("agent:abbey", "queued");
    try store_obj.store("agent:abbey", "trained");

    try std.testing.expectEqual(@as(usize, 1), store_obj.count());
    try std.testing.expectEqualStrings("trained", store_obj.get("agent:abbey") orelse return error.MissingEntry);
}

test "Store accelerates vector search and block chain memory" {
    var store_obj = Store.init(std.testing.allocator);
    defer store_obj.deinit();

    const q = try store_obj.putVector(&.{ 1, 0, 0, 0 });
    const r = try store_obj.putVector(&.{ 0.9, 0.1, 0, 0 });
    _ = try store_obj.putVector(&.{ 0, 1, 0, 0 });

    const results = try store_obj.search(&.{ 1, 0, 0, 0 }, 2);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(q, results[0].id);
    try std.testing.expect(results[0].score >= results[1].score);

    _ = try store_obj.appendBlock("abbey", q, r, "accelerated=true");
    try std.testing.expectEqual(@as(usize, 1), store_obj.blockCount());
    try std.testing.expectEqual(@as(usize, 3), store_obj.vectorCount());
    try std.testing.expect(store_obj.getVector(q) != null);
    try std.testing.expect(store_obj.lastBlock() != null);
    try std.testing.expect(store_obj.verifyBlocks());

    const manifest = try store_obj.exportManifest(std.testing.allocator);
    defer std.testing.allocator.free(manifest);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"vectors\":3") != null);
}

test "Store rejects mismatched vector dimensions" {
    var store_obj = Store.init(std.testing.allocator);
    defer store_obj.deinit();

    _ = try store_obj.putVector(&.{ 1, 0, 0, 0 });
    try std.testing.expectError(error.DimensionMismatch, store_obj.putVector(&.{ 1, 0 }));
    try std.testing.expectError(error.DimensionMismatch, store_obj.search(&.{ 1, 0 }, 1));
}

test "Store validates edge cases and exports complete stats" {
    var store_obj = Store.init(std.testing.allocator);
    defer store_obj.deinit();

    try std.testing.expectError(error.InvalidKey, store_obj.store("", "value"));
    try std.testing.expectError(error.InvalidVector, store_obj.putVector(&.{}));
    try std.testing.expectError(error.InvalidVector, store_obj.search(&.{}, 1));

    var oversized: [HNSW_DIMENSIONS + 1]f32 = undefined;
    @memset(&oversized, 0);
    try std.testing.expectError(error.DimensionMismatch, store_obj.putVector(&oversized));
    try std.testing.expectError(error.DimensionMismatch, store_obj.search(&oversized, 1));

    const empty_results = try store_obj.search(&.{ 1, 0, 0, 0 }, 5);
    defer std.testing.allocator.free(empty_results);
    try std.testing.expectEqual(@as(usize, 0), empty_results.len);

    try store_obj.putSpatial3D(7, .{ .x = 1, .y = 2, .z = 3 }, "payload");
    const stats_value = store_obj.stats();
    try std.testing.expectEqual(@as(usize, 1), stats_value.spatial_records);

    const manifest = try store_obj.exportManifest(std.testing.allocator);
    defer std.testing.allocator.free(manifest);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"spatial_records\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"vector_dimensions\":null") != null);
}

test "Store reuses pool-allocated buffers for vectors and spatial payloads" {
    var pool = foundation_pool.PoolAllocator.init(std.testing.allocator, VECTOR_PADDED_BYTES);
    defer pool.deinit();

    var store_obj = Store.initWithConfig(std.testing.allocator, .{ .pool_alloc = &pool });
    defer store_obj.deinit();

    const initial_chunks = pool.chunks.items.len;

    var i: usize = 0;
    while (i < 5) : (i += 1) {
        var v: [4]f32 = .{ 0, 0, 0, 0 };
        v[0] = @as(f32, @floatFromInt(i));
        _ = try store_obj.putVector(&v);
    }

    try std.testing.expectEqual(@as(usize, 5), store_obj.vectorCount());
    // Pool should have allocated exactly one chunk to satisfy 5 padded buffers
    // (each chunk holds 64 blocks of VECTOR_PADDED_BYTES).
    try std.testing.expectEqual(initial_chunks + 1, pool.chunks.items.len);

    const results = try store_obj.search(&.{ 0, 0, 0, 0 }, 5);
    defer std.testing.allocator.free(results);
    try std.testing.expect(results.len > 0);

    try store_obj.putSpatial3D(99, .{ .x = 1, .y = 2, .z = 3 }, "pooled-payload");
    try std.testing.expectEqual(initial_chunks + 1, pool.chunks.items.len);
    const spatial_results = try store_obj.searchSpatial3D(.{ .x = 1, .y = 2, .z = 3 }, 1, .euclidean);
    defer std.testing.allocator.free(spatial_results);
    try std.testing.expectEqual(@as(usize, 1), spatial_results.len);
    try std.testing.expectEqualStrings("pooled-payload", spatial_results[0].payload);
}

test {
    _ = @import("hnsw.zig");
    _ = @import("chain.zig");
    _ = @import("wal.zig");
    _ = @import("temporal.zig");
    _ = @import("cluster.zig");
    _ = @import("compression.zig");
    _ = @import("crypto_he.zig");
    _ = @import("compute.zig");
    _ = @import("rest.zig");
    _ = @import("runtime.zig");
    _ = @import("types.zig");
    std.testing.refAllDecls(@This());
}
