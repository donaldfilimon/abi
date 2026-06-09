const std = @import("std");
const build_options = @import("build_options");
const memory = @import("../../core/memory.zig");
const gpu = if (build_options.feat_gpu) @import("../gpu/mod.zig") else @import("../gpu/stub.zig");
const types = @import("stub_types.zig");

pub const MAX_LAYERS = types.MAX_LAYERS;
pub const HNSW_DIMENSIONS = types.HNSW_DIMENSIONS;
pub const VECTOR_PADDED_BYTES = types.VECTOR_PADDED_BYTES;
pub const StoreConfig = types.StoreConfig;
pub const VectorRecord = types.VectorRecord;
pub const SearchResult = types.SearchResult;
pub const ConversationBlock = types.ConversationBlock;
pub const AccelerationStatus = types.AccelerationStatus;
pub const StoreStats = types.StoreStats;

pub const spatial_3d = @import("stub_spatial_3d.zig");
pub const index = @import("stub_index.zig");

// WAL, temporal/causal index, cluster, compression, homomorphic crypto,
// compute-backend selection, and the REST listener are compiled out when the
// feature is disabled. Callers gate access at comptime (see abi_cli wdbx
// handler); these markers exist only to preserve mod/stub top-level declaration
// parity.
pub const wal = struct {};
pub const temporal = struct {};
pub const cluster = struct {};
pub const cluster_rpc = struct {};
pub const compression = struct {};
pub const crypto_he = struct {};
pub const compute = struct {};
pub const rest = struct {};
pub const recovery = struct {};
pub const retrieval = struct {};
pub const segments = struct {};

// Durable-session shim: when wdbx is disabled the session is a thin in-memory
// wrapper over the stub Store, so MCP/CLI callers compile and run without
// persistence (all underlying writes return error.FeatureDisabled).
pub const durable_store = struct {
    pub const Session = struct {
        store: Store,

        pub fn open(io: std.Io, allocator: std.mem.Allocator) !Session {
            _ = io;
            return .{ .store = Store.init(allocator) };
        }

        pub fn openAt(io: std.Io, allocator: std.mem.Allocator, base: []const u8) !Session {
            _ = io;
            _ = base;
            return .{ .store = Store.init(allocator) };
        }

        pub fn openInMemory(allocator: std.mem.Allocator) Session {
            return .{ .store = Store.init(allocator) };
        }

        pub fn isPersistent(self: *const Session) bool {
            _ = self;
            return false;
        }

        pub fn storePtr(self: *Session) *Store {
            return &self.store;
        }

        pub fn checkpoint(self: *Session) !void {
            _ = self;
        }

        pub fn deinit(self: *Session) void {
            self.store.deinit();
        }
    };
};

pub const persistence = struct {
    pub const HEADER = "# ABI-WDBX v1";
    pub const CHECKSUM_PREFIX = "# checksum:";

    pub const PersistenceError = error{
        InvalidHeader,
        UnknownLineType,
        MissingField,
        OutOfMemory,
        DuplicateVectorId,
        DimensionMismatch,
        CorruptVectorId,
        ChecksumMismatch,
        FieldOutOfRange,
    };

    pub fn serialize(_: std.mem.Allocator, _: *const Store) ![]u8 {
        return error.FeatureDisabled;
    }

    pub fn deserialize(_: std.mem.Allocator, _: []const u8) !Store {
        return error.FeatureDisabled;
    }

    pub fn saveToPath(_: std.Io, _: std.mem.Allocator, _: *const Store, _: []const u8) !void {
        return error.FeatureDisabled;
    }

    pub fn loadFromPath(_: std.Io, _: std.mem.Allocator, _: []const u8) !Store {
        return error.FeatureDisabled;
    }
};

pub const storage = @import("stub_storage.zig");

pub const Store = struct {
    pub fn init(a: std.mem.Allocator) Store {
        _ = a;
        return .{};
    }

    pub fn initWithConfig(a: std.mem.Allocator, config: StoreConfig) Store {
        _ = a;
        _ = config;
        return .{};
    }

    pub fn deinit(self: *Store) void {
        _ = self;
    }

    pub fn setTracker(self: *Store, t: *memory.MemoryTracker) void {
        _ = self;
        _ = t;
    }

    pub fn attachWal(self: *Store, io: std.Io, path: []const u8) void {
        _ = self;
        _ = io;
        _ = path;
    }

    pub fn store(self: *Store, key: []const u8, val: []const u8) !void {
        _ = self;
        _ = val;
        if (key.len == 0) return error.InvalidKey;
        return error.FeatureDisabled;
    }

    pub fn get(self: *const Store, key: []const u8) ?[]const u8 {
        _ = self;
        _ = key;
        return null;
    }

    pub fn count(self: *const Store) usize {
        _ = self;
        return 0;
    }

    pub fn vectorCount(self: *const Store) usize {
        _ = self;
        return 0;
    }

    pub fn stats(self: *const Store) StoreStats {
        _ = self;
        return .{
            .kv_entries = 0,
            .vectors = 0,
            .blocks = 0,
            .spatial_records = 0,
            .temporal_nodes = 0,
            .temporal_edges = 0,
            .vector_dimensions = null,
            .next_vector_id = 0,
            .acceleration = .{ .backend = .simulated, .mode = .cpu_fallback, .message = "wdbx feature is disabled" },
        };
    }

    pub fn putVector(self: *Store, values: []const f32) !u32 {
        _ = self;
        _ = values;
        return error.FeatureDisabled;
    }

    pub fn search(self: *Store, query: []const f32, limit: usize) ![]SearchResult {
        _ = self;
        _ = query;
        _ = limit;
        return error.FeatureDisabled;
    }

    pub fn addTemporalNode(self: *Store, id: u32, timestamp_ms: i64) !void {
        _ = self;
        _ = id;
        _ = timestamp_ms;
        return error.FeatureDisabled;
    }

    pub fn addTemporalEdge(self: *Store, cause: u32, effect: u32) !void {
        _ = self;
        _ = cause;
        _ = effect;
        return error.FeatureDisabled;
    }

    pub fn temporalNodeCount(self: *const Store) usize {
        _ = self;
        return 0;
    }

    pub fn temporalEdgeCount(self: *const Store) usize {
        _ = self;
        return 0;
    }

    pub fn temporalTimestamp(self: *const Store, id: u32) ?i64 {
        _ = self;
        _ = id;
        return null;
    }

    pub fn getVector(self: *const Store, id: u32) ?[]const f32 {
        _ = self;
        _ = id;
        return null;
    }

    pub fn appendBlock(self: *Store, profile: []const u8, query_id: u32, response_id: u32, metadata: []const u8) ![32]u8 {
        _ = self;
        _ = profile;
        _ = query_id;
        _ = response_id;
        _ = metadata;
        return error.FeatureDisabled;
    }

    pub fn restoreBlock(self: *Store, profile: []const u8, query_id: u32, response_id: u32, metadata: []const u8, timestamp_ms: i64) ![32]u8 {
        _ = timestamp_ms;
        return self.appendBlock(profile, query_id, response_id, metadata);
    }

    pub fn blockCount(self: *const Store) usize {
        _ = self;
        return 0;
    }

    pub fn lastBlock(self: *const Store) ?ConversationBlock {
        _ = self;
        return null;
    }

    pub fn verifyBlocks(self: *const Store) bool {
        _ = self;
        return true;
    }

    pub fn putSpatial3D(self: *Store, id: u32, point: spatial_3d.Point3D, payload: []const u8) !void {
        _ = self;
        _ = id;
        _ = point;
        _ = payload;
        return error.FeatureDisabled;
    }

    pub fn searchSpatial3D(self: *const Store, center: spatial_3d.Point3D, k: usize, metric: spatial_3d.DistanceMetric) ![]spatial_3d.SpatialSearchResult {
        _ = self;
        _ = center;
        _ = k;
        _ = metric;
        return error.FeatureDisabled;
    }

    pub fn searchSpatialRadius3D(self: *const Store, center: spatial_3d.Point3D, radius: f32, metric: spatial_3d.DistanceMetric) ![]spatial_3d.SpatialSearchResult {
        _ = self;
        _ = center;
        _ = radius;
        _ = metric;
        return error.FeatureDisabled;
    }

    pub fn accelerationStatus(self: *const Store) AccelerationStatus {
        _ = self;
        return .{ .backend = .simulated, .mode = .cpu_fallback, .message = "wdbx feature is disabled" };
    }

    pub fn exportManifest(self: *const Store, allocator: std.mem.Allocator) ![]u8 {
        _ = self;
        return try allocator.dupe(u8, "{\"kv_entries\":0,\"vectors\":0,\"blocks\":0,\"spatial_records\":0,\"temporal_nodes\":0,\"temporal_edges\":0,\"vector_dimensions\":null,\"next_vector_id\":0,\"backend\":\"simulated\",\"mode\":\"cpu_fallback\",\"disabled\":true}");
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "wdbx stub reports disabled operations" {
    var store = Store.init(std.testing.allocator);
    defer store.deinit();

    try std.testing.expectEqual(@as(usize, 0), store.count());
    try std.testing.expectEqual(@as(usize, 0), store.vectorCount());
    try std.testing.expectEqual(@as(usize, 0), store.blockCount());
    try std.testing.expectEqual(@as(usize, 0), store.temporalNodeCount());
    try std.testing.expectEqual(@as(usize, 0), store.temporalEdgeCount());
    try std.testing.expect(store.temporalTimestamp(1) == null);
    try std.testing.expect(store.get("missing") == null);
    try std.testing.expect(store.getVector(1) == null);
    try std.testing.expect(store.lastBlock() == null);
    try std.testing.expect(store.verifyBlocks());

    const stats_value = store.stats();
    try std.testing.expectEqual(@as(usize, 0), stats_value.vectors);
    try std.testing.expectEqual(@as(usize, 0), stats_value.blocks);
    try std.testing.expectEqual(@as(usize, 0), stats_value.temporal_nodes);
    try std.testing.expectEqual(@as(usize, 0), stats_value.temporal_edges);
    try std.testing.expectEqual(@as(?usize, null), stats_value.vector_dimensions);
    try std.testing.expectEqual(gpu.ExecutionMode.cpu_fallback, stats_value.acceleration.mode);

    try std.testing.expectError(error.InvalidKey, store.store("", "metadata"));
    try std.testing.expectError(error.FeatureDisabled, store.store("metadata", "value"));
    try std.testing.expectError(error.FeatureDisabled, store.putVector(&.{1.0}));
    try std.testing.expectError(error.FeatureDisabled, store.search(&.{1.0}, 1));
    try std.testing.expectError(error.FeatureDisabled, store.addTemporalNode(1, 123));
    try std.testing.expectError(error.FeatureDisabled, store.addTemporalEdge(1, 2));
    try std.testing.expectError(error.FeatureDisabled, store.appendBlock("abi", 1, 2, "metadata"));
    try std.testing.expectError(error.FeatureDisabled, store.restoreBlock("abi", 1, 2, "metadata", 123));
    try std.testing.expectError(error.FeatureDisabled, store.putSpatial3D(1, .{ .x = 0, .y = 0, .z = 0 }, "payload"));
    try std.testing.expectError(error.FeatureDisabled, store.searchSpatial3D(.{ .x = 0, .y = 0, .z = 0 }, 1, .euclidean));
    try std.testing.expectError(error.FeatureDisabled, store.searchSpatialRadius3D(.{ .x = 0, .y = 0, .z = 0 }, 1.0, .euclidean));

    const manifest = try store.exportManifest(std.testing.allocator);
    defer std.testing.allocator.free(manifest);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"spatial_records\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"temporal_nodes\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"temporal_edges\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"vector_dimensions\":null") != null);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"mode\":\"cpu_fallback\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"disabled\":true") != null);
}

test "wdbx stub nested writes are explicit disabled operations" {
    var vectors = index.VectorStorage.init(std.testing.allocator, 4, 4);
    defer vectors.deinit();
    try std.testing.expect(!vectors.contains(1));
    try std.testing.expectEqual(@as(usize, 0), vectors.get(1).len);
    try std.testing.expectError(error.FeatureDisabled, vectors.insert(1, &.{ 1, 2, 3, 4 }));

    var hnsw = index.HnswIndex(4).init(std.testing.allocator);
    defer hnsw.deinit();
    try std.testing.expectEqual(@as(usize, 0), hnsw.count());
    try std.testing.expectError(error.FeatureDisabled, hnsw.insert(1, &.{ 1, 2, 3, 4 }));
    try std.testing.expectError(error.FeatureDisabled, hnsw.search(&.{ 1, 2, 3, 4 }, 1));

    var spatial = spatial_3d.SpatialIndex3D.init(std.testing.allocator);
    defer spatial.deinit();
    var spatial_pooled = spatial_3d.SpatialIndex3D.initWithPool(std.testing.allocator, null);
    defer spatial_pooled.deinit();
    try std.testing.expectEqual(@as(usize, 0), spatial.count());
    try std.testing.expectEqual(@as(usize, 0), spatial_pooled.count());
    try std.testing.expectEqual(@as(f32, 1.0), spatial_3d.euclideanDistance(.{ .x = 0, .y = 0, .z = 0 }, .{ .x = 1, .y = 0, .z = 0 }));
    try std.testing.expectError(error.FeatureDisabled, spatial.insert(1, .{ .x = 0, .y = 0, .z = 0 }, "payload"));

    var chain = storage.BlockChain.init(std.testing.allocator);
    defer chain.deinit();
    try std.testing.expectEqual(@as(usize, 0), chain.len());
    try std.testing.expect(chain.verifyChain());
    try std.testing.expectError(error.InvalidProfile, chain.append("", 1, 2, "metadata"));
    try std.testing.expectError(error.FeatureDisabled, chain.append("abi", 1, 2, "metadata"));
    try std.testing.expectError(error.FeatureDisabled, chain.appendAt("abi", 1, 2, "metadata", 123));
    var it = chain.iterator();
    try std.testing.expect(it.next() == null);
    chain.releaseIterator();
}
