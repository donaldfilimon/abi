const std = @import("std");
const build_options = @import("build_options");
const memory = @import("../../core/memory.zig");
const gpu = if (build_options.feat_gpu) @import("../gpu/mod.zig") else @import("../gpu/stub.zig");
const foundation_pool = @import("../../foundation/pool_allocator.zig");

pub const MAX_LAYERS = 4;
const HNSW_DIMENSIONS = 128;
pub const VECTOR_PADDED_BYTES = HNSW_DIMENSIONS * @sizeOf(f32);

pub const StoreConfig = struct {
    pool_alloc: ?*foundation_pool.PoolAllocator = null,
};

pub const VectorRecord = struct {
    id: u32,
    values: []f32,
};

pub const SearchResult = struct {
    id: u32,
    score: f32,
};

pub const ConversationBlock = struct {
    id: [32]u8,
    prev_id: [32]u8,
    timestamp_ms: i64,
    profile: []const u8,
    query_id: u32,
    response_id: u32,
    metadata: []const u8,
};

pub const AccelerationStatus = struct {
    backend: gpu.Backend,
    mode: gpu.ExecutionMode,
    message: []const u8,
};

pub const StoreStats = struct {
    kv_entries: usize,
    vectors: usize,
    blocks: usize,
    spatial_records: usize,
    vector_dimensions: ?usize,
    next_vector_id: u32,
    acceleration: AccelerationStatus,
};

pub const spatial_3d = struct {
    pub const Point3D = struct {
        x: f32,
        y: f32,
        z: f32,
    };
    pub const SpatialRecord3D = struct {
        id: u32,
        point: Point3D,
        payload: []const u8,
    };
    pub const DistanceMetric = enum {
        euclidean,
        manhattan,
        cosine,
    };
    pub const SpatialSearchResult = struct {
        id: u32,
        distance: f32,
        point: Point3D,
        payload: []const u8,
    };

    pub fn euclideanDistance(p1: Point3D, p2: Point3D) f32 {
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        const dz = p1.z - p2.z;
        return @sqrt(dx * dx + dy * dy + dz * dz);
    }

    pub fn manhattanDistance(p1: Point3D, p2: Point3D) f32 {
        return @abs(p1.x - p2.x) + @abs(p1.y - p2.y) + @abs(p1.z - p2.z);
    }

    pub fn cosineDistance(p1: Point3D, p2: Point3D) f32 {
        const dot = p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
        const norm1 = @sqrt(p1.x * p1.x + p1.y * p1.y + p1.z * p1.z);
        const norm2 = @sqrt(p2.x * p2.x + p2.y * p2.y + p2.z * p2.z);
        if (norm1 == 0 or norm2 == 0) return 1.0;
        return 1.0 - (dot / (norm1 * norm2));
    }

    pub fn calculateDistance(p1: Point3D, p2: Point3D, metric: DistanceMetric) f32 {
        return switch (metric) {
            .euclidean => euclideanDistance(p1, p2),
            .manhattan => manhattanDistance(p1, p2),
            .cosine => cosineDistance(p1, p2),
        };
    }

    pub const SpatialIndex3D = struct {
        pub fn init(allocator: std.mem.Allocator) SpatialIndex3D {
            _ = allocator;
            return .{};
        }

        pub fn deinit(self: *SpatialIndex3D) void {
            _ = self;
        }

        pub fn insert(self: *SpatialIndex3D, id: u32, point: Point3D, payload: []const u8) !void {
            _ = self;
            _ = id;
            _ = point;
            _ = payload;
            return error.FeatureDisabled;
        }

        pub fn count(self: *const SpatialIndex3D) usize {
            _ = self;
            return 0;
        }

        pub fn radiusSearch(self: *const SpatialIndex3D, center: Point3D, radius: f32, metric: DistanceMetric) ![]SpatialSearchResult {
            _ = self;
            _ = center;
            _ = radius;
            _ = metric;
            return error.FeatureDisabled;
        }

        pub fn nearestNeighbors(self: *const SpatialIndex3D, center: Point3D, k: usize, metric: DistanceMetric) ![]SpatialSearchResult {
            _ = self;
            _ = center;
            _ = k;
            _ = metric;
            return error.FeatureDisabled;
        }
    };
};

pub const index = struct {
    pub const MAX_LAYERS = 4;
    pub const M = 16;
    pub const EF_CONSTRUCTION = 40;
    pub const EF_SEARCH = 32;

    pub const HnswNode = struct {
        pub fn initEdges(allocator: std.mem.Allocator) [4]std.ArrayListUnmanaged(u32) {
            _ = allocator;
            var arr: [4]std.ArrayListUnmanaged(u32) = undefined;
            var i: usize = 0;
            while (i < 4) : (i += 1) {
                arr[i] = .empty;
            }
            return arr;
        }

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            _ = self;
            _ = allocator;
        }
    };

    pub const VectorStorage = struct {
        pub fn init(allocator: std.mem.Allocator, dimensions: usize, initial_capacity: usize) VectorStorage {
            _ = allocator;
            _ = dimensions;
            _ = initial_capacity;
            return .{};
        }

        pub fn deinit(self: *VectorStorage) void {
            _ = self;
        }

        pub fn setTracker(self: *VectorStorage, tracker: *memory.MemoryTracker) void {
            _ = self;
            _ = tracker;
        }

        pub fn insert(self: *VectorStorage, id: u32, values: []const f32) !void {
            _ = self;
            _ = id;
            _ = values;
            return error.FeatureDisabled;
        }

        pub fn get(self: *const VectorStorage, id: u32) []const f32 {
            _ = self;
            _ = id;
            return &[_]f32{};
        }

        pub fn contains(self: *const VectorStorage, id: u32) bool {
            _ = self;
            _ = id;
            return false;
        }
    };

    pub const Candidate = struct {
        id: u32,
        distance: f32,
    };

    pub fn cosineDistanceSIMD(a: []const f32, b: []const f32) f32 {
        _ = a;
        _ = b;
        return 1.0;
    }

    pub fn HnswIndex(comptime D: usize) type {
        return struct {
            storage: VectorStorage,

            pub fn init(allocator: std.mem.Allocator) @This() {
                return .{ .storage = VectorStorage.init(allocator, D, 64) };
            }

            pub fn deinit(self: *@This()) void {
                self.storage.deinit();
            }

            pub fn setTracker(self: *@This(), tracker: *memory.MemoryTracker) void {
                _ = self;
                _ = tracker;
            }

            pub fn insert(self: *@This(), id: u32, values: []const f32) !void {
                _ = self;
                _ = id;
                _ = values;
                return error.FeatureDisabled;
            }

            pub fn search(self: *@This(), query: []const f32, limit: usize) ![]SearchResult {
                _ = self;
                _ = query;
                _ = limit;
                return error.FeatureDisabled;
            }

            pub fn count(self: *const @This()) usize {
                _ = self;
                return 0;
            }
        };
    }
};

pub const storage = struct {
    pub const HASH_LEN = 32;
    pub const GENESIS_HASH: [HASH_LEN]u8 = std.mem.zeroes([HASH_LEN]u8);

    pub const BlockHeader = struct {
        hash: [HASH_LEN]u8 = undefined,
        prev_hash: [HASH_LEN]u8 = undefined,
        timestamp_ms: i64 = 0,
        sequence: u64 = 0,
    };

    pub const MvccBlock = struct {
        header: BlockHeader = .{},
        data: ConversationBlock = undefined,
        next: ?*MvccBlock = null,
        version: u64 = 0,

        pub fn deinit(self: *MvccBlock, allocator: std.mem.Allocator) void {
            _ = self;
            _ = allocator;
        }
    };

    pub fn computeBlockHash(prev_hash: [HASH_LEN]u8, timestamp_ms: i64, sequence: u64, profile: []const u8, metadata: []const u8) [HASH_LEN]u8 {
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        hasher.update(&prev_hash);
        var buf: [32]u8 = undefined;
        std.mem.writeInt(i64, buf[0..8], timestamp_ms, .little);
        std.mem.writeInt(u64, buf[8..16], sequence, .little);
        hasher.update(buf[0..16]);
        hasher.update(profile);
        hasher.update(metadata);
        var out: [HASH_LEN]u8 = undefined;
        hasher.final(&out);
        return out;
    }

    pub const BlockChain = struct {
        pub fn init(allocator: std.mem.Allocator) BlockChain {
            _ = allocator;
            return .{};
        }

        pub fn deinit(self: *BlockChain) void {
            _ = self;
        }

        pub fn append(self: *BlockChain, profile: []const u8, query_id: u32, response_id: u32, metadata: []const u8) ![HASH_LEN]u8 {
            _ = self;
            _ = query_id;
            _ = response_id;
            _ = metadata;
            if (profile.len == 0) return error.InvalidProfile;
            return error.FeatureDisabled;
        }

        pub fn getBlock(self: *BlockChain, hash: [HASH_LEN]u8) ?*const MvccBlock {
            _ = self;
            _ = hash;
            return null;
        }

        pub fn getSnapshot(self: *BlockChain) Snapshot {
            return .{ .head = null, .length = 0, .chain = self };
        }

        pub fn releaseSnapshot(self: *BlockChain) void {
            _ = self;
        }

        pub fn verifyChain(self: *BlockChain) bool {
            _ = self;
            return true;
        }

        pub fn len(self: *const BlockChain) usize {
            _ = self;
            return 0;
        }

        pub fn getTailHash(self: *BlockChain) ?[HASH_LEN]u8 {
            _ = self;
            return null;
        }

        pub fn iterator(self: *const BlockChain) Iterator {
            _ = self;
            return .{ .current = null };
        }

        pub fn releaseIterator(self: *const BlockChain) void {
            _ = self;
        }

        pub const Iterator = struct {
            current: ?*const MvccBlock,

            pub fn next(self: *Iterator) ?*const MvccBlock {
                _ = self;
                return null;
            }
        };

        pub const Snapshot = struct {
            head: ?*const MvccBlock,
            length: usize,
            chain: *const BlockChain,

            pub fn getBlock(self: *const Snapshot, hash: [HASH_LEN]u8) ?*const MvccBlock {
                _ = self;
                _ = hash;
                return null;
            }

            pub fn iterator(self: *const Snapshot) Iterator {
                _ = self;
                return .{ .current = null };
            }

            pub fn len(self: *const Snapshot) usize {
                return self.length;
            }
        };
    };
};

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
        return try allocator.dupe(u8, "{\"kv_entries\":0,\"vectors\":0,\"blocks\":0,\"spatial_records\":0,\"vector_dimensions\":null,\"next_vector_id\":0,\"backend\":\"simulated\",\"mode\":\"cpu_fallback\",\"disabled\":true}");
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
    try std.testing.expect(store.get("missing") == null);
    try std.testing.expect(store.getVector(1) == null);
    try std.testing.expect(store.lastBlock() == null);
    try std.testing.expect(store.verifyBlocks());

    const stats_value = store.stats();
    try std.testing.expectEqual(@as(usize, 0), stats_value.vectors);
    try std.testing.expectEqual(@as(usize, 0), stats_value.blocks);
    try std.testing.expectEqual(@as(?usize, null), stats_value.vector_dimensions);
    try std.testing.expectEqual(gpu.ExecutionMode.cpu_fallback, stats_value.acceleration.mode);

    try std.testing.expectError(error.InvalidKey, store.store("", "metadata"));
    try std.testing.expectError(error.FeatureDisabled, store.store("metadata", "value"));
    try std.testing.expectError(error.FeatureDisabled, store.putVector(&.{1.0}));
    try std.testing.expectError(error.FeatureDisabled, store.search(&.{1.0}, 1));
    try std.testing.expectError(error.FeatureDisabled, store.appendBlock("abi", 1, 2, "metadata"));
    try std.testing.expectError(error.FeatureDisabled, store.putSpatial3D(1, .{ .x = 0, .y = 0, .z = 0 }, "payload"));
    try std.testing.expectError(error.FeatureDisabled, store.searchSpatial3D(.{ .x = 0, .y = 0, .z = 0 }, 1, .euclidean));
    try std.testing.expectError(error.FeatureDisabled, store.searchSpatialRadius3D(.{ .x = 0, .y = 0, .z = 0 }, 1.0, .euclidean));

    const manifest = try store.exportManifest(std.testing.allocator);
    defer std.testing.allocator.free(manifest);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"spatial_records\":0") != null);
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
    try std.testing.expectEqual(@as(usize, 0), spatial.count());
    try std.testing.expectEqual(@as(f32, 1.0), spatial_3d.euclideanDistance(.{ .x = 0, .y = 0, .z = 0 }, .{ .x = 1, .y = 0, .z = 0 }));
    try std.testing.expectError(error.FeatureDisabled, spatial.insert(1, .{ .x = 0, .y = 0, .z = 0 }, "payload"));

    var chain = storage.BlockChain.init(std.testing.allocator);
    defer chain.deinit();
    try std.testing.expectEqual(@as(usize, 0), chain.len());
    try std.testing.expect(chain.verifyChain());
    try std.testing.expectError(error.InvalidProfile, chain.append("", 1, 2, "metadata"));
    try std.testing.expectError(error.FeatureDisabled, chain.append("abi", 1, 2, "metadata"));
    var it = chain.iterator();
    try std.testing.expect(it.next() == null);
    chain.releaseIterator();
}
