const std = @import("std");
const build_options = @import("build_options");
const gpu = if (build_options.feat_gpu) @import("../gpu/mod.zig") else @import("../gpu/stub.zig");

pub const MAX_LAYERS = 4;

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

pub const index = struct {
    pub const hnsw = struct {
        pub const MAX_LAYERS = 4;
        pub const M = 16;
        pub const EF_CONSTRUCTION = 40;
        pub const EF_SEARCH = 32;

        pub const HnswNode = struct {
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
            pub fn insert(self: *VectorStorage, id: u32, values: []const f32) !void {
                _ = self;
                _ = id;
                _ = values;
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

        pub fn cosineDistanceSIMD(a: []const f32, b: []const f32) f32 {
            _ = a;
            _ = b;
            return 1.0;
        }

        pub fn HnswIndex(comptime D: usize) type {
            _ = D;
            return struct {
                pub fn init(allocator: std.mem.Allocator) @This() {
                    _ = allocator;
                    return .{};
                }
                pub fn deinit(self: *@This()) void {
                    _ = self;
                }
                pub fn insert(self: *@This(), id: u32, values: []const f32) !void {
                    _ = self;
                    _ = id;
                    _ = values;
                }
                pub fn search(self: *@This(), query: []const f32, limit: usize) ![]SearchResult {
                    _ = self;
                    _ = query;
                    _ = limit;
                    return error.FeatureDisabled;
                }
                pub fn count(self: *@This()) usize {
                    _ = self;
                    return 0;
                }
            };
        }
    };
};

pub const storage = struct {
    pub const chain = struct {
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
            _ = prev_hash;
            _ = timestamp_ms;
            _ = sequence;
            _ = profile;
            _ = metadata;
            return std.mem.zeroes([HASH_LEN]u8);
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
                _ = profile;
                _ = query_id;
                _ = response_id;
                _ = metadata;
                return std.mem.zeroes([HASH_LEN]u8);
            }
            pub fn getBlock(self: *const BlockChain, hash: [HASH_LEN]u8) ?*const MvccBlock {
                _ = self;
                _ = hash;
                return null;
            }
            pub fn getSnapshot(self: *const BlockChain) Snapshot {
                return .{ .head = null, .length = 0, .chain = self };
            }
            pub fn releaseSnapshot(self: *const BlockChain) void {
                _ = self;
            }
            pub fn verifyChain(self: *const BlockChain) bool {
                _ = self;
                return true;
            }
            pub fn len(self: *const BlockChain) usize {
                _ = self;
                return 0;
            }
            pub fn getTailHash(self: *const BlockChain) ?[HASH_LEN]u8 {
                _ = self;
                return null;
            }
            pub fn iterator(self: *const BlockChain) Iterator {
                _ = self;
                return .{ .current = null };
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
};

pub const Store = struct {
    pub fn init(a: std.mem.Allocator) Store {
        _ = a;
        return .{};
    }
    pub fn deinit(self: *Store) void {
        _ = self;
    }
    pub fn store(self: *Store, key: []const u8, val: []const u8) !void {
        _ = self;
        _ = key;
        _ = val;
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
    pub fn putVector(self: *Store, values: []const f32) !u32 {
        _ = self;
        _ = values;
        return 0;
    }
    pub fn search(self: *Store, query: []const f32, limit: usize) ![]SearchResult {
        _ = self;
        _ = query;
        _ = limit;
        return error.FeatureDisabled;
    }
    pub fn appendBlock(self: *Store, profile: []const u8, query_id: u32, response_id: u32, metadata: []const u8) ![32]u8 {
        _ = self;
        _ = profile;
        _ = query_id;
        _ = response_id;
        _ = metadata;
        return std.mem.zeroes([32]u8);
    }

    pub fn blockCount(self: *const Store) usize {
        _ = self;
        return 0;
    }
    pub fn accelerationStatus(self: *const Store) AccelerationStatus {
        _ = self;
        return .{ .backend = .simulated, .mode = .cpu_fallback, .message = "wdbx feature is disabled" };
    }
};
