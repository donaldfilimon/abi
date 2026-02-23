//! Distributed WDBX Architecture
//!
//! Implements the "wide distributed block exchange" from research with:
//! - Intelligent sharding (tenant → session → semantic clustering)
//! - Raft consensus for block coordination
//! - Block exchange protocol with anti-entropy
//! - Conflict resolution with MVCC and version vectors

const std = @import("std");
const db_parent = @import("../mod.zig");

// Dependencies from parent modules (re-exported for child modules)
pub const time = db_parent.time; // Re-exported by database parent
pub const network = @import("../../network/mod.zig");
pub const block_chain = db_parent.block_chain;

// Internal module imports
const shard_manager = @import("./shard_manager.zig");
const block_exchange = @import("./block_exchange.zig");
const raft_block_chain = @import("./raft_block_chain.zig");
pub const wal = @import("./wal.zig");
pub const cluster = @import("./cluster.zig");

// Module exports
pub const ShardManager = shard_manager.ShardManager;
pub const ShardConfig = shard_manager.ShardConfig;
pub const ShardKey = shard_manager.ShardKey;
pub const ShardManagerError = shard_manager.ShardManagerError;
pub const HashRing = shard_manager.HashRing;
pub const LoadStats = shard_manager.LoadStats;

pub const BlockExchangeManager = block_exchange.BlockExchangeManager;
pub const BlockExchangeError = block_exchange.BlockExchangeError;
pub const SyncState = block_exchange.SyncState;
pub const VersionVector = block_exchange.VersionVector;
pub const VersionComparison = block_exchange.VersionComparison;
pub const SyncRequest = block_exchange.SyncRequest;
pub const SyncResponse = block_exchange.SyncResponse;
pub const BlockConflict = block_exchange.BlockConflict;

pub const DistributedBlockChain = raft_block_chain.DistributedBlockChain;
pub const DistributedBlockChainConfig = raft_block_chain.DistributedBlockChainConfig;
pub const DistributedBlockChainError = raft_block_chain.DistributedBlockChainError;
pub const LocalStats = raft_block_chain.LocalStats;

pub const WalWriter = wal.WalWriter;
pub const WalReader = wal.WalReader;
pub const WalEntry = wal.WalEntry;
pub const WalEntryType = wal.WalEntryType;

pub const ClusterManager = cluster.ClusterManager;
pub const ClusterConfig = cluster.ClusterConfig;
pub const ClusterStatus = cluster.ClusterStatus;
pub const NodeRole = cluster.NodeRole;
pub const NodeState = cluster.NodeState;
pub const TransportType = cluster.TransportType;
pub const ClusterMessage = cluster.ClusterMessage;
pub const MessageType = cluster.MessageType;
pub const PeerAddress = cluster.PeerAddress;
pub const ClusterError = cluster.ClusterError;

/// Search result for distributed queries.
pub const SearchResult = struct {
    vector_id: u64,
    distance: f32,
};

/// Routes vector operations to the correct shard/node.
pub const ShardRouter = struct {
    num_shards: u32,

    pub fn init(num_shards: u32) ShardRouter {
        return .{ .num_shards = if (num_shards == 0) 1 else num_shards };
    }

    /// Determine which shard owns a vector by ID (consistent hashing).
    pub fn routeVector(self: *const ShardRouter, vector_id: u64) u32 {
        return @intCast(vector_id % self.num_shards);
    }
};

/// Merge and re-rank search results from multiple shards.
pub fn mergeSearchResults(
    allocator: std.mem.Allocator,
    result_sets: []const []const SearchResult,
    top_k: usize,
) ![]SearchResult {
    // Count total results
    var total: usize = 0;
    for (result_sets) |rs| total += rs.len;
    if (total == 0) return try allocator.alloc(SearchResult, 0);

    // Collect all results
    const all = try allocator.alloc(SearchResult, total);
    defer allocator.free(all);
    var offset: usize = 0;
    for (result_sets) |rs| {
        @memcpy(all[offset .. offset + rs.len], rs);
        offset += rs.len;
    }

    // Sort by distance (ascending = closest first)
    std.mem.sort(SearchResult, all, {}, struct {
        fn lessThan(_: void, a: SearchResult, b: SearchResult) bool {
            return a.distance < b.distance;
        }
    }.lessThan);

    // Return top_k
    const result_count = @min(top_k, all.len);
    const result = try allocator.alloc(SearchResult, result_count);
    @memcpy(result, all[0..result_count]);
    return result;
}

/// Distributed WDBX configuration
pub const DistributedConfig = struct {
    /// Sharding configuration
    sharding: ShardConfig = .{},
    /// Cluster configuration
    cluster_config: cluster.ClusterConfig = .{},
    /// Enable Raft consensus
    enable_consensus: bool = true,
    /// Enable WAL for durability
    enable_wal: bool = true,
    /// WAL path (default: alongside database file)
    wal_path: [512]u8 = [_]u8{0} ** 512,
    wal_path_len: u16 = 0,
    /// Replication factor
    replication_factor: u32 = 3,
    /// Enable anti-entropy
    enable_anti_entropy: bool = true,
    /// Anti-entropy interval (seconds)
    anti_entropy_interval_s: i64 = 300,
    /// Enable locality-aware placement
    enable_locality_aware: bool = true,
    /// Transport type for cross-node communication
    transport: TransportType = .tcp,
};

/// Distributed context for framework integration
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: DistributedConfig,
    shard_manager: ?*ShardManager = null,
    block_exchange: ?*BlockExchangeManager = null,

    pub fn init(allocator: std.mem.Allocator, cfg: DistributedConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.shard_manager) |sm| {
            sm.deinit();
            self.allocator.destroy(sm);
        }
        if (self.block_exchange) |be| {
            be.deinit();
            self.allocator.destroy(be);
        }
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ShardRouter deterministic routing" {
    const router = ShardRouter.init(4);
    // Same vector_id always maps to same shard
    const shard1 = router.routeVector(100);
    const shard2 = router.routeVector(100);
    try std.testing.expectEqual(shard1, shard2);
    // 100 % 4 = 0
    try std.testing.expectEqual(@as(u32, 0), shard1);
    // 101 % 4 = 1
    try std.testing.expectEqual(@as(u32, 1), router.routeVector(101));
    // 102 % 4 = 2
    try std.testing.expectEqual(@as(u32, 2), router.routeVector(102));
}

test "ShardRouter with single shard" {
    const router = ShardRouter.init(1);
    // All vectors go to shard 0
    try std.testing.expectEqual(@as(u32, 0), router.routeVector(0));
    try std.testing.expectEqual(@as(u32, 0), router.routeVector(999));
    try std.testing.expectEqual(@as(u32, 0), router.routeVector(123456));

    // Zero shards should be treated as 1
    const router_zero = ShardRouter.init(0);
    try std.testing.expectEqual(@as(u32, 0), router_zero.routeVector(42));
}

test "mergeSearchResults correctness" {
    const allocator = std.testing.allocator;

    const set1 = [_]SearchResult{
        .{ .vector_id = 1, .distance = 0.1 },
        .{ .vector_id = 3, .distance = 0.5 },
    };
    const set2 = [_]SearchResult{
        .{ .vector_id = 2, .distance = 0.2 },
        .{ .vector_id = 4, .distance = 0.3 },
    };

    const result_sets = [_][]const SearchResult{
        &set1,
        &set2,
    };

    const merged = try mergeSearchResults(allocator, &result_sets, 3);
    defer allocator.free(merged);

    // Should be sorted by distance, top-3
    try std.testing.expectEqual(@as(usize, 3), merged.len);
    try std.testing.expectEqual(@as(u64, 1), merged[0].vector_id); // distance 0.1
    try std.testing.expectEqual(@as(u64, 2), merged[1].vector_id); // distance 0.2
    try std.testing.expectEqual(@as(u64, 4), merged[2].vector_id); // distance 0.3
}

test {
    std.testing.refAllDecls(@This());
}
