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

/// Distributed WDBX configuration
pub const DistributedConfig = struct {
    /// Sharding configuration
    sharding: ShardConfig = .{},
    /// Enable Raft consensus
    enable_consensus: bool = true,
    /// Replication factor
    replication_factor: u32 = 3,
    /// Enable anti-entropy
    enable_anti_entropy: bool = true,
    /// Anti-entropy interval (seconds)
    anti_entropy_interval_s: i64 = 300,
    /// Enable locality-aware placement
    enable_locality_aware: bool = true,
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
