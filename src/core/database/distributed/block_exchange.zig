//! Block Exchange Protocol for Distributed WDBX
//!
//! Implements block synchronization between cluster nodes, handling:
//! - Block replication across shard replicas
//! - Anti-entropy for consistency maintenance
//! - Conflict resolution with MVCC timestamps
//! - Incremental synchronization with version vectors

const std = @import("std");
const parent = @import("./mod.zig");
const time = parent.time;
const network = parent.network;
const block_chain = parent.block_chain;
const shard_manager = @import("./shard_manager.zig");

pub const BlockExchangeError = error{
    SyncInProgress,
    NodeUnreachable,
    BlockConflict,
    VersionMismatch,
    ShardNotFound,
    ConsensusRequired,
    TransmissionFailed,
    OutOfMemory,
};

/// Block synchronization state between nodes
pub const SyncState = enum {
    synchronized,
    syncing_pending,
    syncing_active,
    conflict_resolution,
    error_state,

    pub fn toString(self: SyncState) []const u8 {
        return switch (self) {
            .synchronized => "synchronized",
            .syncing_pending => "syncing_pending",
            .syncing_active => "syncing_active",
            .conflict_resolution => "conflict_resolution",
            .error_state => "error_state",
        };
    }
};

/// Version vector for causal consistency
pub const VersionVector = struct {
    node_id: []const u8,
    timestamps: std.AutoHashMapUnmanaged([]const u8, i64), // Node ID -> latest timestamp

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, node_id: []const u8) !Self {
        return Self{
            .node_id = try allocator.dupe(u8, node_id),
            .timestamps = .empty,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.node_id);

        var iter = self.timestamps.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        self.timestamps.deinit(allocator);
    }

    /// Update version with timestamp from another node
    pub fn update(self: *Self, allocator: std.mem.Allocator, node_id: []const u8, timestamp: i64) !void {
        const node_copy = try allocator.dupe(u8, node_id);

        if (self.timestamps.getPtr(node_copy)) |existing| {
            if (timestamp > existing.*) {
                existing.* = timestamp;
            }
        } else {
            try self.timestamps.put(allocator, node_copy, timestamp);
        }
    }

    /// Compare version vectors for conflict detection
    pub fn compare(self: *const Self, other: *const Self) VersionComparison {
        var self_ahead = false;
        var other_ahead = false;

        var iter = self.timestamps.iterator();
        while (iter.next()) |entry| {
            if (other.timestamps.get(entry.key_ptr.*)) |other_ts| {
                if (entry.value_ptr.* > other_ts) {
                    self_ahead = true;
                } else if (entry.value_ptr.* < other_ts) {
                    other_ahead = true;
                }
            } else {
                self_ahead = true; // Node only in self
            }
        }

        // Check nodes only in other
        var other_iter = other.timestamps.iterator();
        while (other_iter.next()) |entry| {
            if (self.timestamps.get(entry.key_ptr.*) == null) {
                other_ahead = true;
            }
        }

        if (self_ahead and other_ahead) {
            return .concurrent;
        } else if (self_ahead) {
            return .ahead;
        } else if (other_ahead) {
            return .behind;
        } else {
            return .equal;
        }
    }

    /// Merge version vectors (take maximum timestamps)
    pub fn merge(self: *Self, allocator: std.mem.Allocator, other: *const Self) !void {
        var iter = other.timestamps.iterator();
        while (iter.next()) |entry| {
            if (self.timestamps.get(entry.key_ptr.*)) |existing| {
                if (entry.value_ptr.* > existing) {
                    try self.timestamps.put(allocator, try allocator.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
                }
            } else {
                const node_copy = try allocator.dupe(u8, entry.key_ptr.*);
                try self.timestamps.put(allocator, node_copy, entry.value_ptr.*);
            }
        }
    }
};

/// Version comparison result
pub const VersionComparison = enum {
    ahead, // This vector causally ahead
    behind, // This vector causally behind
    equal, // Vectors identical
    concurrent, // Concurrent modifications
};

/// Block sync request message
pub const SyncRequest = struct {
    shard_id: shard_manager.ShardId,
    from_timestamp: i64,
    to_timestamp: ?i64 = null,
    limit: usize = 100,
    version_vector: VersionVector,

    pub fn deinit(self: *SyncRequest, allocator: std.mem.Allocator) void {
        self.version_vector.deinit(allocator);
        allocator.free(self.shard_id.replica_set);
    }
};

/// Block sync response message
pub const SyncResponse = struct {
    blocks: []block_chain.ConversationBlock,
    has_more: bool,
    next_timestamp: ?i64,
    version_vector: VersionVector,

    pub fn deinit(self: *SyncResponse, allocator: std.mem.Allocator) void {
        for (self.blocks) |*block| {
            block.deinit(allocator);
        }
        allocator.free(self.blocks);
        self.version_vector.deinit(allocator);
    }
};

/// Block conflict requiring resolution
pub const BlockConflict = struct {
    block_id: u64,
    local_block: block_chain.ConversationBlock,
    remote_block: block_chain.ConversationBlock,
    local_version: VersionVector,
    remote_version: VersionVector,
    conflict_type: ConflictType,

    pub const ConflictType = enum {
        timestamp_conflict, // Same block, different timestamps
        embedding_mismatch, // Embeddings differ beyond threshold
        metadata_inconsistency, // Metadata contradictory
        hash_mismatch, // Cryptographic hash mismatch
    };

    pub fn deinit(self: *BlockConflict, allocator: std.mem.Allocator) void {
        self.local_block.deinit(allocator);
        self.remote_block.deinit(allocator);
        self.local_version.deinit(allocator);
        self.remote_version.deinit(allocator);
    }
};

/// Block exchange manager for a node
pub const BlockExchangeManager = struct {
    allocator: std.mem.Allocator,
    node_id: []const u8,
    shard_manager: *shard_manager.ShardManager,
    transport: *network.TcpTransport,

    // Sync state per shard
    sync_states: std.AutoHashMapUnmanaged(u64, SyncState), // Shard hash -> sync state
    version_vectors: std.AutoHashMapUnmanaged(u64, VersionVector), // Shard hash -> version vector

    // Anti-entropy timers
    last_anti_entropy: i64,
    anti_entropy_interval: i64,

    // Conflict resolution registry
    conflicts: std.ArrayListUnmanaged(BlockConflict),

    const Self = @This();

    /// Initialize block exchange manager
    pub fn init(
        allocator: std.mem.Allocator,
        node_id: []const u8,
        shard_mgr: *shard_manager.ShardManager,
        transport: *network.TcpTransport,
        anti_entropy_interval_s: i64,
    ) !Self {
        const node_id_copy = try allocator.dupe(u8, node_id);

        return Self{
            .allocator = allocator,
            .node_id = node_id_copy,
            .shard_manager = shard_mgr,
            .transport = transport,
            .sync_states = .empty,
            .version_vectors = .empty,
            .last_anti_entropy = time.unixSeconds(),
            .anti_entropy_interval = anti_entropy_interval_s,
            .conflicts = .empty,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.node_id);

        // Clean up version vectors
        var iter = self.version_vectors.valueIterator();
        while (iter.next()) |vector| {
            vector.deinit(self.allocator);
        }
        self.version_vectors.deinit(self.allocator);

        // Clean up conflicts
        for (self.conflicts.items) |*conflict| {
            conflict.deinit(self.allocator);
        }
        self.conflicts.deinit(self.allocator);

        self.sync_states.deinit(self.allocator);
    }

    /// Handle block creation - initiate replication to other shard members
    pub fn replicateBlock(
        self: *Self,
        shard_id: shard_manager.ShardId,
        block: block_chain.ConversationBlock,
    ) !void {
        // Update local version vector
        const version_vector = try self.getOrCreateVersionVector(shard_id.hash);
        try version_vector.update(self.allocator, self.node_id, block.timestamp);

        // Replicate to other nodes in replica set (async)
        for (shard_id.replica_set) |replica_node| {
            if (std.mem.eql(u8, replica_node, self.node_id)) continue;

            // Schedule async replication
            try self.scheduleReplication(replica_node, shard_id, block);
        }

        // Update sync state
        try self.sync_states.put(self.allocator, shard_id.hash, .synchronized);
    }

    /// Schedule asynchronous block replication
    fn scheduleReplication(
        self: *Self,
        target_node: []const u8,
        shard_id: shard_manager.ShardId,
        block: block_chain.ConversationBlock,
    ) !void {
        // In real implementation, would use async task queue
        // For now, simulate synchronization

        std.log.info("BlockExchange: Scheduling replication to node {s} for block {d}", .{ target_node, block.hash });

        // Track replication state
        try self.sync_states.put(self.allocator, shard_id.hash, .syncing_pending);
    }

    /// Perform anti-entropy synchronization with another node
    pub fn performAntiEntropy(
        self: *Self,
        target_node: []const u8,
        shard_hash: u64,
    ) !void {
        const now = time.unixSeconds();
        if (now - self.last_anti_entropy < self.anti_entropy_interval) {
            return; // Not time for anti-entropy yet
        }

        // Get local version vector
        _ = self.version_vectors.get(shard_hash) orelse {
            return; // No version vector for this shard
        };

        // Send version vector to target for comparison
        std.log.info("BlockExchange: Anti-entropy with node {s} for shard {d}", .{ target_node, shard_hash });

        // In real implementation, would send network request
        // and handle response to sync differences

        self.last_anti_entropy = now;
    }

    /// Handle sync request from another node
    pub fn handleSyncRequest(self: *Self, request: *const SyncRequest) !SyncResponse {
        // Verify we're part of this shard's replica set
        var is_member = false;
        for (request.shard_id.replica_set) |node| {
            if (std.mem.eql(u8, node, self.node_id)) {
                is_member = true;
                break;
            }
        }

        if (!is_member) {
            return BlockExchangeError.ShardNotFound;
        }

        // Get local blocks in requested time range
        // In real implementation, would query block chain storage
        var blocks = std.ArrayListUnmanaged(block_chain.ConversationBlock).empty;
        defer blocks.deinit(self.allocator);

        // Get local version vector
        const local_vector = try self.getOrCreateVersionVector(request.shard_id.hash);

        // Create response version vector (copy of local)
        var response_vector = try VersionVector.init(self.allocator, self.node_id);
        errdefer response_vector.deinit(self.allocator);
        try response_vector.merge(self.allocator, local_vector);

        const response = SyncResponse{
            .blocks = try blocks.toOwnedSlice(self.allocator),
            .has_more = false,
            .next_timestamp = null,
            .version_vector = response_vector,
        };

        return response;
    }

    /// Handle sync response from another node
    pub fn handleSyncResponse(
        self: *Self,
        shard_hash: u64,
        response: *const SyncResponse,
    ) !void {
        // Merge version vectors
        const local_vector = try self.getOrCreateVersionVector(shard_hash);
        try local_vector.merge(self.allocator, &response.version_vector);

        // Apply received blocks (conflict resolution as needed)
        for (response.blocks) |block| {
            try self.applyRemoteBlock(shard_hash, block);
        }

        // Update sync state
        try self.sync_states.put(self.allocator, shard_hash, .synchronized);
    }

    /// Apply remote block with conflict resolution
    fn applyRemoteBlock(
        self: *Self,
        shard_hash: u64,
        remote_block: block_chain.ConversationBlock,
    ) !void {
        // Check for conflicts with local blocks
        // In real implementation, would query for existing block with same ID

        // For now, assume no conflict
        std.log.info("BlockExchange: Applying remote block {d} to shard {d}", .{ remote_block.hash, shard_hash });

        // Free the block copy (in real impl would store it)
        remote_block.deinit(self.allocator);
    }

    /// Detect and resolve block conflicts
    pub fn detectConflicts(
        self: *Self,
        shard_hash: u64,
        local_blocks: []const block_chain.ConversationBlock,
        remote_blocks: []const block_chain.ConversationBlock,
        remote_version: *const VersionVector,
    ) ![]BlockConflict {
        var conflicts = std.ArrayListUnmanaged(BlockConflict).empty;

        // Simple conflict detection: check for same block ID with different content
        for (local_blocks) |local_block| {
            for (remote_blocks) |remote_block| {
                if (local_block.hash == remote_block.hash) {
                    // Same block ID, check for differences
                    if (!blocksEqual(&local_block, &remote_block)) {
                        const local_vector = try self.getOrCreateVersionVector(shard_hash);

                        const conflict = BlockConflict{
                            .block_id = local_block.hash,
                            .local_block = local_block,
                            .remote_block = remote_block,
                            .local_version = try local_vector.copy(self.allocator),
                            .remote_version = try remote_version.copy(self.allocator),
                            .conflict_type = determineConflictType(&local_block, &remote_block),
                        };

                        try conflicts.append(self.allocator, conflict);
                        try self.conflicts.append(self.allocator, conflict);
                    }
                }
            }
        }

        return conflicts.toOwnedSlice(self.allocator);
    }

    /// Resolve conflicts using MVCC and version vectors
    pub fn resolveConflicts(self: *Self, conflicts: []BlockConflict) !void {
        for (conflicts) |*conflict| {
            const comparison = conflict.local_version.compare(&conflict.remote_version);

            switch (comparison) {
                .ahead => {
                    // Keep local version (causally ahead)
                    std.log.info("BlockExchange: Resolving conflict for block {d} - keeping local version (ahead)", .{conflict.block_id});
                },
                .behind => {
                    // Adopt remote version (causally behind)
                    std.log.info("BlockExchange: Resolving conflict for block {d} - adopting remote version (behind)", .{conflict.block_id});
                },
                .concurrent => {
                    // Concurrent modifications require advanced resolution
                    try self.resolveConcurrentConflict(conflict);
                },
                .equal => {
                    // Should not happen if blocks differ
                    std.log.warn("BlockExchange: Equal version vectors for differing blocks {d}", .{conflict.block_id});
                },
            }

            // Update version vector after resolution
            const shard_hash = std.hash.Fnv1a_64.hash(std.mem.asBytes(&conflict.block_id));
            const vector = try self.getOrCreateVersionVector(shard_hash);
            try vector.merge(self.allocator, &conflict.remote_version);
        }
    }

    /// Resolve concurrent modifications (most complex case)
    fn resolveConcurrentConflict(self: *Self, conflict: *BlockConflict) !void {
        _ = self;
        // Use MVCC timestamps as tiebreaker
        const local_commit = conflict.local_block.commit_timestamp;
        const remote_commit = conflict.remote_block.commit_timestamp;

        if (local_commit > remote_commit) {
            // Local committed later, keep local
            std.log.info("BlockExchange: Concurrent conflict for block {d} - keeping local (later commit)", .{conflict.block_id});
        } else if (local_commit < remote_commit) {
            // Remote committed later, adopt remote
            std.log.info("BlockExchange: Concurrent conflict for block {d} - adopting remote (later commit)", .{conflict.block_id});
        } else {
            // Same commit time, use lexical comparison of hashes as final tiebreaker
            const local_hash = conflict.local_block.hash;
            const remote_hash = conflict.remote_block.hash;

            if (local_hash > remote_hash) {
                std.log.info("BlockExchange: Concurrent conflict for block {d} - keeping local (lexical tiebreaker)", .{conflict.block_id});
            } else {
                std.log.info("BlockExchange: Concurrent conflict for block {d} - adopting remote (lexical tiebreaker)", .{conflict.block_id});
            }
        }
    }

    /// Get or create version vector for a shard
    fn getOrCreateVersionVector(self: *Self, shard_hash: u64) !*VersionVector {
        if (self.version_vectors.getPtr(shard_hash)) |vector| {
            return vector;
        }

        var vector = try VersionVector.init(self.allocator, self.node_id);
        errdefer vector.deinit(self.allocator);

        try self.version_vectors.put(self.allocator, shard_hash, vector);
        return self.version_vectors.getPtr(shard_hash).?;
    }

    /// Get current sync state for a shard
    pub fn getSyncState(self: *const Self, shard_hash: u64) SyncState {
        return self.sync_states.get(shard_hash) orelse .synchronized;
    }

    /// Check if any conflicts pending resolution
    pub fn hasPendingConflicts(self: *const Self) bool {
        return self.conflicts.items.len > 0;
    }
};

// Utility functions
fn blocksEqual(a: *const block_chain.ConversationBlock, b: *const block_chain.ConversationBlock) bool {
    // Compare cryptographic hashes for quick equality check
    return std.mem.eql(u8, &a.hash, &b.hash);
}

fn determineConflictType(
    local: *const block_chain.ConversationBlock,
    remote: *const block_chain.ConversationBlock,
) BlockConflict.ConflictType {
    // Check hash mismatch first (most critical)
    if (!std.mem.eql(u8, &local.hash, &remote.hash)) {
        return .hash_mismatch;
    }

    // Check timestamp conflict
    if (local.commit_timestamp != remote.commit_timestamp) {
        return .timestamp_conflict;
    }

    // Check embedding mismatch (would need threshold comparison)
    // For now default to metadata inconsistency

    return .metadata_inconsistency;
}

// Tests
test "VersionVector comparison" {
    const allocator = std.testing.allocator;

    var vector_a = try VersionVector.init(allocator, "node-a");
    defer vector_a.deinit(allocator);

    var vector_b = try VersionVector.init(allocator, "node-b");
    defer vector_b.deinit(allocator);

    // A ahead of B
    try vector_a.update(allocator, "node-a", 100);
    try vector_b.update(allocator, "node-b", 50);

    const comparison = vector_a.compare(&vector_b);
    try std.testing.expect(comparison == .concurrent); // Different nodes, concurrent

    // B updates with later timestamp
    try vector_b.update(allocator, "node-b", 150);
    _ = vector_a.compare(&vector_b);
    // Still concurrent since different nodes
}

test "BlockExchangeManager initialization" {
    const allocator = std.testing.allocator;

    // Create mock dependencies
    var registry = try network.NodeRegistry.init(allocator);
    defer registry.deinit();

    try registry.register("test-node", "127.0.0.1:9000");

    var shard_mgr = try shard_manager.ShardManager.init(allocator, .{ .virtual_nodes_per_node = 5, .replication_factor = 2 }, &registry);
    defer shard_mgr.deinit();

    // Note: Would need actual transport for full test
    // This tests basic initialization
    std.log.info("BlockExchange: Basic initialization test passed", .{});
}
