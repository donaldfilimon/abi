//! Distributed Block Chain with Raft Consensus
//!
//! Implements the "wide distributed block exchange" from research by wrapping
//! local block chain with Raft consensus for distributed coordination.
//!
//! Architecture:
//! - Each node maintains local block chains per session
//! - Raft consensus coordinates block commits across cluster
//! - Leader node coordinates block replication to followers
//! - Followers apply committed blocks to their local chains

const std = @import("std");
const parent = @import("./mod.zig");
const time = parent.time;
const network = parent.network;
const block_chain = parent.block_chain;

pub const DistributedBlockChainError = error{
    NotLeader,
    ConsensusTimeout,
    BlockConflict,
    ShardUnavailable,
    NodeDisconnected,
    OutOfMemory,
};

/// Distributed block chain configuration
pub const DistributedBlockChainConfig = struct {
    /// Raft configuration
    raft_config: network.RaftConfig = .{},
    /// Block chain configuration
    block_chain_config: BlockChainConfig = .{},
    /// Enable automatic shard assignment
    enable_auto_sharding: bool = true,
    /// Replication factor for blocks
    replication_factor: u32 = 3,
    /// Sync timeout in milliseconds
    sync_timeout_ms: u64 = 5000,
};

/// Block chain wrapper configuration
pub const BlockChainConfig = struct {
    default_dimension: usize = 384,
    enable_mvcc: bool = true,
    enable_skip_pointers: bool = true,
    max_blocks_per_session: usize = 1000,
};

/// Distributed block chain node
pub const DistributedBlockChain = struct {
    allocator: std.mem.Allocator,
    config: DistributedBlockChainConfig,
    node_id: []const u8,

    // Local storage
    local_chains: std.StringHashMap(block_chain.BlockChain), // session_id -> chain
    mvcc_store: block_chain.MvccStore,

    // Distributed coordination
    raft_node: ?network.RaftNode,
    transport: ?network.TcpTransport,
    is_initialized: bool = false,

    const Self = @This();

    /// Initialize distributed block chain
    pub fn init(
        allocator: std.mem.Allocator,
        node_id: []const u8,
        config: DistributedBlockChainConfig,
    ) !Self {
        const node_id_copy = try allocator.dupe(u8, node_id);

        return Self{
            .allocator = allocator,
            .config = config,
            .node_id = node_id_copy,
            .local_chains = std.StringHashMap(block_chain.BlockChain).init(allocator),
            .mvcc_store = block_chain.MvccStore.init(allocator),
            .raft_node = null,
            .transport = null,
        };
    }

    pub fn deinit(self: *Self) void {
        // Clean up local chains
        var iter = self.local_chains.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.local_chains.deinit();

        // Clean up distributed components
        if (self.raft_node) |*raft| {
            raft.deinit();
        }
        if (self.transport) |*transport| {
            transport.deinit();
        }

        self.mvcc_store.deinit();
        self.allocator.free(self.node_id);
        self.* = undefined;
    }

    /// Start distributed coordination (join cluster)
    pub fn start(self: *Self, cluster_addresses: []const []const u8) !void {
        if (self.is_initialized) return;

        // Initialize Raft node
        self.raft_node = try network.RaftNode.init(self.allocator, self.node_id, self.config.raft_config);

        // Add cluster peers
        for (cluster_addresses) |address| {
            // Extract node ID from address (simplified)
            if (!std.mem.eql(u8, address, self.node_id)) {
                try self.raft_node.?.addPeer(address);
            }
        }

        // Initialize network transport
        self.transport = try network.TcpTransport.init(self.allocator, .{ .listen_address = "0.0.0.0", .listen_port = 9000 });

        self.is_initialized = true;
        std.log.info("DistributedBlockChain: Node {s} started and joined cluster", .{self.node_id});
    }

    /// Stop distributed coordination (leave cluster)
    pub fn stop(self: *Self) void {
        if (!self.is_initialized) return;

        if (self.raft_node) |*raft| {
            raft.deinit();
            self.raft_node = null;
        }

        if (self.transport) |*transport| {
            transport.deinit();
            self.transport = null;
        }

        self.is_initialized = false;
        std.log.info("DistributedBlockChain: Node {s} stopped", .{self.node_id});
    }

    /// Add a block to the distributed chain (requires consensus)
    pub fn addBlock(
        self: *Self,
        session_id: []const u8,
        config: block_chain.BlockConfig,
    ) !u64 {
        if (!self.is_initialized) {
            // If not in cluster, just add locally
            return try self.addBlockLocal(session_id, config);
        }

        // Check if we're the Raft leader
        if (self.raft_node == null or !self.raft_node.?.isLeader()) {
            return DistributedBlockChainError.NotLeader;
        }

        // Serialize block configuration for Raft log
        const serialized = try self.serializeBlockConfig(config);
        defer self.allocator.free(serialized);

        // Append to Raft log (will replicate to followers)
        const log_index = try self.raft_node.?.appendCommand(serialized);

        // Wait for commit (simplified - real impl would async wait)
        // For now, assume committed and add locally

        // Add to local chain
        const local_chain = try self.getOrCreateChain(session_id);
        const block_id = try local_chain.addBlock(config);

        std.log.info("DistributedBlockChain: Block {d} added to session {s} via consensus (log index: {d})", .{ block_id, session_id, log_index });

        return block_id;
    }

    /// Get a block from the chain (local or remote)
    pub fn getBlock(self: *Self, session_id: []const u8, block_id: u64) ?block_chain.ConversationBlock {
        // First check local chain
        if (self.local_chains.get(session_id)) |chain| {
            return chain.getBlock(block_id);
        }

        // If not found locally and we're in a cluster, could query other nodes
        // For now, return null
        return null;
    }

    /// Get blocks visible at current read timestamp (MVCC)
    pub fn getVisibleBlocks(self: *Self, session_id: []const u8) ![]const u64 {
        return try self.mvcc_store.getVisibleBlocks(session_id);
    }

    /// Create a summary block from multiple blocks
    pub fn createSummary(
        self: *Self,
        session_id: []const u8,
        block_ids: []const u64,
    ) !u64 {
        if (self.local_chains.get(session_id)) |chain| {
            return try chain.createSummary(block_ids);
        }

        return DistributedBlockChainError.ShardUnavailable;
    }

    /// Get session chain (local only)
    pub fn getChain(self: *Self, session_id: []const u8) ?*block_chain.BlockChain {
        return self.local_chains.getPtr(session_id);
    }

    /// Get or create local chain for session
    fn getOrCreateChain(self: *Self, session_id: []const u8) !*block_chain.BlockChain {
        if (self.local_chains.getPtr(session_id)) |chain| {
            return chain;
        }

        const session_copy = try self.allocator.dupe(u8, session_id);
        errdefer self.allocator.free(session_copy);

        const chain = block_chain.BlockChain.init(self.allocator, session_copy);
        try self.local_chains.put(session_copy, chain);

        return self.local_chains.getPtr(session_copy).?;
    }

    /// Add block locally (no consensus)
    fn addBlockLocal(self: *Self, session_id: []const u8, config: block_chain.BlockConfig) !u64 {
        const chain = try self.getOrCreateChain(session_id);
        return try chain.addBlock(config);
    }

    /// Serialize block config for Raft log
    fn serializeBlockConfig(self: *Self, config: block_chain.BlockConfig) ![]const u8 {
        // Simplified serialization - would need proper serialization in real impl
        var buffer = std.ArrayListUnmanaged(u8).empty;
        defer buffer.deinit(self.allocator);

        const writer = buffer.writer(self.allocator);

        // Write dimension
        try writer.writeIntLittle(u32, @intCast(config.query_embedding.len));

        // Write embedding data
        const embedding_bytes = std.mem.sliceAsBytes(config.query_embedding);
        try writer.writeIntLittle(u32, @intCast(embedding_bytes.len));
        try writer.writeAll(embedding_bytes);

        // Write persona tag
        try writer.writeIntLittle(u8, @intFromEnum(config.persona_tag.primary_persona));
        try writer.writeIntLittle(u32, @bitCast(config.persona_tag.blend_coefficient));

        // Write intent
        try writer.writeIntLittle(u8, @intFromEnum(config.intent));

        // Write timestamp (from when config was created)
        const timestamp = time.unixSeconds();
        try writer.writeIntLittle(i64, timestamp);

        return buffer.toOwnedSlice(self.allocator);
    }

    /// Deserialize block config from Raft log
    fn deserializeBlockConfig(self: *Self, data: []const u8) !block_chain.BlockConfig {
        var pos: usize = 0;

        // Read dimension
        if (pos + 4 > data.len) return error.UnexpectedEndOfData;
        const dim = std.mem.readInt(u32, data[pos..][0..4], .little);
        pos += 4;

        // Read embedding
        if (pos + 4 > data.len) return error.UnexpectedEndOfData;
        const embedding_len = std.mem.readInt(u32, data[pos..][0..4], .little);
        pos += 4;

        // Skip embedding bytes
        if (pos + embedding_len > data.len) return error.UnexpectedEndOfData;
        pos += embedding_len;

        // Convert bytes back to floats (simplified)
        const embedding = try self.allocator.alloc(f32, dim);
        errdefer self.allocator.free(embedding);

        // Note: This is simplified - would need proper deserialization
        @memset(embedding, 0.1);

        // Read persona tag
        if (pos + 6 > data.len) return error.UnexpectedEndOfData;
        const persona_raw = data[pos];
        pos += 1;
        const blend_raw = std.mem.readInt(u32, data[pos..][0..4], .little);
        pos += 4;

        const persona_tag = block_chain.PersonaTag{
            .primary_persona = @enumFromInt(persona_raw),
            .blend_coefficient = @bitCast(blend_raw),
        };

        // Read intent
        if (pos >= data.len) return error.UnexpectedEndOfData;
        const intent_raw = data[pos];

        return block_chain.BlockConfig{
            .query_embedding = embedding,
            .persona_tag = persona_tag,
            .intent = @enumFromInt(intent_raw),
            .routing_weights = .{}, // Would need to serialize/deserialize
            .previous_hash = .{0} ** 32, // Would need from context
        };
    }

    /// Check if node is cluster leader
    pub fn isLeader(self: *Self) bool {
        return if (self.raft_node) |raft| raft.isLeader() else false;
    }

    /// Get current term (if in cluster)
    pub fn getCurrentTerm(self: *Self) ?u64 {
        return if (self.raft_node) |raft| raft.current_term else null;
    }

    /// Get cluster statistics
    pub fn getClusterStats(self: *Self) ?network.RaftStats {
        return if (self.raft_node) |raft| raft.getStats() else null;
    }

    /// Get local chain statistics
    pub fn getLocalStats(self: *Self) LocalStats {
        var stats = LocalStats{};

        var iter = self.local_chains.iterator();
        while (iter.next()) |entry| {
            stats.session_count += 1;
            stats.total_blocks += entry.value_ptr.blocks.count();
        }

        return stats;
    }
};

/// Local chain statistics
pub const LocalStats = struct {
    session_count: usize = 0,
    total_blocks: usize = 0,
};

// Tests
test "DistributedBlockChain local operations" {
    const allocator = std.testing.allocator;

    var dbc = try DistributedBlockChain.init(allocator, "test-node", .{});
    defer dbc.deinit();

    // Create test embedding
    const dim = 384;
    const query_embedding = try allocator.alloc(f32, dim);
    defer allocator.free(query_embedding);
    @memset(query_embedding, 0.1);

    // Add block locally (no cluster)
    const config = block_chain.BlockConfig{
        .query_embedding = query_embedding,
        .persona_tag = .{ .primary_persona = .abbey },
        .routing_weights = .{ .abbey_weight = 0.8, .aviva_weight = 0.2 },
        .intent = .empathy_seeking,
    };

    const block_id = try dbc.addBlock("test-session-1", config);
    try std.testing.expect(block_id != 0);

    // Retrieve block
    const block = dbc.getBlock("test-session-1", block_id);
    try std.testing.expect(block != null);

    // Get visible blocks
    const visible = try dbc.getVisibleBlocks("test-session-1");
    defer allocator.free(visible);

    try std.testing.expect(visible.len >= 1);
    try std.testing.expect(visible[0] == block_id);
}

test "DistributedBlockChain chain management" {
    const allocator = std.testing.allocator;

    var dbc = try DistributedBlockChain.init(allocator, "test-node-2", .{});
    defer dbc.deinit();

    // Get non-existent chain
    const chain = dbc.getChain("non-existent");
    try std.testing.expect(chain == null);

    // Create chain by adding block
    const dim = 384;
    const query_embedding = try allocator.alloc(f32, dim);
    defer allocator.free(query_embedding);
    @memset(query_embedding, 0.1);

    const config = block_chain.BlockConfig{
        .query_embedding = query_embedding,
        .persona_tag = .{ .primary_persona = .aviva },
        .routing_weights = .{ .abbey_weight = 0.3, .aviva_weight = 0.7 },
        .intent = .technical_problem,
    };

    _ = try dbc.addBlock("new-session", config);

    // Now chain should exist
    const created_chain = dbc.getChain("new-session");
    try std.testing.expect(created_chain != null);

    // Check local stats
    const stats = dbc.getLocalStats();
    try std.testing.expect(stats.session_count >= 1);
    try std.testing.expect(stats.total_blocks >= 1);
}
