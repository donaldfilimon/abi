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
    CommitTimeout,
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
    /// TCP listen port for transport
    listen_port: u16 = 9000,
    /// Maximum poll iterations to wait for Raft commit
    commit_timeout_polls: u32 = 1000,
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
    local_chains: std.StringHashMapUnmanaged(block_chain.BlockChain), // session_id -> chain
    mvcc_store: block_chain.MvccStore,

    // Distributed coordination
    raft_node: ?network.RaftNode,
    transport: ?*network.TcpTransport,
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
            .local_chains = .empty,
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
        self.local_chains.deinit(self.allocator);

        // Clean up distributed components
        if (self.raft_node) |*raft| {
            raft.deinit();
        }
        if (self.transport) |transport| {
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

        // Initialize network transport with configurable port
        self.transport = try network.TcpTransport.init(self.allocator, .{
            .listen_address = "0.0.0.0",
            .listen_port = self.config.listen_port,
        });

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

        if (self.transport) |transport| {
            transport.deinit();
            self.transport = null;
        }

        self.is_initialized = false;
        std.log.info("DistributedBlockChain: Node {s} stopped", .{self.node_id});
    }

    /// Add a block to the distributed chain (requires consensus)
    ///
    /// When running in a cluster, the block is first appended to the Raft log.
    /// The method then polls the Raft commit index to confirm the entry has been
    /// committed by a quorum before applying it to local storage.  In single-node
    /// mode (no peers) the leader auto-commits, so the poll succeeds immediately.
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

        // Poll for Raft commit confirmation
        var committed = false;
        var poll_count: u32 = 0;
        while (poll_count < self.config.commit_timeout_polls) : (poll_count += 1) {
            if (self.raft_node.?.commit_index >= log_index) {
                committed = true;
                break;
            }
            // In single-node mode the leader auto-commits; no peers to wait for.
            if (self.raft_node.?.peers.count() == 0) {
                // Advance commit index directly for solo leader
                self.raft_node.?.commit_index = log_index;
                committed = true;
                break;
            }
        }
        if (!committed) return DistributedBlockChainError.CommitTimeout;

        // Raft entry is committed — apply to local chain
        const local_chain = try self.getOrCreateChain(session_id);
        const block_id = try local_chain.addBlock(config);

        std.log.info("DistributedBlockChain: Block {d} committed to session {s} (log index: {d}, polls: {d})", .{ block_id, session_id, log_index, poll_count });

        return block_id;
    }

    /// Get a block from the chain (local or remote).
    ///
    /// First checks local storage. If the block is not found locally and the
    /// node is part of a cluster with an active TCP transport, a best-effort
    /// remote query is sent to the Raft leader (or first available peer) using
    /// the `db_search_request` message type.  If the transport is unavailable
    /// or no peer responds, `null` is returned.
    pub fn getBlock(self: *Self, session_id: []const u8, block_id: u64) ?block_chain.ConversationBlock {
        // First check local chain
        if (self.local_chains.get(session_id)) |chain| {
            return chain.getBlock(block_id);
        }

        // Attempt remote query if we are in a cluster with transport
        if (self.is_initialized) {
            if (self.transport) |transport| {
                if (self.raft_node) |raft| {
                    // Build a lightweight query payload: session_id + block_id
                    var query_buf: [256]u8 = undefined;
                    const session_len: u32 = @intCast(session_id.len);
                    var pos: usize = 0;

                    // Write session length (4 bytes LE)
                    @memcpy(query_buf[pos..][0..4], &std.mem.toBytes(session_len));
                    pos += 4;

                    // Write session id
                    if (pos + session_id.len <= query_buf.len) {
                        @memcpy(query_buf[pos..][0..session_id.len], session_id);
                        pos += session_id.len;
                    } else {
                        return null; // session_id too long for buffer
                    }

                    // Write block_id (8 bytes LE)
                    @memcpy(query_buf[pos..][0..8], &std.mem.toBytes(block_id));
                    pos += 8;

                    const payload = query_buf[0..pos];

                    // Determine target: prefer the known leader, else first peer
                    const target_id = raft.leader_id orelse blk: {
                        var peer_iter = raft.peers.keyIterator();
                        break :blk if (peer_iter.next()) |k| k.* else null;
                    };

                    if (target_id) |peer_addr| {
                        // Use the configured listen port for the remote node
                        const response_data = transport.sendRequest(
                            peer_addr,
                            self.config.listen_port,
                            .db_search_request,
                            payload,
                        ) catch |err| {
                            std.log.warn("DistributedBlockChain: Remote block query failed for session {s} block {d}: {t}", .{ session_id, block_id, err });
                            return null;
                        };
                        defer self.allocator.free(response_data);

                        // NOTE: Full deserialization of a remote block response
                        // requires a wire-format contract between nodes. For now
                        // we log the attempt and fall through to null — a future
                        // iteration will decode db_search_response payloads into
                        // ConversationBlock values.
                        std.log.debug("DistributedBlockChain: Remote query sent for session {s} block {d}, got {d} bytes", .{ session_id, block_id, response_data.len });
                    }
                }
            } else {
                std.log.debug("DistributedBlockChain: No transport available for remote block query", .{});
            }
        }

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
        try self.local_chains.put(self.allocator, session_copy, chain);

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
        errdefer buffer.deinit(self.allocator);

        // Write dimension
        try buffer.appendSlice(self.allocator, &std.mem.toBytes(@as(u32, @intCast(config.query_embedding.len))));

        // Write embedding data
        const embedding_bytes = std.mem.sliceAsBytes(config.query_embedding);
        try buffer.appendSlice(self.allocator, &std.mem.toBytes(@as(u32, @intCast(embedding_bytes.len))));
        try buffer.appendSlice(self.allocator, embedding_bytes);

        // Write persona tag
        try buffer.append(self.allocator, @intFromEnum(config.persona_tag.primary_persona));
        try buffer.appendSlice(self.allocator, &std.mem.toBytes(@as(u32, @bitCast(config.persona_tag.blend_coefficient))));

        // Write intent
        try buffer.append(self.allocator, @intFromEnum(config.intent));

        // Write timestamp (from when config was created)
        const timestamp = time.unixSeconds();
        try buffer.appendSlice(self.allocator, &std.mem.toBytes(timestamp));

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

test "DistributedBlockChain single-node addBlock auto-commits" {
    const allocator = std.testing.allocator;

    var dbc = try DistributedBlockChain.init(allocator, "solo-node", .{});
    defer dbc.deinit();

    // Initialize Raft without peers (single-node cluster)
    dbc.raft_node = try network.RaftNode.init(allocator, "solo-node", dbc.config.raft_config);
    dbc.is_initialized = true;

    // Force leader state so appendCommand succeeds
    dbc.raft_node.?.state = .leader;
    dbc.raft_node.?.current_term = 1;

    const dim = 64;
    const embedding = try allocator.alloc(f32, dim);
    defer allocator.free(embedding);
    @memset(embedding, 0.5);

    const config = block_chain.BlockConfig{
        .query_embedding = embedding,
        .persona_tag = .{ .primary_persona = .abbey },
        .routing_weights = .{ .abbey_weight = 1.0 },
        .intent = .empathy_seeking,
    };

    // Single-node: should auto-commit without timeout
    const block_id = try dbc.addBlock("auto-session", config);
    try std.testing.expect(block_id != 0);

    // Verify Raft commit index was advanced
    try std.testing.expect(dbc.raft_node.?.commit_index >= 1);
}

test "DistributedBlockChain getBlock returns locally added block" {
    const allocator = std.testing.allocator;

    var dbc = try DistributedBlockChain.init(allocator, "local-get-node", .{});
    defer dbc.deinit();

    const dim = 64;
    const embedding = try allocator.alloc(f32, dim);
    defer allocator.free(embedding);
    @memset(embedding, 0.3);

    const config = block_chain.BlockConfig{
        .query_embedding = embedding,
        .persona_tag = .{ .primary_persona = .aviva },
        .routing_weights = .{ .aviva_weight = 1.0 },
        .intent = .technical_problem,
    };

    const block_id = try dbc.addBlock("get-session", config);
    const block = dbc.getBlock("get-session", block_id);
    try std.testing.expect(block != null);
}

test "DistributedBlockChain getBlock non-existent returns null" {
    const allocator = std.testing.allocator;

    var dbc = try DistributedBlockChain.init(allocator, "null-get-node", .{});
    defer dbc.deinit();

    // Query a session/block that was never added
    const block = dbc.getBlock("no-such-session", 999);
    try std.testing.expect(block == null);
}

test {
    std.testing.refAllDecls(@This());
}
