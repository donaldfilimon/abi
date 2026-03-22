//! Shard Manager for Distributed WDBX
//!
//! Implements intelligent sharding strategy for distributing conversation blocks
//! across cluster nodes. Based on research section 2.1.1: "Intelligent Sharding
//! and Latency Model".
//!
//! Sharding Strategy:
//! - Tenant → session → semantic clustering hierarchy
//! - Consistent hashing ring for data placement
//! - Locality-aware replication based on network topology
//! - Dynamic rebalancing with minimal data movement

const std = @import("std");
const parent = @import("./mod.zig");
const time = parent.time;
const network = parent.network;

pub const ShardManagerError = error{
    NodeNotFound,
    InvalidShardKey,
    InsufficientReplicas,
    ShardMigrationFailed,
    ConsensusTimeout,
    TopologyConflict,
    OutOfMemory,
};

/// Shard identifier based on consistent hashing
pub const ShardId = struct {
    hash: u64,
    replica_set: []const []const u8, // Node IDs responsible for this shard

    pub fn getPrimaryNode(self: *const ShardId) []const u8 {
        return self.replica_set[0];
    }

    pub fn getReplicaNodes(self: *const ShardId) []const []const u8 {
        return self.replica_set[1..];
    }
};

/// Sharding configuration
pub const ShardConfig = struct {
    /// Number of virtual nodes per physical node in consistent hash ring
    virtual_nodes_per_node: u32 = 100,
    /// Replication factor (minimum 3 for fault tolerance)
    replication_factor: u32 = 3,
    /// Enable anti-entropy for replica synchronization
    enable_anti_entropy: bool = true,
    /// Anti-entropy interval in seconds
    anti_entropy_interval_s: u64 = 300,
    /// Enable dynamic rebalancing based on load
    enable_dynamic_rebalancing: bool = true,
    /// Load threshold for triggering rebalance (%)
    rebalance_threshold_pct: f32 = 20.0,
    /// Enable locality-aware placement
    enable_locality_aware: bool = true,
    /// Maximum network latency for locality (ms)
    max_locality_latency_ms: u64 = 50,
};

/// Shard key for block placement decision
pub const ShardKey = struct {
    /// Tenant identifier (organization/account)
    tenant_id: u64,
    /// Session identifier within tenant
    session_hash: u64,
    /// Semantic cluster hash (based on embeddings similarity)
    semantic_cluster_hash: ?u64 = null,
    /// Timestamp for temporal locality
    timestamp: i64,

    /// Compute shard key from conversation context
    pub fn fromConversation(
        tenant_id: u64,
        session_id: []const u8,
        query_embedding: []const f32,
        timestamp: i64,
    ) ShardKey {
        // Hash session ID
        const session_hash = std.hash.Fnv1a_64.hash(session_id);

        // Compute semantic cluster hash if embedding provided
        var semantic_hash: ?u64 = null;
        if (query_embedding.len >= 8) {
            // Use first 8 floats of embedding for quick hash
            const bytes = std.mem.sliceAsBytes(query_embedding[0..8]);
            semantic_hash = std.hash.Fnv1a_64.hash(bytes);
        }

        return ShardKey{
            .tenant_id = tenant_id,
            .session_hash = session_hash,
            .semantic_cluster_hash = semantic_hash,
            .timestamp = timestamp,
        };
    }

    /// Compute consistent hash for shard placement
    pub fn computeHash(self: ShardKey) u64 {
        var hasher = std.hash.XxHash3.init(0);

        // Hash tenant and session
        hasher.update(std.mem.asBytes(&self.tenant_id));
        hasher.update(std.mem.asBytes(&self.session_hash));

        // Include semantic cluster if available
        if (self.semantic_cluster_hash) |cluster_hash| {
            hasher.update(std.mem.asBytes(&cluster_hash));
        }

        // Temporal component for load distribution
        const time_bucket = @divFloor(self.timestamp, 3600); // Hourly buckets
        hasher.update(std.mem.asBytes(&time_bucket));

        return hasher.final();
    }
};

/// Virtual node in consistent hash ring
pub const VirtualNode = struct {
    node_id: []const u8,
    physical_node: []const u8,
    hash_position: u64,
    capacity_score: f32, // 0.0-1.0 based on node capacity

    pub fn init(
        allocator: std.mem.Allocator,
        physical_node: []const u8,
        hash_position: u64,
        capacity_score: f32,
    ) !VirtualNode {
        const node_id = try std.fmt.allocPrint(allocator, "{s}:{d}", .{ physical_node, hash_position });

        return VirtualNode{
            .node_id = node_id,
            .physical_node = physical_node,
            .hash_position = hash_position,
            .capacity_score = capacity_score,
        };
    }

    pub fn deinit(self: *VirtualNode, allocator: std.mem.Allocator) void {
        allocator.free(self.node_id);
        allocator.free(self.physical_node);
    }
};

/// Consistent hash ring for shard placement
pub const HashRing = struct {
    allocator: std.mem.Allocator,
    config: ShardConfig,
    virtual_nodes: std.ArrayListUnmanaged(VirtualNode),
    sorted_positions: std.ArrayListUnmanaged(u64), // Sorted hash positions for binary search

    const Self = @This();

    /// Initialize hash ring with node list
    pub fn init(allocator: std.mem.Allocator, config: ShardConfig) !Self {
        const ring = Self{
            .allocator = allocator,
            .config = config,
            .virtual_nodes = .empty,
            .sorted_positions = .empty,
        };

        return ring;
    }

    pub fn deinit(self: *Self) void {
        for (self.virtual_nodes.items) |*node| {
            node.deinit(self.allocator);
        }
        self.virtual_nodes.deinit(self.allocator);
        self.sorted_positions.deinit(self.allocator);
    }

    /// Add a physical node to the hash ring
    pub fn addNode(self: *Self, node_id: []const u8, capacity_score: f32) !void {
        // Create virtual nodes for this physical node
        for (0..self.config.virtual_nodes_per_node) |i| {
            // Distribute virtual nodes evenly around ring
            const base_hash = std.hash.Fnv1a_64.hash(node_id);
            const position_hash = base_hash ^ @as(u64, @intCast(i * 2654435761)); // Fibonacci hash

            const node_id_copy = try self.allocator.dupe(u8, node_id);
            var vnode = VirtualNode.init(self.allocator, node_id_copy, position_hash, capacity_score) catch |err| {
                self.allocator.free(node_id_copy);
                return err;
            };
            self.virtual_nodes.append(self.allocator, vnode) catch |err| {
                vnode.deinit(self.allocator);
                return err;
            };
        }

        // Re-sort positions
        try self.sortPositions();
    }

    /// Remove a physical node from the hash ring
    pub fn removeNode(self: *Self, node_id: []const u8) void {
        var i: usize = 0;
        while (i < self.virtual_nodes.items.len) {
            if (std.mem.eql(u8, self.virtual_nodes.items[i].physical_node, node_id)) {
                self.virtual_nodes.items[i].deinit(self.allocator);
                _ = self.virtual_nodes.swapRemove(i);
            } else {
                i += 1;
            }
        }

        // Re-sort positions
        self.sortPositions() catch {
            // On error, clear and rebuild
            self.sorted_positions.clearRetainingCapacity();
        };
    }

    /// Find shard placement for a key
    pub fn findShard(self: *const Self, key_hash: u64) !ShardId {
        if (self.virtual_nodes.items.len == 0) {
            return ShardManagerError.NodeNotFound;
        }

        // Binary search for closest virtual node clockwise
        const idx = self.findClosestNode(key_hash);
        const primary_vnode = self.virtual_nodes.items[idx];

        // Collect replica set (including primary)
        var replica_set = std.ArrayListUnmanaged([]const u8).empty;
        defer replica_set.deinit(self.allocator);

        try replica_set.append(self.allocator, primary_vnode.physical_node);

        // Find additional replicas for redundancy
        var replica_count: usize = 1;
        var current_idx = (idx + 1) % self.virtual_nodes.items.len;

        while (replica_count < self.config.replication_factor and
            current_idx != idx)
        {
            const replica_vnode = self.virtual_nodes.items[current_idx];

            // Skip if same physical node (already have replica)
            if (!std.mem.eql(u8, replica_vnode.physical_node, primary_vnode.physical_node)) {
                try replica_set.append(self.allocator, replica_vnode.physical_node);
                replica_count += 1;
            }

            current_idx = (current_idx + 1) % self.virtual_nodes.items.len;
        }

        // If we couldn't find enough distinct nodes, use what we have
        if (replica_count < self.config.replication_factor) {
            std.log.warn("HashRing: Could only find {d} distinct nodes for replication factor {d}", .{ replica_count, self.config.replication_factor });
        }

        const replica_slice = try replica_set.toOwnedSlice(self.allocator);

        return ShardId{
            .hash = key_hash,
            .replica_set = replica_slice,
        };
    }

    /// Find closest virtual node clockwise for a given hash
    fn findClosestNode(self: *const Self, hash: u64) usize {
        // Binary search through sorted positions
        var low: usize = 0;
        var high: usize = self.sorted_positions.items.len;

        while (low < high) {
            const mid = (low + high) / 2;
            if (self.sorted_positions.items[mid] < hash) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        // Wrap around if at end
        return if (low < self.sorted_positions.items.len) low else 0;
    }

    /// Sort virtual node positions for binary search
    fn sortPositions(self: *Self) !void {
        self.sorted_positions.clearRetainingCapacity();

        // Collect positions
        for (self.virtual_nodes.items) |vnode| {
            try self.sorted_positions.append(self.allocator, vnode.hash_position);
        }

        // Sort positions
        std.sort.sort(u64, self.sorted_positions.items, {}, struct {
            fn lessThan(_: void, a: u64, b: u64) bool {
                return a < b;
            }
        }.lessThan);
    }

    /// Get load distribution statistics
    pub fn getLoadStats(self: *const Self) LoadStats {
        var stats = LoadStats{};

        // Count virtual nodes per physical node
        var node_counts: std.StringHashMapUnmanaged(usize) = .empty;
        defer node_counts.deinit(self.allocator);

        for (self.virtual_nodes.items) |vnode| {
            const count = node_counts.get(vnode.physical_node) orelse 0;
            node_counts.put(self.allocator, vnode.physical_node, count + 1) catch continue;
        }

        // Calculate statistics
        var iter = node_counts.iterator();
        while (iter.next()) |entry| {
            stats.total_virtual_nodes += entry.value_ptr.*;
            stats.physical_node_count += 1;

            if (entry.value_ptr.* > stats.max_vnodes_per_node) {
                stats.max_vnodes_per_node = entry.value_ptr.*;
            }
            if (entry.value_ptr.* < stats.min_vnodes_per_node or stats.min_vnodes_per_node == 0) {
                stats.min_vnodes_per_node = entry.value_ptr.*;
            }
        }

        if (stats.physical_node_count > 0) {
            stats.avg_vnodes_per_node = @as(f32, @floatFromInt(stats.total_virtual_nodes)) /
                @as(f32, @floatFromInt(stats.physical_node_count));
            stats.imbalance_ratio = if (stats.min_vnodes_per_node > 0)
                @as(f32, @floatFromInt(stats.max_vnodes_per_node)) /
                    @as(f32, @floatFromInt(stats.min_vnodes_per_node))
            else
                1.0;
        }

        return stats;
    }
};

/// Load distribution statistics
pub const LoadStats = struct {
    total_virtual_nodes: usize = 0,
    physical_node_count: usize = 0,
    max_vnodes_per_node: usize = 0,
    min_vnodes_per_node: usize = 0,
    avg_vnodes_per_node: f32 = 0.0,
    imbalance_ratio: f32 = 0.0,
};

/// Shard manager main interface
pub const ShardManager = struct {
    allocator: std.mem.Allocator,
    config: ShardConfig,
    hash_ring: HashRing,
    node_registry: *network.NodeRegistry,

    // Topology awareness for locality placement
    node_latencies: std.StringHashMapUnmanaged(u64), // Node -> average latency (ms)
    node_regions: std.StringHashMapUnmanaged([]const u8), // Node -> region/zone

    const Self = @This();

    /// Initialize shard manager with node registry
    pub fn init(
        allocator: std.mem.Allocator,
        config: ShardConfig,
        node_registry: *network.NodeRegistry,
    ) !Self {
        var ring = try HashRing.init(allocator, config);

        // Initialize all existing nodes in registry
        const nodes = node_registry.list();
        for (nodes) |node| {
            try ring.addNode(node.node_id, 1.0); // Default capacity
        }

        return Self{
            .allocator = allocator,
            .config = config,
            .hash_ring = ring,
            .node_registry = node_registry,
            .node_latencies = .empty,
            .node_regions = .empty,
        };
    }

    pub fn deinit(self: *Self) void {
        self.hash_ring.deinit();

        var iter = self.node_latencies.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.node_latencies.deinit(self.allocator);

        var region_iter = self.node_regions.iterator();
        while (region_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.node_regions.deinit(self.allocator);
    }

    /// Determine shard placement for a conversation block
    pub fn placeBlock(
        self: *Self,
        tenant_id: u64,
        session_id: []const u8,
        query_embedding: []const f32,
    ) !ShardId {
        const timestamp = time.unixSeconds();

        // Create shard key
        const shard_key = ShardKey.fromConversation(tenant_id, session_id, query_embedding, timestamp);

        // Compute hash and find shard
        const key_hash = shard_key.computeHash();
        const shard_id = try self.hash_ring.findShard(key_hash);

        // If locality-aware placement enabled, optimize replica selection
        if (self.config.enable_locality_aware) {
            try self.optimizeForLocality(&shard_id);
        }

        return shard_id;
    }

    /// Optimize replica selection for network locality
    fn optimizeForLocality(self: *Self, shard_id: *ShardId) !void {
        if (shard_id.replica_set.len < 2) return;

        // Reorder replicas based on latency measurements
        var replicas = try self.allocator.alloc([]const u8, shard_id.replica_set.len);
        defer self.allocator.free(replicas);
        @memcpy(replicas[0..shard_id.replica_set.len], shard_id.replica_set[0..shard_id.replica_set.len]);

        // Sort by measured latency (if available)
        std.sort.sort([]const u8, replicas, {}, struct {
            fn lessThan(ctx: *Self, a: []const u8, b: []const u8) bool {
                const latency_a = ctx.node_latencies.get(a) orelse std.math.maxInt(u64);
                const latency_b = ctx.node_latencies.get(b) orelse std.math.maxInt(u64);
                return latency_a < latency_b;
            }
        }.lessThan.bind(self));

        // Update replica set order
        @memcpy(shard_id.replica_set[0..replicas.len], replicas[0..replicas.len]);
    }

    /// Update node latency measurements
    pub fn updateNodeLatency(self: *Self, node_id: []const u8, latency_ms: u64) !void {
        const node_copy = try self.allocator.dupe(u8, node_id);

        // Store or update latency
        if (self.node_latencies.getPtr(node_copy)) |latency_ptr| {
            // Exponential moving average: new = 0.8*old + 0.2*current
            latency_ptr.* = @as(u64, @intFromFloat(0.8 * @as(f64, @floatFromInt(latency_ptr.*)) +
                0.2 * @as(f64, @floatFromInt(latency_ms))));
        } else {
            try self.node_latencies.put(self.allocator, node_copy, latency_ms);
        }
    }

    /// Set node region/zone for locality awareness
    pub fn setNodeRegion(self: *Self, node_id: []const u8, region: []const u8) !void {
        const node_copy = try self.allocator.dupe(u8, node_id);
        const region_copy = try self.allocator.dupe(u8, region);

        try self.node_regions.put(self.allocator, node_copy, region_copy);
    }

    /// Handle node join (add to hash ring)
    pub fn handleNodeJoin(self: *Self, node_id: []const u8) !void {
        try self.hash_ring.addNode(node_id, 1.0);
        std.log.info("ShardManager: Node {s} joined cluster, added to hash ring", .{node_id});
    }

    /// Handle node leave (remove from hash ring)
    pub fn handleNodeLeave(self: *Self, node_id: []const u8) void {
        self.hash_ring.removeNode(node_id);

        // Clean up latency/region records
        if (self.node_latencies.fetchRemove(node_id)) |kv| {
            self.allocator.free(kv.key);
        }
        if (self.node_regions.fetchRemove(node_id)) |kv| {
            self.allocator.free(kv.key);
            self.allocator.free(kv.value);
        }

        std.log.info("ShardManager: Node {s} left cluster, removed from hash ring", .{node_id});
    }

    /// Check if rebalancing is needed based on load
    pub fn checkRebalanceNeeded(self: *const Self) bool {
        if (!self.config.enable_dynamic_rebalancing) return false;

        const stats = self.hash_ring.getLoadStats();
        return stats.imbalance_ratio > (1.0 + self.config.rebalance_threshold_pct / 100.0);
    }

    /// Get cluster load statistics
    pub fn getClusterStats(self: *const Self) LoadStats {
        return self.hash_ring.getLoadStats();
    }
};

// Tests
test "ShardKey computation" {
    const allocator = std.testing.allocator;

    // Create test embedding
    const embedding = try allocator.alloc(f32, 384);
    defer allocator.free(embedding);
    @memset(embedding, 0.1);

    const key = try ShardKey.fromConversation(allocator, 12345, "test-session-abc", embedding, time.unixSeconds());

    const hash = key.computeHash();
    try std.testing.expect(hash != 0);
}

test "HashRing shard placement" {
    const allocator = std.testing.allocator;

    var ring = try HashRing.init(allocator, .{
        .virtual_nodes_per_node = 10,
        .replication_factor = 2,
    });
    defer ring.deinit();

    // Add nodes
    try ring.addNode("node-a", 1.0);
    try ring.addNode("node-b", 1.0);
    try ring.addNode("node-c", 1.0);

    // Find shard placement
    const shard = try ring.findShard(123456789);

    try std.testing.expect(shard.hash == 123456789);
    try std.testing.expect(shard.replica_set.len >= 1);
    try std.testing.expect(shard.replica_set.len <= 3);

    allocator.free(shard.replica_set);
}

test "ShardManager block placement" {
    const allocator = std.testing.allocator;

    // Create mock node registry
    var registry = try network.NodeRegistry.init(allocator);
    defer registry.deinit();

    try registry.register("node-1", "127.0.0.1:9000");
    try registry.register("node-2", "127.0.0.1:9001");

    // Create shard manager
    var manager = try ShardManager.init(allocator, .{
        .virtual_nodes_per_node = 5,
        .replication_factor = 2,
    }, &registry);
    defer manager.deinit();

    // Create test embedding
    const embedding = try allocator.alloc(f32, 384);
    defer allocator.free(embedding);
    @memset(embedding, 0.1);

    // Place block
    const shard_id = try manager.placeBlock(1001, "session-xyz", embedding);
    defer allocator.free(shard_id.replica_set);

    try std.testing.expect(shard_id.replica_set.len >= 1);

    // Verify nodes in replica set are from our registry
    for (shard_id.replica_set) |node| {
        const node_found = for (registry.list()) |reg_node| {
            if (std.mem.eql(u8, reg_node.node_id, node)) break true;
        } else false;
        try std.testing.expect(node_found);
    }
}
