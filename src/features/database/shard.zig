//! Database sharding router for distributed vector database.
const std = @import("std");
const hashring = @import("hashring.zig");
const database = @import("database.zig");
const index = @import("index.zig");

pub const ShardConfig = struct {
    shard_count: usize = 16,
    replica_count: u32 = 3,
    virtual_nodes: u32 = 100,
    enable_auto_researching: bool = true,
    min_nodes_per_shard: usize = 1,
};

pub const ShardId = u32;

pub const ShardInfo = struct {
    shard_id: ShardId,
    nodes: []const []const u8,
    primary_node: []const u8,
    key_range: struct {
        start: u128,
        end: u128,
    },
    record_count: usize,
    status: ShardStatus,
};

pub const ShardStatus = enum {
    healthy,
    degraded,
    offline,
    resharding,
};

pub const ShardRouter = struct {
    allocator: std.mem.Allocator,
    config: ShardConfig,
    hash_ring: hashring.HashRing,
    shards: std.ArrayListUnmanaged(ShardInfo),
    node_shards: std.StringArrayHashMapUnmanaged(std.ArrayListUnmanaged(ShardId)),
    key_to_shard: std.AutoHashMapUnmanaged(u128, ShardId),
    version: u64,

    pub fn init(allocator: std.mem.Allocator, config: ShardConfig) !ShardRouter {
        return .{
            .allocator = allocator,
            .config = config,
            .hash_ring = hashring.HashRing.init(allocator, config.virtual_nodes),
            .shards = std.ArrayListUnmanaged(ShardInfo).empty,
            .node_shards = std.StringArrayHashMapUnmanaged(std.ArrayListUnmanaged(ShardId)).empty,
            .key_to_shard = .{},
            .version = 0,
        };
    }

    pub fn deinit(self: *ShardRouter) void {
        self.hash_ring.deinit();
        for (self.shards.items) |*shard| {
            self.allocator.free(shard.nodes);
            self.allocator.free(shard.primary_node);
        }
        self.shards.deinit(self.allocator);

        var it = self.node_shards.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.node_shards.deinit(self.allocator);
        self.key_to_shard.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn addNode(self: *ShardRouter, node_id: []const u8, weight: u32) !void {
        try self.hash_ring.addNode(node_id, weight);

        const existing = self.node_shards.get(node_id);
        if (existing == null) {
            var list = std.ArrayListUnmanaged(ShardId).empty;
            try list.append(self.allocator, 0);
            try self.node_shards.put(self.allocator, node_id, list);
        }

        if (self.config.enable_auto_researching) {
            try self.redistributeShards();
        }
        self.version += 1;
    }

    pub fn removeNode(self: *ShardRouter, node_id: []const u8) !void {
        const removed = self.hash_ring.removeNode(node_id);
        if (!removed) return;

        if (self.node_shards.remove(node_id)) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }

        if (self.config.enable_auto_researching) {
            try self.redistributeShards();
        }
        self.version += 1;
    }

    pub fn getShardForKey(self: *ShardRouter, key: []const u8) !ShardId {
        const key_hash = hashKey(key);
        const shard_id = @as(ShardId, @intCast(key_hash % @as(u128, @intCast(self.config.shard_count))));
        return shard_id;
    }

    pub fn getShardNodes(self: *ShardRouter, shard_id: ShardId) ![]const []const u8 {
        if (shard_id >= self.shards.items.len) return error.InvalidShardId;
        return self.shards.items[shard_id].nodes;
    }

    pub fn getPrimaryNode(self: *ShardRouter, shard_id: ShardId) ![]const u8 {
        if (shard_id >= self.shards.items.len) return error.InvalidShardId;
        return self.shards.items[shard_id].primary_node;
    }

    /// Route a search query to the appropriate shard and return results
    pub fn routeSearch(
        self: *ShardRouter,
        allocator: std.mem.Allocator,
        key: []const u8,
        top_k: usize,
    ) ![]database.SearchResult {
        const shard_id = try self.getShardForKey(key);

        // Get the shard info to find which nodes handle this shard
        if (shard_id >= self.shards.items.len) {
            return allocator.alloc(database.SearchResult, 0);
        }

        const shard = self.shards.items[shard_id];

        // Log the routing decision
        std.log.debug("Routing search to shard {d} (primary: {s}, nodes: {d})", .{
            shard_id,
            shard.primary_node,
            shard.nodes.len,
        });

        // In a distributed system, this would:
        // 1. Connect to the primary node (or a healthy replica)
        // 2. Send the search request
        // 3. Aggregate results from multiple nodes if needed
        // 4. Return the combined results

        // For now, return an empty result set
        // A real implementation would forward the query to the shard's nodes
        var results = std.ArrayListUnmanaged(database.SearchResult){};
        errdefer results.deinit(allocator);

        // Simulate query execution on shard
        _ = top_k;

        return try results.toOwnedSlice(allocator);
    }

    /// Route an insert operation to the appropriate shard
    pub fn routeInsert(
        self: *ShardRouter,
        id: u64,
        vector: []const f32,
        metadata: ?[]const u8,
    ) !void {
        const key = try std.fmt.allocPrint(self.allocator, "vec:{d}", .{id});
        defer self.allocator.free(key);

        const shard_id = try self.getShardForKey(key);

        if (shard_id >= self.shards.items.len) {
            return error.InvalidShardId;
        }

        const shard = &self.shards.items[shard_id];

        // Log the routing decision
        std.log.debug("Routing insert to shard {d} (primary: {s})", .{
            shard_id,
            shard.primary_node,
        });

        // In a distributed system, this would:
        // 1. Connect to the primary node for this shard
        // 2. Send the insert request
        // 3. Wait for replication confirmation based on write quorum
        // 4. Return success/failure

        // Update record count for this shard
        shard.record_count += 1;

        // Track the data for potential forwarding
        _ = vector;
        _ = metadata;
    }

    /// Route a delete operation to the appropriate shard
    pub fn routeDelete(self: *ShardRouter, id: u64) !bool {
        const key = try std.fmt.allocPrint(self.allocator, "vec:{d}", .{id});
        defer self.allocator.free(key);

        const shard_id = try self.getShardForKey(key);

        if (shard_id >= self.shards.items.len) {
            return false;
        }

        const shard = &self.shards.items[shard_id];

        // Log the routing decision
        std.log.debug("Routing delete to shard {d} (primary: {s})", .{
            shard_id,
            shard.primary_node,
        });

        // In a distributed system, this would:
        // 1. Connect to the primary node for this shard
        // 2. Send the delete request
        // 3. Wait for replication confirmation
        // 4. Return success/failure

        // Update record count
        if (shard.record_count > 0) {
            shard.record_count -= 1;
        }

        return true;
    }

    /// Route an update operation to the appropriate shard
    pub fn routeUpdate(
        self: *ShardRouter,
        id: u64,
        vector: []const f32,
        metadata: ?[]const u8,
    ) !bool {
        const key = try std.fmt.allocPrint(self.allocator, "vec:{d}", .{id});
        defer self.allocator.free(key);

        const shard_id = try self.getShardForKey(key);

        if (shard_id >= self.shards.items.len) {
            return false;
        }

        const shard = &self.shards.items[shard_id];

        std.log.debug("Routing update to shard {d} (primary: {s})", .{
            shard_id,
            shard.primary_node,
        });

        _ = vector;
        _ = metadata;

        return true;
    }

    /// Get the routing info for a specific vector ID
    pub fn getRoutingInfo(self: *ShardRouter, id: u64) !RoutingInfo {
        const key = try std.fmt.allocPrint(self.allocator, "vec:{d}", .{id});
        defer self.allocator.free(key);

        const shard_id = try self.getShardForKey(key);

        if (shard_id >= self.shards.items.len) {
            return error.InvalidShardId;
        }

        const shard = self.shards.items[shard_id];

        return .{
            .shard_id = shard_id,
            .primary_node = shard.primary_node,
            .replica_nodes = shard.nodes,
            .status = shard.status,
        };
    }

    pub const RoutingInfo = struct {
        shard_id: ShardId,
        primary_node: []const u8,
        replica_nodes: []const []const u8,
        status: ShardStatus,
    };

    pub fn getShardInfo(self: *ShardRouter, shard_id: ShardId) ?*ShardInfo {
        if (shard_id >= self.shards.items.len) return null;
        return &self.shards.items[shard_id];
    }

    pub fn getAllShardInfo(self: *ShardRouter) []const ShardInfo {
        return self.shards.items;
    }

    pub fn redistributeShards(self: *ShardRouter) !void {
        if (self.hash_ring.nodes.items.len == 0) return;

        var new_shards = std.ArrayListUnmanaged(ShardInfo).empty;
        errdefer {
            for (new_shards.items) |*shard| {
                self.allocator.free(shard.nodes);
                self.allocator.free(shard.primary_node);
            }
            new_shards.deinit(self.allocator);
        }

        const shard_range = @as(u128, std.math.maxInt(u128)) / @as(u128, @intCast(self.config.shard_count));
        var shard_id: ShardId = 0;
        while (shard_id < self.config.shard_count) : (shard_id += 1) {
            const start_key = std.fmt.allocPrint(self.allocator, "shard:{d}:start", .{shard_id}) catch return error.OutOfMemory;
            defer self.allocator.free(start_key);

            const end_key = std.fmt.allocPrint(self.allocator, "shard:{d}:end", .{shard_id}) catch return error.OutOfMemory;
            defer self.allocator.free(end_key);

            const nodes = self.hash_ring.getNodes(
                start_key,
                @as(usize, @intCast(self.config.replica_count)),
            ) catch |err| blk: {
                if (err == error.NotEnoughNodes) {
                    const all_nodes = try self.allocator.alloc([]const u8, self.hash_ring.nodes.items.len);
                    for (self.hash_ring.nodes.items, 0..) |node, i| {
                        all_nodes[i] = node.node_id;
                    }
                    break :blk all_nodes;
                }
                return err;
            };

            const primary = if (nodes.len > 0) nodes[0] else "";
            const start_hash = hashKey(start_key);
            const end_hash = start_hash + shard_range;

            const shard_info = ShardInfo{
                .shard_id = shard_id,
                .nodes = nodes,
                .primary_node = try self.allocator.dupe(u8, primary),
                .key_range = .{
                    .start = start_hash,
                    .end = end_hash,
                },
                .record_count = 0,
                .status = .healthy,
            };

            try new_shards.append(self.allocator, shard_info);
        }

        for (self.shards.items) |*old_shard| {
            self.allocator.free(old_shard.nodes);
            self.allocator.free(old_shard.primary_node);
        }
        self.shards = new_shards;
    }

    pub fn getStats(self: *ShardRouter) RouterStats {
        var healthy: usize = 0;
        var degraded: usize = 0;
        var offline: usize = 0;

        for (self.shards.items) |shard| {
            switch (shard.status) {
                .healthy => healthy += 1,
                .degraded => degraded += 1,
                .offline => offline += 1,
                else => {},
            }
        }

        return .{
            .shard_count = self.shards.items.len,
            .node_count = self.hash_ring.nodes.items.len,
            .healthy_shards = healthy,
            .degraded_shards = degraded,
            .offline_shards = offline,
            .version = self.version,
        };
    }
};

pub const RouterStats = struct {
    shard_count: usize,
    node_count: usize,
    healthy_shards: usize,
    degraded_shards: usize,
    offline_shards: usize,
    version: u64,
};

fn hashKey(key: []const u8) u128 {
    var state: std.crypto.hash.sha2.Sha256 = .{};
    state.update(key);
    var result: [32]u8 = undefined;
    state.final(result[0..]);
    return std.mem.readInt(u128, result[0..16], .big);
}

test "shard router initialization" {
    const allocator = std.testing.allocator;
    var router = try ShardRouter.init(allocator, .{ .shard_count = 4 });
    defer router.deinit();

    try std.testing.expectEqual(@as(usize, 0), router.hash_ring.nodes.items.len);
}

test "shard router add node" {
    const allocator = std.testing.allocator;
    var router = try ShardRouter.init(allocator, .{ .shard_count = 4 });
    defer router.deinit();

    try router.addNode("node1", 100);

    try std.testing.expectEqual(@as(usize, 1), router.hash_ring.nodes.items.len);
}

test "shard router get shard for key" {
    const allocator = std.testing.allocator;
    var router = try ShardRouter.init(allocator, .{ .shard_count = 4 });
    defer router.deinit();

    try router.addNode("node1", 100);
    try router.addNode("node2", 100);

    const shard_id = try router.getShardForKey("test-key");
    try std.testing.expect(shard_id < 4);
}
