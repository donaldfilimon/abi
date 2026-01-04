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
    key_to_shard: std.AutoHashMap(u128, ShardId),
    version: u64,

    pub fn init(allocator: std.mem.Allocator, config: ShardConfig) !ShardRouter {
        return .{
            .allocator = allocator,
            .config = config,
            .hash_ring = hashring.HashRing.init(allocator, config.virtual_nodes),
            .shards = std.ArrayListUnmanaged(ShardInfo).empty,
            .node_shards = std.StringArrayHashMapUnmanaged(std.ArrayListUnmanaged(ShardId)).empty,
            .key_to_shard = std.AutoHashMap(u128, ShardId).init(allocator),
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
        self.key_to_shard.deinit();
        self.* = undefined;
    }

    pub fn addNode(self: *ShardRouter, node_id: []const u8, weight: u32) !void {
        try self.hash_ring.addNode(node_id, weight);

        var existing = self.node_shards.get(node_id);
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

    pub fn routeSearch(
        self: *ShardRouter,
        key: []const u8,
        top_k: usize,
    ) !std.ArrayListUnmanaged(database.SearchResult) {
        var results = std.ArrayListUnmanaged(database.SearchResult).empty;

        const shard_id = try self.getShardForKey(key);
        const shard = &self.shards.items[shard_id];

        for (shard.nodes) |node_id| {
            _ = node_id;
        }

        return results;
    }

    pub fn routeInsert(
        self: *ShardRouter,
        id: u64,
        vector: []const f32,
        metadata: ?[]const u8,
    ) !void {
        const key = std.fmt.allocPrint(self.allocator, "vec:{d}", .{id}) catch return;
        defer self.allocator.free(key);

        const shard_id = try self.getShardForKey(key);
        const shard = &self.shards.items[shard_id];

        _ = shard;
        _ = vector;
        _ = metadata;
    }

    pub fn routeDelete(self: *ShardRouter, id: u64) !bool {
        const key = std.fmt.allocPrint(self.allocator, "vec:{d}", .{id}) catch return false;
        defer self.allocator.free(key);

        const shard_id = try self.getShardForKey(key);
        if (shard_id >= self.shards.items.len) return false;

        return true;
    }

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

            const nodes = self.hash_ring.getNodes(start_key, self.config.replica_count) catch |err| {
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
}

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
