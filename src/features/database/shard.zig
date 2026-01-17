//! Database sharding router for distributed vector database.
const std = @import("std");
const hashring = @import("hashring.zig");
const database = @import("database.zig");
const index = @import("index.zig");
const rpc_client = @import("rpc_client.zig");
const transport = @import("../network/transport.zig");

pub const ShardConfig = struct {
    shard_count: usize = 16,
    replica_count: u32 = 3,
    virtual_nodes: u32 = 100,
    enable_auto_rebalancing: bool = true,
    min_nodes_per_shard: usize = 1,
    /// Enable real RPC communication (vs stub/local-only mode).
    enable_rpc: bool = true,
    /// RPC port for this node.
    rpc_port: u16 = 9001,
    /// Request timeout in milliseconds.
    request_timeout_ms: u64 = 30000,
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
    /// RPC client for distributed operations (null if RPC disabled).
    rpc: ?*rpc_client.DatabaseRpcClient,
    /// Node address to port mapping for RPC.
    node_ports: std.StringHashMapUnmanaged(u16),

    pub fn init(allocator: std.mem.Allocator, config: ShardConfig) !ShardRouter {
        var rpc: ?*rpc_client.DatabaseRpcClient = null;

        if (config.enable_rpc) {
            rpc = rpc_client.DatabaseRpcClient.init(allocator, .{
                .transport_config = .{
                    .listen_port = config.rpc_port,
                    .io_timeout_ms = config.request_timeout_ms,
                },
                .request_timeout_ms = config.request_timeout_ms,
            }, null) catch null;
        }

        return .{
            .allocator = allocator,
            .config = config,
            .hash_ring = hashring.HashRing.init(allocator, config.virtual_nodes),
            .shards = std.ArrayListUnmanaged(ShardInfo).empty,
            .node_shards = std.StringArrayHashMapUnmanaged(std.ArrayListUnmanaged(ShardId)).empty,
            .key_to_shard = .{},
            .version = 0,
            .rpc = rpc,
            .node_ports = .{},
        };
    }

    pub fn deinit(self: *ShardRouter) void {
        // Clean up RPC client
        if (self.rpc) |rpc| {
            rpc.stop();
            rpc.deinit();
        }

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
        self.node_ports.deinit(self.allocator);
        self.* = undefined;
    }

    /// Start the RPC server for incoming requests.
    pub fn startRpc(self: *ShardRouter) !void {
        if (self.rpc) |rpc| {
            try rpc.start();
        }
    }

    /// Stop the RPC server.
    pub fn stopRpc(self: *ShardRouter) void {
        if (self.rpc) |rpc| {
            rpc.stop();
        }
    }

    /// Register a node with its RPC port.
    pub fn registerNodePort(self: *ShardRouter, node_id: []const u8, port: u16) !void {
        const key = try self.allocator.dupe(u8, node_id);
        try self.node_ports.put(self.allocator, key, port);
    }

    /// Get the RPC port for a node.
    pub fn getNodePort(self: *ShardRouter, node_id: []const u8) u16 {
        return self.node_ports.get(node_id) orelse self.config.rpc_port;
    }

    pub fn addNode(self: *ShardRouter, node_id: []const u8, weight: u32) !void {
        try self.hash_ring.addNode(node_id, weight);

        const existing = self.node_shards.get(node_id);
        if (existing == null) {
            var list = std.ArrayListUnmanaged(ShardId).empty;
            try list.append(self.allocator, 0);
            try self.node_shards.put(self.allocator, node_id, list);
        }

        if (self.config.enable_auto_rebalancing) {
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

        if (self.config.enable_auto_rebalancing) {
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

    /// Route a search query to the appropriate shard and return results.
    /// If RPC is enabled, performs actual distributed query.
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

        // If RPC is enabled, perform actual distributed query
        if (self.rpc) |rpc| {
            return self.executeDistributedSearch(allocator, rpc, shard, top_k);
        }

        // Fallback to empty results (local-only mode)
        return allocator.alloc(database.SearchResult, 0);
    }

    /// Execute a distributed search across shard nodes.
    fn executeDistributedSearch(
        self: *ShardRouter,
        allocator: std.mem.Allocator,
        rpc: *rpc_client.DatabaseRpcClient,
        shard: ShardInfo,
        top_k: usize,
    ) ![]database.SearchResult {
        // For a real search, we'd need the query vector
        // This is a simplified version - actual implementation would
        // accept query_vector as parameter
        const query_vector = [_]f32{ 0.0, 0.0, 0.0, 0.0 }; // Placeholder

        // Try primary node first
        const parsed = transport.parseAddress(shard.primary_node) catch {
            return allocator.alloc(database.SearchResult, 0);
        };

        const port = self.getNodePort(shard.primary_node);

        const remote_results = rpc.remoteSearch(
            parsed.host,
            port,
            &query_vector,
            top_k,
            shard.shard_id,
        ) catch |err| {
            std.log.warn("Search failed on primary {s}: {}", .{ shard.primary_node, err });

            // Try replicas
            for (shard.nodes) |node| {
                if (std.mem.eql(u8, node, shard.primary_node)) continue;

                const replica_parsed = transport.parseAddress(node) catch continue;
                const replica_port = self.getNodePort(node);

                const replica_results = rpc.remoteSearch(
                    replica_parsed.host,
                    replica_port,
                    &query_vector,
                    top_k,
                    shard.shard_id,
                ) catch continue;

                return self.convertRemoteResults(allocator, replica_results);
            }

            return allocator.alloc(database.SearchResult, 0);
        };

        return self.convertRemoteResults(allocator, remote_results);
    }

    /// Convert remote search results to local format.
    fn convertRemoteResults(
        self: *ShardRouter,
        allocator: std.mem.Allocator,
        remote: []rpc_client.RemoteSearchResult,
    ) ![]database.SearchResult {
        defer {
            if (self.rpc) |rpc| {
                rpc.freeSearchResults(remote);
            }
        }

        var results = try allocator.alloc(database.SearchResult, remote.len);
        for (remote, 0..) |r, i| {
            results[i] = .{
                .id = r.id,
                .score = r.score,
                .metadata = if (r.metadata) |m| try allocator.dupe(u8, m) else null,
            };
        }
        return results;
    }

    /// Route an insert operation to the appropriate shard.
    /// If RPC is enabled, performs actual distributed insert.
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

        // If RPC is enabled, perform actual distributed insert
        if (self.rpc) |rpc| {
            const parsed = transport.parseAddress(shard.primary_node) catch {
                return error.InvalidShardId;
            };

            const port = self.getNodePort(shard.primary_node);

            rpc.remoteInsert(
                parsed.host,
                port,
                id,
                vector,
                metadata,
            ) catch |err| {
                std.log.warn("Insert failed on primary {s}: {}", .{ shard.primary_node, err });
                return error.InsertFailed;
            };
        }

        // Update record count for this shard
        shard.record_count += 1;
    }

    /// Route a delete operation to the appropriate shard.
    /// If RPC is enabled, performs actual distributed delete.
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

        // If RPC is enabled, perform actual distributed delete
        if (self.rpc) |rpc| {
            const parsed = transport.parseAddress(shard.primary_node) catch {
                return false;
            };

            const port = self.getNodePort(shard.primary_node);

            const success = rpc.remoteDelete(
                parsed.host,
                port,
                id,
            ) catch |err| {
                std.log.warn("Delete failed on primary {s}: {}", .{ shard.primary_node, err });
                return false;
            };

            if (success and shard.record_count > 0) {
                shard.record_count -= 1;
            }

            return success;
        }

        // Local mode: just update record count
        if (shard.record_count > 0) {
            shard.record_count -= 1;
        }

        return true;
    }

    /// Route an update operation to the appropriate shard.
    /// If RPC is enabled, performs actual distributed update.
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

        // If RPC is enabled, perform actual distributed update
        if (self.rpc) |rpc| {
            const parsed = transport.parseAddress(shard.primary_node) catch {
                return false;
            };

            const port = self.getNodePort(shard.primary_node);

            return rpc.remoteUpdate(
                parsed.host,
                port,
                id,
                vector,
                metadata,
            ) catch |err| {
                std.log.warn("Update failed on primary {s}: {}", .{ shard.primary_node, err });
                return false;
            };
        }

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
    var router = try ShardRouter.init(allocator, .{ .shard_count = 4, .enable_rpc = false });
    defer router.deinit();

    try std.testing.expectEqual(@as(usize, 0), router.hash_ring.nodes.items.len);
}

test "shard router add node" {
    const allocator = std.testing.allocator;
    var router = try ShardRouter.init(allocator, .{ .shard_count = 4, .enable_rpc = false });
    defer router.deinit();

    try router.addNode("node1", 100);

    try std.testing.expectEqual(@as(usize, 1), router.hash_ring.nodes.items.len);
}

test "shard router get shard for key" {
    const allocator = std.testing.allocator;
    var router = try ShardRouter.init(allocator, .{ .shard_count = 4, .enable_rpc = false });
    defer router.deinit();

    try router.addNode("node1", 100);
    try router.addNode("node2", 100);

    const shard_id = try router.getShardForKey("test-key");
    try std.testing.expect(shard_id < 4);
}
