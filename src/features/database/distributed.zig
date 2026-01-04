//! Distributed database wrapper combining sharding and replication.
const std = @import("std");
const database = @import("database.zig");
const shard = @import("shard.zig");
const replication = @import("replication.zig");
const reindex = @import("reindex.zig");
const index = @import("index.zig");

pub const DistributedConfig = struct {
    shard_count: usize = 16,
    replica_count: usize = 3,
    enable_auto_reindex: bool = true,
    enable_replication: bool = true,
    consistency_level: replication.ConsistencyLevel = .quorum,
    node_weight: u32 = 100,
};

pub const DistributedDatabase = struct {
    allocator: std.mem.Allocator,
    config: DistributedConfig,
    local_shard: ?*shard.ShardRouter,
    local_database: ?*database.Database,
    reindexer: ?*reindex.AutoReindexer,
    index_manager: *index.IndexManager,
    is_distributed: bool,
    cluster_nodes: std.StringArrayHashMapUnmanaged(ClusterNodeInfo),

    const ClusterNodeInfo = struct {
        node_id: []const u8,
        address: []const u8,
        port: u16,
        status: NodeStatus,
        last_heartbeat: i64,
    };

    const NodeStatus = enum {
        online,
        offline,
        joining,
        leaving,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        config: DistributedConfig,
        db: *database.Database,
    ) !DistributedDatabase {
        const idx_manager = try allocator.create(index.IndexManager);
        errdefer allocator.destroy(idx_manager);

        idx_manager.* = index.IndexManager.init(.{
            .index_type = .hnsw,
            .auto_rebuild = true,
        });

        var dist_db = DistributedDatabase{
            .allocator = allocator,
            .config = config,
            .local_shard = null,
            .local_database = null,
            .reindexer = null,
            .index_manager = idx_manager,
            .is_distributed = false,
            .cluster_nodes = std.StringArrayHashMapUnmanaged(ClusterNodeInfo).empty,
        };

        if (config.shard_count > 1) {
            dist_db.is_distributed = true;
            const router = try allocator.create(shard.ShardRouter);
            errdefer allocator.destroy(router);

            router.* = try shard.ShardRouter.init(allocator, .{
                .shard_count = config.shard_count,
                .replica_count = @as(u32, @intCast(config.replica_count)),
                .enable_auto_researching = true,
            });
            dist_db.local_shard = router;

            if (config.enable_replication) {
                const repl = try allocator.create(replication.ReplicationManager);
                errdefer allocator.destroy(repl);

                repl.* = replication.ReplicationManager.init(allocator, .{
                    .replica_count = config.replica_count,
                    .write_quorum = @divFloor(config.replica_count, 2) + 1,
                    .read_quorum = 1,
                    .consistency_level = config.consistency_level,
                });
            }

            if (config.enable_auto_reindex) {
                const auto_reindex = try allocator.create(reindex.AutoReindexer);
                errdefer allocator.destroy(auto_reindex);

                auto_reindex.* = try reindex.AutoReindexer.init(allocator, .{}, idx_manager);
                dist_db.reindexer = auto_reindex;
            }
        }

        return dist_db;
    }

    pub fn deinit(self: *DistributedDatabase) void {
        if (self.reindexer) |ri| {
            ri.stop();
            ri.deinit();
            self.allocator.destroy(ri);
        }

        if (self.local_shard) |router| {
            router.deinit();
            self.allocator.destroy(router);
        }

        self.index_manager.deinit(self.allocator);
        self.allocator.destroy(self.index_manager);

        var it = self.cluster_nodes.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.cluster_nodes.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn joinCluster(self: *DistributedDatabase, node_id: []const u8, address: []const u8, port: u16) !void {
        const id_copy = try self.allocator.dupe(u8, node_id);
        errdefer self.allocator.free(id_copy);

        const addr_copy = try self.allocator.dupe(u8, address);
        errdefer self.allocator.free(addr_copy);

        const node_info = ClusterNodeInfo{
            .node_id = id_copy,
            .address = addr_copy,
            .port = port,
            .status = .joining,
            .last_heartbeat = std.time.timestamp(),
        };

        try self.cluster_nodes.put(self.allocator, id_copy, node_info);

        if (self.local_shard) |router| {
            try router.addNode(node_id, self.config.node_weight);
        }

        if (self.cluster_nodes.get(node_id)) |*info| {
            info.status = .online;
        }
    }

    pub fn leaveCluster(self: *DistributedDatabase, node_id: []const u8) void {
        if (self.cluster_nodes.remove(node_id)) |entry| {
            self.allocator.free(entry.key_ptr.*);

            if (self.local_shard) |router| {
                router.removeNode(node_id) catch {};
            }
        }
    }

    pub fn insert(
        self: *DistributedDatabase,
        id: u64,
        vector: []const f32,
        metadata: ?[]const u8,
    ) !void {
        if (self.is_distributed and self.local_shard) |router| {
            try router.routeInsert(id, vector, metadata);
        } else {
            _ = vector;
            _ = metadata;
        }
    }

    pub fn search(
        self: *DistributedDatabase,
        allocator: std.mem.Allocator,
        query: []const f32,
        top_k: usize,
    ) ![]database.SearchResult {
        if (self.is_distributed and self.local_shard) |router| {
            const key = std.fmt.allocPrint(allocator, "vec:{d}", .{query.len}) catch {
                return error.OutOfMemory;
            };
            defer allocator.free(key);

            return try router.routeSearch(key, top_k);
        }

        return allocator.alloc(database.SearchResult, 0);
    }

    pub fn delete(self: *DistributedDatabase, id: u64) !bool {
        if (self.is_distributed and self.local_shard) |router| {
            return try router.routeDelete(id);
        }
        return false;
    }

    pub fn get(self: *DistributedDatabase, id: u64) ?database.VectorView {
        _ = id;
        return null;
    }

    pub fn getStats(self: *DistributedDatabase) DistributedStats {
        var stats = DistributedStats{
            .is_distributed = self.is_distributed,
            .shard_count = self.config.shard_count,
            .replica_count = self.config.replica_count,
            .node_count = self.cluster_nodes.count(),
            .shard_stats = undefined,
            .replication_stats = undefined,
        };

        if (self.local_shard) |router| {
            const router_stats = router.getStats();
            stats.shard_stats = router_stats;
        }

        return stats;
    }

    pub fn getNodeStatus(self: *DistributedDatabase, node_id: []const u8) ?*const ClusterNodeInfo {
        return self.cluster_nodes.get(node_id);
    }

    pub fn updateHeartbeat(self: *DistributedDatabase, node_id: []const u8) void {
        if (self.cluster_nodes.getPtr(node_id)) |info| {
            info.last_heartbeat = std.time.timestamp();
        }
    }

    pub fn detectFailedNodes(self: *DistributedDatabase) []const []const u8 {
        var failed = std.ArrayListUnmanaged([]const u8).empty;
        const now = std.time.timestamp();
        const timeout = 30;

        var it = self.cluster_nodes.iterator();
        while (it.next()) |entry| {
            if (now - entry.value_ptr.last_heartbeat > timeout) {
                entry.value_ptr.status = .offline;
                try failed.append(self.allocator, entry.key_ptr.*);
            }
        }

        return failed.toOwnedSlice(self.allocator);
    }
}

pub const DistributedStats = struct {
    is_distributed: bool,
    shard_count: usize,
    replica_count: usize,
    node_count: usize,
    shard_stats: shard.RouterStats,
    replication_stats: replication.ReplicationMetrics,
};

test "distributed database init" {
    const allocator = std.testing.allocator;
    var db = try database.Database.init(allocator, "test");
    defer db.deinit();

    var dist_db = try DistributedDatabase.init(allocator, .{}, &db);
    defer dist_db.deinit();

    try std.testing.expect(!dist_db.is_distributed);
}

test "distributed database cluster join" {
    const allocator = std.testing.allocator;
    var db = try database.Database.init(allocator, "test");
    defer db.deinit();

    var dist_db = try DistributedDatabase.init(allocator, .{ .shard_count = 4 }, &db);
    defer dist_db.deinit();

    try dist_db.joinCluster("node1", "192.168.1.1", 8080);

    try std.testing.expectEqual(@as(usize, 1), dist_db.cluster_nodes.count());
}

test "distributed database stats" {
    const allocator = std.testing.allocator;
    var db = try database.Database.init(allocator, "test");
    defer db.deinit();

    var dist_db = try DistributedDatabase.init(allocator, .{
        .shard_count = 4,
        .replica_count = 2,
    }, &db);
    defer dist_db.deinit();

    try dist_db.joinCluster("node1", "192.168.1.1", 8080);
    try dist_db.joinCluster("node2", "192.168.1.2", 8080);

    const stats = dist_db.getStats();
    try std.testing.expect(stats.is_distributed);
    try std.testing.expectEqual(@as(usize, 2), stats.node_count);
}
