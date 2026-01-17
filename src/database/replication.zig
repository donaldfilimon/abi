//! Database replication manager for distributed vector database.
const std = @import("std");
const database = @import("database.zig");

pub const ReplicationConfig = struct {
    replica_count: usize = 3,
    write_quorum: usize = 2,
    read_quorum: usize = 1,
    consistency_level: ConsistencyLevel = .quorum,
    sync_replication: bool = true,
    replication_factor: usize = 3,
};

pub const ConsistencyLevel = enum {
    one,
    quorum,
    all,
};

pub const ReplicaState = enum {
    leader,
    follower,
    candidate,
    offline,
    syncing,
};

pub const ReplicaInfo = struct {
    node_id: []const u8,
    state: ReplicaState,
    last_sync_time: i64,
    lag_bytes: u64,
    lag_records: usize,
    is_healthy: bool,
};

pub const ReplicationMetrics = struct {
    total_replicas: usize = 0,
    healthy_replicas: usize = 0,
    leaders: usize = 0,
    followers: usize = 0,
    offline_replicas: usize = 0,
    replication_lag_ms: u64 = 0,
    sync_operations: u64 = 0,
    failed_operations: u64 = 0,
};

pub const ReplicationManager = struct {
    allocator: std.mem.Allocator,
    config: ReplicationConfig,
    metrics: ReplicationMetrics,
    replicas: std.StringArrayHashMapUnmanaged(ReplicaInfo),
    leader_for_shard: std.AutoHashMapUnmanaged(u32, []const u8),
    shard_replicas: std.AutoHashMapUnmanaged(u32, std.ArrayListUnmanaged([]const u8)),
    pending_syncs: std.ArrayListUnmanaged(PendingSync),
    running: std.atomic.Value(bool),
    thread: ?std.Thread = null,

    const PendingSync = struct {
        shard_id: u32,
        source_node: []const u8,
        target_node: []const u8,
        start_record: u64,
        end_record: u64,
        start_time: i64,
    };

    pub fn init(allocator: std.mem.Allocator, config: ReplicationConfig) ReplicationManager {
        return .{
            .allocator = allocator,
            .config = config,
            .metrics = .{},
            .replicas = std.StringArrayHashMapUnmanaged(ReplicaInfo).empty,
            .leader_for_shard = .{},
            .shard_replicas = .{},
            .pending_syncs = std.ArrayListUnmanaged(PendingSync).empty,
            .running = std.atomic.Value(bool).init(false),
            .thread = null,
        };
    }

    pub fn deinit(self: *ReplicationManager) void {
        self.stop();

        var it = self.replicas.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.replicas.deinit(self.allocator);

        var shard_it = self.shard_replicas.iterator();
        while (shard_it.next()) |entry| {
            for (entry.value_ptr.items) |replica| {
                self.allocator.free(replica);
            }
            entry.value_ptr.deinit(self.allocator);
        }
        self.shard_replicas.deinit(self.allocator);
        self.leader_for_shard.deinit(self.allocator);

        self.pending_syncs.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn addReplica(
        self: *ReplicationManager,
        node_id: []const u8,
        shard_id: u32,
        is_leader: bool,
    ) !void {
        const id_copy = try self.allocator.dupe(u8, node_id);
        errdefer self.allocator.free(id_copy);

        const state: ReplicaState = if (is_leader) .leader else .follower;
        const replica_info = ReplicaInfo{
            .node_id = id_copy,
            .state = state,
            .last_sync_time = 0,
            .lag_bytes = 0,
            .lag_records = 0,
            .is_healthy = true,
        };

        try self.replicas.put(self.allocator, id_copy, replica_info);

        if (is_leader) {
            try self.leader_for_shard.put(self.allocator, shard_id, id_copy);
        }

        var shard_list = self.shard_replicas.get(shard_id) orelse blk: {
            var list = std.ArrayListUnmanaged([]const u8).empty;
            try list.append(self.allocator, id_copy);
            try self.shard_replicas.put(self.allocator, shard_id, list);
            break :blk list;
        };
        try shard_list.append(self.allocator, id_copy);

        self.updateMetrics();
    }

    pub fn removeReplica(self: *ReplicationManager, node_id: []const u8, shard_id: u32) bool {
        if (self.replicas.remove(node_id)) |entry| {
            self.allocator.free(entry.key_ptr.*);

            if (self.leader_for_shard.get(shard_id)) |leader| {
                if (std.mem.eql(u8, leader, node_id)) {
                    _ = self.leader_for_shard.remove(shard_id);
                }
            }

            var shard_list = self.shard_replicas.get(shard_id);
            if (shard_list) |*list| {
                var i: usize = 0;
                while (i < list.items.len) {
                    if (std.mem.eql(u8, list.items[i], node_id)) {
                        self.allocator.free(list.items[i]);
                        _ = list.swapRemove(i);
                    } else {
                        i += 1;
                    }
                }
            }

            self.updateMetrics();
            return true;
        }
        return false;
    }

    pub fn promoteReplica(self: *ReplicationManager, node_id: []const u8, shard_id: u32) !void {
        if (self.replicas.getPtr(node_id)) |replica| {
            replica.state = .leader;
        }

        try self.leader_for_shard.put(self.allocator, shard_id, node_id);
        self.updateMetrics();
    }

    pub fn replicateWrite(
        self: *ReplicationManager,
        shard_id: u32,
        _: []const u8,
    ) !ReplicationResult {
        const replicas = self.shard_replicas.get(shard_id) orelse
            return error.NoReplicasForShard;

        if (replicas.items.len == 0) return error.NoReplicasForShard;

        const quorum = @min(self.config.write_quorum, replicas.items.len);
        var success_count: usize = 0;
        var failed_nodes = std.ArrayListUnmanaged([]const u8).empty;
        defer failed_nodes.deinit(self.allocator);

        for (replicas.items) |node_id| {
            const replica = self.replicas.get(node_id);
            if (replica) |r| {
                if (r.is_healthy) {
                    success_count += 1;
                } else {
                    try failed_nodes.append(self.allocator, node_id);
                }
            }
        }

        if (success_count < quorum) {
            self.metrics.failed_operations += 1;
            return error.QuorumNotAchieved;
        }

        self.metrics.sync_operations += 1;
        return .{
            .success_count = success_count,
            .quorum = quorum,
            .failed_nodes = failed_nodes.toOwnedSlice(self.allocator),
        };
    }

    pub fn replicateRead(
        self: *ReplicationManager,
        shard_id: u32,
    ) !ReplicationReadResult {
        const replicas = self.shard_replicas.get(shard_id) orelse
            return error.NoReplicasForShard;

        const leader = self.leader_for_shard.get(shard_id);
        if (leader) |leader_id| {
            const replica = self.replicas.get(leader_id);
            if (replica) |r| {
                if (r.is_healthy) {
                    return .{ .node_id = leader_id, .is_leader = true };
                }
            }
        }

        for (replicas.items) |node_id| {
            const replica = self.replicas.get(node_id);
            if (replica) |r| {
                if (r.is_healthy) {
                    return .{ .node_id = node_id, .is_leader = false };
                }
            }
        }

        return error.NoHealthyReplicas;
    }

    pub fn start(self: *ReplicationManager) !void {
        if (self.running.load(.acquire)) return;
        self.running.store(true, .release);
    }

    pub fn stop(self: *ReplicationManager) void {
        if (!self.running.load(.acquire)) return;
        self.running.store(false, .release);
        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }
    }

    pub fn getReplicaInfo(self: *ReplicationManager, node_id: []const u8) ?*const ReplicaInfo {
        return self.replicas.get(node_id);
    }

    pub fn getAllReplicaInfo(self: *ReplicationManager) []const ReplicaInfo {
        return self.replicas.values();
    }

    pub fn getMetrics(self: *ReplicationManager) ReplicationMetrics {
        return self.metrics;
    }

    fn updateMetrics(self: *ReplicationManager) void {
        self.metrics.total_replicas = self.replicas.count();
        self.metrics.healthy_replicas = 0;
        self.metrics.leaders = 0;
        self.metrics.followers = 0;
        self.metrics.offline_replicas = 0;

        for (self.replicas.values()) |replica| {
            switch (replica.state) {
                .leader => self.metrics.leaders += 1,
                .follower => self.metrics.followers += 1,
                else => {},
            }
            if (replica.is_healthy) {
                self.metrics.healthy_replicas += 1;
            } else {
                self.metrics.offline_replicas += 1;
            }
        }
    }
};

pub const ReplicationResult = struct {
    success_count: usize,
    quorum: usize,
    failed_nodes: []const []const u8,
};

pub const ReplicationReadResult = struct {
    node_id: []const u8,
    is_leader: bool,
};

pub const ReplicationError = error{
    NoReplicasForShard,
    QuorumNotAchieved,
    NoHealthyReplicas,
    LeaderNotAvailable,
};

test "replication manager init" {
    const allocator = std.testing.allocator;
    var repl = ReplicationManager.init(allocator, .{ .replica_count = 3 });
    defer repl.deinit();

    try std.testing.expectEqual(@as(usize, 0), repl.metrics.total_replicas);
}

test "replication manager add replica" {
    const allocator = std.testing.allocator;
    var repl = ReplicationManager.init(allocator, .{ .replica_count = 3 });
    defer repl.deinit();

    try repl.addReplica("node1", 0, true);
    try repl.addReplica("node2", 0, false);

    try std.testing.expectEqual(@as(usize, 2), repl.metrics.total_replicas);
    try std.testing.expectEqual(@as(usize, 1), repl.metrics.leaders);
    try std.testing.expectEqual(@as(usize, 1), repl.metrics.followers);
}

test "replication manager promote" {
    const allocator = std.testing.allocator;
    var repl = ReplicationManager.init(allocator, .{ .replica_count = 3 });
    defer repl.deinit();

    try repl.addReplica("node1", 0, true);
    try repl.addReplica("node2", 0, false);

    try repl.promoteReplica("node2", 0);

    const leader = repl.leader_for_shard.get(0);
    try std.testing.expect(leader != null);
    try std.testing.expectEqualStrings("node2", leader.?);
}
