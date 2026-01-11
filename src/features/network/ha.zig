//! High availability mechanisms
//!
//! Provides failover, health checks, and leader election
//! for distributed compute clusters.

const std = @import("std");

pub const HaError = error{
    NoHealthyNodes,
    ElectionFailed,
    ClusterUnstable,
    ConfigError,
};

pub const NodeHealth = enum {
    healthy,
    unhealthy,
    degraded,
    unknown,
};

pub const ClusterState = enum {
    forming,
    stable,
    unstable,
    partitioned,
    degraded,
};

pub const HealthCheckResult = struct {
    node_id: []const u8,
    healthy: bool,
    response_time_ms: u64,
    error_message: ?[]const u8,
};

pub const FailoverPolicy = enum {
    automatic,
    manual,
    disable,
};

pub const ClusterConfig = struct {
    health_check_interval_ms: u64 = 5_000,
    health_check_timeout_ms: u64 = 2_000,
    max_failed_checks: u8 = 3,
    failover_policy: FailoverPolicy = .automatic,
    election_timeout_ms: u64 = 30_000,
};

pub const HealthCheck = struct {
    allocator: std.mem.Allocator,
    config: ClusterConfig,
    node_health: std.StringHashMap(NodeHealth),
    cluster_state: ClusterState,
    primary_node: ?[]const u8 = null,
    check_count: std.StringHashMap(u8),

    pub fn init(allocator: std.mem.Allocator, config: ClusterConfig) !HealthCheck {
        return HealthCheck{
            .allocator = allocator,
            .config = config,
            .node_health = std.StringHashMap(NodeHealth).init(allocator),
            .cluster_state = .forming,
            .check_count = std.StringHashMap(u8).init(allocator),
        };
    }

    pub fn deinit(self: *HealthCheck) void {
        self.node_health.deinit();
        self.check_count.deinit();
        self.* = undefined;
    }

    pub fn addNode(self: *HealthCheck, node_id: []const u8) !void {
        try self.node_health.put(node_id, .unknown);
        try self.check_count.put(node_id, 0);
    }

    pub fn removeNode(self: *HealthCheck, node_id: []const u8) void {
        _ = self.node_health.remove(node_id);
        _ = self.check_count.remove(node_id);

        if (self.primary_node) |primary| {
            if (std.mem.eql(u8, primary, node_id)) {
                self.primary_node = null;
                self.cluster_state = .forming;
            }
        }
    }

    pub fn reportHealth(self: *HealthCheck, result: HealthCheckResult) !void {
        try self.node_health.put(result.node_id, if (result.healthy) .healthy else .unhealthy);

        const count_ptr = try self.check_count.getOrPut(result.node_id, 0);
        if (result.healthy) {
            count_ptr.value_ptr.* = 0;
        } else {
            count_ptr.value_ptr.* +%= 1;
        }

        try self.evaluateClusterState();
    }

    pub fn getHealthyNodes(self: *const HealthCheck) !std.ArrayListUnmanaged([]const u8) {
        var list = std.ArrayListUnmanaged([]const u8){};
        errdefer list.deinit(self.allocator);

        var iter = self.node_health.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* == .healthy) {
                try list.append(self.allocator, entry.key_ptr.*);
            }
        }

        return list;
    }

    pub fn getUnhealthyNodes(self: *const HealthCheck) !std.ArrayListUnmanaged([]const u8) {
        var list = std.ArrayListUnmanaged([]const u8){};
        errdefer list.deinit(self.allocator);

        var iter = self.node_health.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* == .unhealthy) {
                try list.append(self.allocator, entry.key_ptr.*);
            }
        }

        return list;
    }

    pub fn isNodeHealthy(self: *const HealthCheck, node_id: []const u8) bool {
        const health = self.node_health.get(node_id) orelse return false;
        return health == .healthy;
    }

    pub fn electPrimary(self: *HealthCheck) ![]const u8 {
        const new_primary = try self.selectNewPrimary();
        return new_primary;
    }

    pub fn triggerFailover(self: *HealthCheck) !void {
        if (self.config.failover_policy != .automatic) {
            return HaError.ConfigError;
        }
        _ = try self.selectNewPrimary();
    }

    /// Internal helper to select and set a new primary from healthy nodes.
    /// Frees the previous primary if one exists.
    fn selectNewPrimary(self: *HealthCheck) ![]const u8 {
        var healthy = try self.getHealthyNodes();
        defer healthy.deinit(self.allocator);

        if (healthy.items.len == 0) {
            return HaError.NoHealthyNodes;
        }

        const new_primary = healthy.items[0];

        // Free existing primary before setting new one
        if (self.primary_node) |current| {
            self.allocator.free(current);
        }

        self.primary_node = try self.allocator.dupe(u8, new_primary);
        self.cluster_state = .stable;

        return new_primary;
    }

    fn evaluateClusterState(self: *HealthCheck) !void {
        var healthy_count: usize = 0;
        var unhealthy_count: usize = 0;
        var unknown_count: usize = 0;

        var iter = self.node_health.valueIterator();
        while (iter.next()) |health| {
            switch (health.*) {
                .healthy => healthy_count += 1,
                .unhealthy => unhealthy_count += 1,
                .unknown => unknown_count += 1,
                .degraded => healthy_count += 1,
            }
        }

        const total_nodes = healthy_count + unhealthy_count + unknown_count;

        if (unhealthy_count > total_nodes / 2) {
            self.cluster_state = .unstable;
        } else if (unhealthy_count == 0 and unknown_count == 0) {
            self.cluster_state = .stable;
        } else {
            self.cluster_state = .degraded;
        }
    }
};

test "health check tracking" {
    const allocator = std.testing.allocator;

    var health_check = try HealthCheck.init(allocator, .{});
    defer health_check.deinit();

    try health_check.addNode("node-1");
    try health_check.addNode("node-2");
    try health_check.addNode("node-3");

    const result = HealthCheckResult{
        .node_id = "node-1",
        .healthy = true,
        .response_time_ms = 50,
        .error_message = null,
    };

    try health_check.reportHealth(result);

    try std.testing.expect(health_check.isNodeHealthy("node-1"));
    try std.testing.expect(!health_check.isNodeHealthy("node-2"));

    const primary = try health_check.electPrimary();
    try std.testing.expectEqualStrings("node-1", primary);
    allocator.free(primary);
}
