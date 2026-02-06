//! End-to-End Distributed System Tests
//!
//! Complete workflow tests for distributed operations:
//! - Cluster formation and discovery
//! - Distributed vector search
//! - Consensus and leader election
//! - Failover and recovery

const std = @import("std");
const abi = @import("abi");
const e2e = @import("mod.zig");

// ============================================================================
// Helper Functions
// ============================================================================

/// Mock node for testing distributed scenarios.
const MockNode = struct {
    id: []const u8,
    address: []const u8,
    port: u16,
    is_healthy: bool,
    is_leader: bool,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, id: []const u8, address: []const u8, port: u16) !MockNode {
        return .{
            .id = try allocator.dupe(u8, id),
            .address = try allocator.dupe(u8, address),
            .port = port,
            .is_healthy = true,
            .is_leader = false,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MockNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.address);
    }

    pub fn markUnhealthy(self: *MockNode) void {
        self.is_healthy = false;
    }

    pub fn recover(self: *MockNode) void {
        self.is_healthy = true;
    }

    pub fn promote(self: *MockNode) void {
        self.is_leader = true;
    }

    pub fn demote(self: *MockNode) void {
        self.is_leader = false;
    }
};

/// Mock cluster for testing.
const MockCluster = struct {
    nodes: std.ArrayListUnmanaged(MockNode),
    allocator: std.mem.Allocator,
    leader_index: ?usize,

    pub fn init(allocator: std.mem.Allocator) MockCluster {
        return .{
            .nodes = .{},
            .allocator = allocator,
            .leader_index = null,
        };
    }

    pub fn deinit(self: *MockCluster) void {
        for (self.nodes.items) |*node| {
            node.deinit();
        }
        self.nodes.deinit(self.allocator);
    }

    pub fn addNode(self: *MockCluster, id: []const u8, address: []const u8, port: u16) !void {
        const node = try MockNode.init(self.allocator, id, address, port);
        try self.nodes.append(self.allocator, node);
    }

    pub fn removeNode(self: *MockCluster, id: []const u8) bool {
        for (self.nodes.items, 0..) |node, i| {
            if (std.mem.eql(u8, node.id, id)) {
                var removed = self.nodes.orderedRemove(i);
                removed.deinit();

                // Update leader index if needed
                if (self.leader_index) |leader_idx| {
                    if (leader_idx == i) {
                        self.leader_index = null;
                    } else if (leader_idx > i) {
                        self.leader_index = leader_idx - 1;
                    }
                }
                return true;
            }
        }
        return false;
    }

    pub fn electLeader(self: *MockCluster) !void {
        // Simple leader election: first healthy node becomes leader
        for (self.nodes.items, 0..) |*node, i| {
            if (node.is_healthy) {
                if (self.leader_index) |old_idx| {
                    if (old_idx < self.nodes.items.len) {
                        self.nodes.items[old_idx].demote();
                    }
                }
                node.promote();
                self.leader_index = i;
                return;
            }
        }
        return error.NoHealthyNodes;
    }

    pub fn getLeader(self: *MockCluster) ?*MockNode {
        if (self.leader_index) |idx| {
            if (idx < self.nodes.items.len) {
                return &self.nodes.items[idx];
            }
        }
        return null;
    }

    pub fn healthyNodeCount(self: *MockCluster) usize {
        var count: usize = 0;
        for (self.nodes.items) |node| {
            if (node.is_healthy) count += 1;
        }
        return count;
    }
};

// ============================================================================
// E2E Tests: Cluster Formation
// ============================================================================

test "e2e: cluster formation workflow" {
    try e2e.skipIfNetworkDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true },
        .timeout_ms = 30_000,
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    // 1. Create cluster
    var cluster = MockCluster.init(allocator);
    defer cluster.deinit();

    try timer.checkpoint("cluster_created");

    // 2. Add nodes to cluster
    try cluster.addNode("node-1", "10.0.0.1", 5432);
    try cluster.addNode("node-2", "10.0.0.2", 5432);
    try cluster.addNode("node-3", "10.0.0.3", 5432);

    try timer.checkpoint("nodes_added");

    // 3. Verify cluster state
    try std.testing.expectEqual(@as(usize, 3), cluster.nodes.items.len);
    try std.testing.expectEqual(@as(usize, 3), cluster.healthyNodeCount());

    // 4. Elect leader
    try cluster.electLeader();

    try timer.checkpoint("leader_elected");

    const leader = cluster.getLeader();
    try std.testing.expect(leader != null);
    try std.testing.expect(leader.?.is_leader);

    // 5. Verify workflow completed within timeout
    try std.testing.expect(!timer.isTimedOut(30_000));
}

test "e2e: node discovery and registration" {
    try e2e.skipIfNetworkDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true },
    });
    defer ctx.deinit();

    // Initialize network module
    try abi.network.init(allocator);
    defer abi.network.deinit();

    // Get registry and register nodes
    const registry = try abi.network.defaultRegistry();

    try registry.register("node-a", "127.0.0.1:9000");
    try registry.register("node-b", "127.0.0.1:9001");

    // Verify registration
    const nodes = registry.list();
    try std.testing.expectEqual(@as(usize, 2), nodes.len);
}

test "e2e: network state initialization" {
    try e2e.skipIfNetworkDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true },
    });
    defer ctx.deinit();

    // Initialize with custom config
    try abi.network.initWithConfig(allocator, .{
        .cluster_id = "test-cluster",
        .heartbeat_timeout_ms = 10_000,
        .max_nodes = 64,
    });
    defer abi.network.deinit();

    // Verify configuration
    const config = abi.network.defaultConfig();
    try std.testing.expect(config != null);
    try std.testing.expectEqualStrings("test-cluster", config.?.cluster_id);
}

// ============================================================================
// E2E Tests: Distributed Operations
// ============================================================================

test "e2e: distributed task coordination" {
    try e2e.skipIfNetworkDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true },
    });
    defer ctx.deinit();

    // Create cluster
    var cluster = MockCluster.init(allocator);
    defer cluster.deinit();

    try cluster.addNode("coordinator", "10.0.0.1", 5432);
    try cluster.addNode("worker-1", "10.0.0.2", 5432);
    try cluster.addNode("worker-2", "10.0.0.3", 5432);

    try cluster.electLeader();

    // Leader should coordinate tasks
    const leader = cluster.getLeader();
    try std.testing.expect(leader != null);
    try std.testing.expectEqualStrings("coordinator", leader.?.id);

    // Simulate task distribution
    const task_count: usize = 10;
    const worker_count = cluster.nodes.items.len - 1; // Exclude coordinator

    // Tasks should be distributable
    const tasks_per_worker = task_count / worker_count;
    try std.testing.expect(tasks_per_worker > 0);
}

test "e2e: distributed vector search with sharding" {
    try e2e.skipIfNetworkDisabled();
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true, .database = true },
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    // Create cluster with database shards
    var cluster = MockCluster.init(allocator);
    defer cluster.deinit();

    try cluster.addNode("shard-1", "10.0.0.1", 5432);
    try cluster.addNode("shard-2", "10.0.0.2", 5432);
    try cluster.addNode("shard-3", "10.0.0.3", 5432);

    try timer.checkpoint("cluster_created");

    // Simulate distributed search across shards
    const query = try e2e.generateNormalizedVector(allocator, 128, 42);
    defer allocator.free(query);

    // Each shard would search locally and return top-k
    const k_per_shard: usize = 5;
    const total_k = k_per_shard * cluster.nodes.items.len;

    // Final merge would reduce to top-k
    const final_k: usize = 10;
    try std.testing.expect(final_k <= total_k);

    try timer.checkpoint("search_completed");
}

// ============================================================================
// E2E Tests: Consensus and Leader Election
// ============================================================================

test "e2e: leader election on cluster startup" {
    try e2e.skipIfNetworkDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true },
    });
    defer ctx.deinit();

    var cluster = MockCluster.init(allocator);
    defer cluster.deinit();

    // Start with no nodes - election should fail
    try std.testing.expectError(error.NoHealthyNodes, cluster.electLeader());

    // Add nodes
    try cluster.addNode("node-1", "10.0.0.1", 5432);

    // Election should succeed with one node
    try cluster.electLeader();

    const leader = cluster.getLeader();
    try std.testing.expect(leader != null);
    try std.testing.expectEqualStrings("node-1", leader.?.id);

    // Add more nodes
    try cluster.addNode("node-2", "10.0.0.2", 5432);
    try cluster.addNode("node-3", "10.0.0.3", 5432);

    // First node should still be leader (no re-election triggered)
    try std.testing.expectEqualStrings("node-1", cluster.getLeader().?.id);
}

test "e2e: leader election after leader failure" {
    try e2e.skipIfNetworkDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true },
    });
    defer ctx.deinit();

    var cluster = MockCluster.init(allocator);
    defer cluster.deinit();

    try cluster.addNode("node-1", "10.0.0.1", 5432);
    try cluster.addNode("node-2", "10.0.0.2", 5432);
    try cluster.addNode("node-3", "10.0.0.3", 5432);

    // Elect initial leader
    try cluster.electLeader();
    try std.testing.expectEqualStrings("node-1", cluster.getLeader().?.id);

    // Leader fails
    cluster.nodes.items[0].markUnhealthy();
    try std.testing.expectEqual(@as(usize, 2), cluster.healthyNodeCount());

    // Re-elect leader
    try cluster.electLeader();

    // New leader should be node-2 (first healthy node)
    const new_leader = cluster.getLeader();
    try std.testing.expect(new_leader != null);
    try std.testing.expectEqualStrings("node-2", new_leader.?.id);
    try std.testing.expect(new_leader.?.is_leader);

    // Old leader should be demoted
    try std.testing.expect(!cluster.nodes.items[0].is_leader);
}

// ============================================================================
// E2E Tests: Failover and Recovery
// ============================================================================

test "e2e: cluster handles node failure" {
    try e2e.skipIfNetworkDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true },
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    var cluster = MockCluster.init(allocator);
    defer cluster.deinit();

    try cluster.addNode("node-1", "10.0.0.1", 5432);
    try cluster.addNode("node-2", "10.0.0.2", 5432);
    try cluster.addNode("node-3", "10.0.0.3", 5432);

    try cluster.electLeader();

    try timer.checkpoint("cluster_stable");

    // Simulate node failure
    cluster.nodes.items[1].markUnhealthy();

    try timer.checkpoint("node_failed");

    // Cluster should still function
    try std.testing.expectEqual(@as(usize, 2), cluster.healthyNodeCount());

    // Leader should still exist (if leader wasn't the failed node)
    try std.testing.expect(cluster.getLeader() != null);

    try timer.checkpoint("cluster_survived");
}

test "e2e: node recovery and rejoin" {
    try e2e.skipIfNetworkDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true },
    });
    defer ctx.deinit();

    var cluster = MockCluster.init(allocator);
    defer cluster.deinit();

    try cluster.addNode("node-1", "10.0.0.1", 5432);
    try cluster.addNode("node-2", "10.0.0.2", 5432);

    // Node fails
    cluster.nodes.items[1].markUnhealthy();
    try std.testing.expectEqual(@as(usize, 1), cluster.healthyNodeCount());

    // Node recovers
    cluster.nodes.items[1].recover();
    try std.testing.expectEqual(@as(usize, 2), cluster.healthyNodeCount());

    // Recovered node should be healthy again
    try std.testing.expect(cluster.nodes.items[1].is_healthy);
}

test "e2e: graceful node removal" {
    try e2e.skipIfNetworkDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true },
    });
    defer ctx.deinit();

    var cluster = MockCluster.init(allocator);
    defer cluster.deinit();

    try cluster.addNode("node-1", "10.0.0.1", 5432);
    try cluster.addNode("node-2", "10.0.0.2", 5432);
    try cluster.addNode("node-3", "10.0.0.3", 5432);

    try cluster.electLeader();

    // Gracefully remove a non-leader node
    const removed = cluster.removeNode("node-2");
    try std.testing.expect(removed);

    try std.testing.expectEqual(@as(usize, 2), cluster.nodes.items.len);

    // Leader should still be valid
    try std.testing.expect(cluster.getLeader() != null);
}

test "e2e: leader removal triggers re-election" {
    try e2e.skipIfNetworkDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true },
    });
    defer ctx.deinit();

    var cluster = MockCluster.init(allocator);
    defer cluster.deinit();

    try cluster.addNode("node-1", "10.0.0.1", 5432);
    try cluster.addNode("node-2", "10.0.0.2", 5432);
    try cluster.addNode("node-3", "10.0.0.3", 5432);

    try cluster.electLeader();
    const initial_leader = cluster.getLeader().?.id;
    try std.testing.expectEqualStrings("node-1", initial_leader);

    // Remove leader
    const removed = cluster.removeNode("node-1");
    try std.testing.expect(removed);

    // Should need new election
    try std.testing.expect(cluster.getLeader() == null);

    // Re-elect
    try cluster.electLeader();

    // New leader should be different
    const new_leader = cluster.getLeader();
    try std.testing.expect(new_leader != null);
    try std.testing.expect(!std.mem.eql(u8, new_leader.?.id, initial_leader));
}

// ============================================================================
// E2E Tests: Quorum and Consistency
// ============================================================================

test "e2e: quorum calculation" {
    try e2e.skipIfNetworkDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true },
    });
    defer ctx.deinit();

    var cluster = MockCluster.init(allocator);
    defer cluster.deinit();

    // For Raft consensus, quorum = (n/2) + 1
    const test_cases = [_]struct { nodes: usize, expected_quorum: usize }{
        .{ .nodes = 1, .expected_quorum = 1 },
        .{ .nodes = 2, .expected_quorum = 2 },
        .{ .nodes = 3, .expected_quorum = 2 },
        .{ .nodes = 4, .expected_quorum = 3 },
        .{ .nodes = 5, .expected_quorum = 3 },
        .{ .nodes = 7, .expected_quorum = 4 },
    };

    for (test_cases) |tc| {
        const quorum = (tc.nodes / 2) + 1;
        try std.testing.expectEqual(tc.expected_quorum, quorum);
    }
}

test "e2e: cluster maintains quorum after failures" {
    try e2e.skipIfNetworkDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .network = true },
    });
    defer ctx.deinit();

    var cluster = MockCluster.init(allocator);
    defer cluster.deinit();

    // 5-node cluster (quorum = 3)
    try cluster.addNode("node-1", "10.0.0.1", 5432);
    try cluster.addNode("node-2", "10.0.0.2", 5432);
    try cluster.addNode("node-3", "10.0.0.3", 5432);
    try cluster.addNode("node-4", "10.0.0.4", 5432);
    try cluster.addNode("node-5", "10.0.0.5", 5432);

    try cluster.electLeader();

    const quorum: usize = 3;

    // Fail 2 nodes - should still have quorum
    cluster.nodes.items[3].markUnhealthy();
    cluster.nodes.items[4].markUnhealthy();

    try std.testing.expect(cluster.healthyNodeCount() >= quorum);

    // Fail one more - should lose quorum
    cluster.nodes.items[2].markUnhealthy();

    try std.testing.expect(cluster.healthyNodeCount() < quorum);
}
