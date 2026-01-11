//! Load Balancing module for distributed request routing.
//!
//! Provides various load balancing strategies including round-robin,
//! weighted routing, least connections, and health-based routing.

const std = @import("std");
const registry = @import("registry.zig");
const time = @import("../../shared/utils/time.zig");

pub const LoadBalancerStrategy = enum {
    round_robin,
    weighted_round_robin,
    least_connections,
    random,
    ip_hash,
    health_weighted,
};

pub const LoadBalancerConfig = struct {
    strategy: LoadBalancerStrategy = .round_robin,
    /// Health check interval in milliseconds
    health_check_interval_ms: u64 = 5000,
    /// Node unhealthy threshold (failed health checks)
    unhealthy_threshold: u32 = 3,
    /// Node recovery threshold (successful health checks)
    recovery_threshold: u32 = 2,
    /// Enable sticky sessions (IP affinity)
    sticky_sessions: bool = false,
    /// Session timeout in milliseconds
    session_timeout_ms: u64 = 3600_000,
    /// Maximum retries on failure
    max_retries: u32 = 3,
};

pub const NodeState = struct {
    id: []const u8,
    address: []const u8,
    weight: u32 = 100,
    current_connections: std.atomic.Value(u32),
    total_requests: std.atomic.Value(u64),
    failed_requests: std.atomic.Value(u64),
    is_healthy: bool,
    consecutive_failures: u32,
    consecutive_successes: u32,
    last_health_check_ms: i64,
    response_time_avg_ms: f64,

    pub fn init(id: []const u8, address: []const u8, weight: u32) NodeState {
        return .{
            .id = id,
            .address = address,
            .weight = weight,
            .current_connections = std.atomic.Value(u32).init(0),
            .total_requests = std.atomic.Value(u64).init(0),
            .failed_requests = std.atomic.Value(u64).init(0),
            .is_healthy = true,
            .consecutive_failures = 0,
            .consecutive_successes = 0,
            .last_health_check_ms = 0,
            .response_time_avg_ms = 0,
        };
    }

    pub fn incrementConnections(self: *NodeState) void {
        _ = self.current_connections.fetchAdd(1, .monotonic);
        _ = self.total_requests.fetchAdd(1, .monotonic);
    }

    pub fn decrementConnections(self: *NodeState) void {
        _ = self.current_connections.fetchSub(1, .monotonic);
    }

    pub fn recordFailure(self: *NodeState) void {
        _ = self.failed_requests.fetchAdd(1, .monotonic);
        self.consecutive_failures += 1;
        self.consecutive_successes = 0;
    }

    pub fn recordSuccess(self: *NodeState) void {
        self.consecutive_successes += 1;
        self.consecutive_failures = 0;
    }

    pub fn getScore(self: *const NodeState) f64 {
        if (!self.is_healthy) return 0;

        const weight_factor = @as(f64, @floatFromInt(self.weight)) / 100.0;
        const conn_factor = 1.0 / (1.0 + @as(f64, @floatFromInt(self.current_connections.load(.monotonic))));
        const latency_factor = if (self.response_time_avg_ms > 0)
            1.0 / (1.0 + self.response_time_avg_ms / 1000.0)
        else
            1.0;

        return weight_factor * conn_factor * latency_factor;
    }
};

pub const LoadBalancerError = error{
    NoHealthyNodes,
    NodeNotFound,
    MaxRetriesExceeded,
    AllNodesFailed,
};

pub const LoadBalancer = struct {
    allocator: std.mem.Allocator,
    config: LoadBalancerConfig,
    nodes: std.ArrayListUnmanaged(NodeState),
    round_robin_index: std.atomic.Value(usize),
    session_map: std.StringArrayHashMapUnmanaged([]const u8),
    prng: std.Random.DefaultPrng,
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator, config: LoadBalancerConfig) LoadBalancer {
        return .{
            .allocator = allocator,
            .config = config,
            .nodes = std.ArrayListUnmanaged(NodeState){},
            .round_robin_index = std.atomic.Value(usize).init(0),
            .session_map = std.StringArrayHashMapUnmanaged([]const u8){},
            .prng = std.Random.DefaultPrng.init(@intCast(time.unixMilliseconds())),
            .mutex = std.Thread.Mutex{},
        };
    }

    pub fn deinit(self: *LoadBalancer) void {
        for (self.nodes.items) |node| {
            self.allocator.free(node.id);
            self.allocator.free(node.address);
        }
        self.nodes.deinit(self.allocator);

        var it = self.session_map.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.session_map.deinit(self.allocator);

        self.* = undefined;
    }

    /// Add a node to the load balancer pool.
    pub fn addNode(self: *LoadBalancer, id: []const u8, address: []const u8, weight: u32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check if node already exists
        for (self.nodes.items) |*node| {
            if (std.mem.eql(u8, node.id, id)) {
                // Update existing node
                if (!std.mem.eql(u8, node.address, address)) {
                    self.allocator.free(node.address);
                    node.address = try self.allocator.dupe(u8, address);
                }
                node.weight = weight;
                return;
            }
        }

        try self.nodes.append(self.allocator, NodeState.init(
            try self.allocator.dupe(u8, id),
            try self.allocator.dupe(u8, address),
            weight,
        ));
    }

    /// Remove a node from the load balancer pool.
    pub fn removeNode(self: *LoadBalancer, id: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.nodes.items, 0..) |node, i| {
            if (std.mem.eql(u8, node.id, id)) {
                self.allocator.free(node.id);
                self.allocator.free(node.address);
                _ = self.nodes.swapRemove(i);
                return true;
            }
        }
        return false;
    }

    /// Get the next node for a request.
    pub fn getNode(self: *LoadBalancer, client_id: ?[]const u8) !*NodeState {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check sticky session first
        if (self.config.sticky_sessions) {
            if (client_id) |cid| {
                if (self.session_map.get(cid)) |node_id| {
                    for (self.nodes.items) |*node| {
                        if (std.mem.eql(u8, node.id, node_id) and node.is_healthy) {
                            return node;
                        }
                    }
                }
            }
        }

        const node = switch (self.config.strategy) {
            .round_robin => try self.selectRoundRobin(),
            .weighted_round_robin => try self.selectWeightedRoundRobin(),
            .least_connections => try self.selectLeastConnections(),
            .random => try self.selectRandom(),
            .ip_hash => try self.selectIpHash(client_id),
            .health_weighted => try self.selectHealthWeighted(),
        };

        // Store session mapping
        if (self.config.sticky_sessions) {
            if (client_id) |cid| {
                const cid_copy = try self.allocator.dupe(u8, cid);
                errdefer self.allocator.free(cid_copy);
                const node_id_copy = try self.allocator.dupe(u8, node.id);
                errdefer self.allocator.free(node_id_copy);

                if (self.session_map.fetchRemove(cid_copy)) |old| {
                    self.allocator.free(old.key);
                    self.allocator.free(old.value);
                }
                try self.session_map.put(self.allocator, cid_copy, node_id_copy);
            }
        }

        return node;
    }

    /// Record a successful request to a node.
    pub fn recordSuccess(self: *LoadBalancer, node_id: []const u8, response_time_ms: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.nodes.items) |*node| {
            if (std.mem.eql(u8, node.id, node_id)) {
                node.recordSuccess();
                node.decrementConnections();

                // Update running average of response time
                const alpha = 0.1; // Smoothing factor
                const new_time = @as(f64, @floatFromInt(response_time_ms));
                node.response_time_avg_ms = alpha * new_time + (1.0 - alpha) * node.response_time_avg_ms;

                // Check for recovery
                if (!node.is_healthy and node.consecutive_successes >= self.config.recovery_threshold) {
                    node.is_healthy = true;
                }
                return;
            }
        }
    }

    /// Record a failed request to a node.
    pub fn recordFailure(self: *LoadBalancer, node_id: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.nodes.items) |*node| {
            if (std.mem.eql(u8, node.id, node_id)) {
                node.recordFailure();
                node.decrementConnections();

                // Check for unhealthy threshold
                if (node.is_healthy and node.consecutive_failures >= self.config.unhealthy_threshold) {
                    node.is_healthy = false;
                }
                return;
            }
        }
    }

    /// Update node health status.
    pub fn setNodeHealth(self: *LoadBalancer, node_id: []const u8, healthy: bool) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.nodes.items) |*node| {
            if (std.mem.eql(u8, node.id, node_id)) {
                node.is_healthy = healthy;
                node.last_health_check_ms = time.nowMilliseconds();
                if (healthy) {
                    node.consecutive_failures = 0;
                } else {
                    node.consecutive_successes = 0;
                }
                return;
            }
        }
    }

    /// Get statistics for all nodes.
    /// Caller must free the returned slice with `allocator.free(slice)`.
    pub fn getStats(self: *LoadBalancer, allocator: std.mem.Allocator) ![]NodeStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.nodes.items.len == 0) return &.{};
        const stats = try allocator.alloc(NodeStats, self.nodes.items.len);

        for (self.nodes.items, 0..) |*node, i| {
            stats[i] = .{
                .id = node.id,
                .address = node.address,
                .weight = node.weight,
                .current_connections = node.current_connections.load(.monotonic),
                .total_requests = node.total_requests.load(.monotonic),
                .failed_requests = node.failed_requests.load(.monotonic),
                .is_healthy = node.is_healthy,
                .response_time_avg_ms = node.response_time_avg_ms,
            };
        }

        return stats;
    }

    /// Sync nodes from a registry.
    pub fn syncFromRegistry(self: *LoadBalancer, node_registry: *registry.NodeRegistry) !void {
        const nodes = node_registry.list();

        for (nodes) |node| {
            const weight: u32 = switch (node.status) {
                .healthy => 100,
                .degraded => 50,
                .offline => 0,
            };
            try self.addNode(node.id, node.address, weight);

            if (node.status == .offline) {
                self.setNodeHealth(node.id, false);
            }
        }
    }

    // Selection strategies

    fn selectRoundRobin(self: *LoadBalancer) !*NodeState {
        const healthy_count = self.countHealthyNodes();
        if (healthy_count == 0) return LoadBalancerError.NoHealthyNodes;

        var attempts: usize = 0;
        while (attempts < self.nodes.items.len) : (attempts += 1) {
            const index = self.round_robin_index.fetchAdd(1, .monotonic) % self.nodes.items.len;
            const node = &self.nodes.items[index];
            if (node.is_healthy) {
                node.incrementConnections();
                return node;
            }
        }

        return LoadBalancerError.NoHealthyNodes;
    }

    fn selectWeightedRoundRobin(self: *LoadBalancer) !*NodeState {
        var total_weight: u32 = 0;
        for (self.nodes.items) |*node| {
            if (node.is_healthy) {
                total_weight += node.weight;
            }
        }

        if (total_weight == 0) return LoadBalancerError.NoHealthyNodes;

        const index = self.round_robin_index.fetchAdd(1, .monotonic);
        var target = @as(u32, @intCast(index % total_weight));

        for (self.nodes.items) |*node| {
            if (!node.is_healthy) continue;

            if (target < node.weight) {
                node.incrementConnections();
                return node;
            }
            target -= node.weight;
        }

        return LoadBalancerError.NoHealthyNodes;
    }

    fn selectLeastConnections(self: *LoadBalancer) !*NodeState {
        var best: ?*NodeState = null;
        var best_conns: u32 = std.math.maxInt(u32);

        for (self.nodes.items) |*node| {
            if (!node.is_healthy) continue;

            const conns = node.current_connections.load(.monotonic);
            if (conns < best_conns) {
                best_conns = conns;
                best = node;
            }
        }

        if (best) |node| {
            node.incrementConnections();
            return node;
        }

        return LoadBalancerError.NoHealthyNodes;
    }

    fn selectRandom(self: *LoadBalancer) !*NodeState {
        const healthy_count = self.countHealthyNodes();
        if (healthy_count == 0) return LoadBalancerError.NoHealthyNodes;

        const target = self.prng.random().intRangeLessThan(usize, 0, healthy_count);
        var count: usize = 0;

        for (self.nodes.items) |*node| {
            if (!node.is_healthy) continue;

            if (count == target) {
                node.incrementConnections();
                return node;
            }
            count += 1;
        }

        return LoadBalancerError.NoHealthyNodes;
    }

    fn selectIpHash(self: *LoadBalancer, client_id: ?[]const u8) !*NodeState {
        const healthy_count = self.countHealthyNodes();
        if (healthy_count == 0) return LoadBalancerError.NoHealthyNodes;

        const hash: u64 = if (client_id) |cid|
            std.hash.Wyhash.hash(0, cid)
        else
            @intCast(time.unixMilliseconds());

        const target = hash % healthy_count;
        var count: usize = 0;

        for (self.nodes.items) |*node| {
            if (!node.is_healthy) continue;

            if (count == target) {
                node.incrementConnections();
                return node;
            }
            count += 1;
        }

        return LoadBalancerError.NoHealthyNodes;
    }

    fn selectHealthWeighted(self: *LoadBalancer) !*NodeState {
        var best: ?*NodeState = null;
        var best_score: f64 = 0;

        for (self.nodes.items) |*node| {
            if (!node.is_healthy) continue;

            const score = node.getScore();
            if (score > best_score) {
                best_score = score;
                best = node;
            }
        }

        if (best) |node| {
            node.incrementConnections();
            return node;
        }

        return LoadBalancerError.NoHealthyNodes;
    }

    fn countHealthyNodes(self: *LoadBalancer) usize {
        var count: usize = 0;
        for (self.nodes.items) |node| {
            if (node.is_healthy) count += 1;
        }
        return count;
    }
};

pub const NodeStats = struct {
    id: []const u8,
    address: []const u8,
    weight: u32,
    current_connections: u32,
    total_requests: u64,
    failed_requests: u64,
    is_healthy: bool,
    response_time_avg_ms: f64,
};

test "load balancer round robin" {
    const allocator = std.testing.allocator;
    var lb = LoadBalancer.init(allocator, .{ .strategy = .round_robin });
    defer lb.deinit();

    try lb.addNode("node-1", "192.168.1.1:9000", 100);
    try lb.addNode("node-2", "192.168.1.2:9000", 100);
    try lb.addNode("node-3", "192.168.1.3:9000", 100);

    const node1 = try lb.getNode(null);
    const node2 = try lb.getNode(null);
    const node3 = try lb.getNode(null);
    const node4 = try lb.getNode(null);

    // Should cycle through nodes
    try std.testing.expect(!std.mem.eql(u8, node1.id, node2.id) or !std.mem.eql(u8, node2.id, node3.id));
    _ = node4;
}

test "load balancer least connections" {
    const allocator = std.testing.allocator;
    var lb = LoadBalancer.init(allocator, .{ .strategy = .least_connections });
    defer lb.deinit();

    try lb.addNode("node-1", "192.168.1.1:9000", 100);
    try lb.addNode("node-2", "192.168.1.2:9000", 100);

    // First request goes to either node
    const node1 = try lb.getNode(null);

    // Second request should go to the other node (0 connections)
    const node2 = try lb.getNode(null);

    try std.testing.expect(!std.mem.eql(u8, node1.id, node2.id));
}

test "load balancer health checking" {
    const allocator = std.testing.allocator;
    var lb = LoadBalancer.init(allocator, .{
        .strategy = .round_robin,
        .unhealthy_threshold = 2,
    });
    defer lb.deinit();

    try lb.addNode("node-1", "192.168.1.1:9000", 100);
    try lb.addNode("node-2", "192.168.1.2:9000", 100);

    // Mark node-1 as failed twice
    lb.recordFailure("node-1");
    lb.recordFailure("node-1");

    // Node-1 should now be unhealthy
    var node1_healthy = false;
    for (lb.nodes.items) |node| {
        if (std.mem.eql(u8, node.id, "node-1")) {
            node1_healthy = node.is_healthy;
            break;
        }
    }
    try std.testing.expect(!node1_healthy);
}

test "load balancer weighted" {
    const allocator = std.testing.allocator;
    var lb = LoadBalancer.init(allocator, .{ .strategy = .weighted_round_robin });
    defer lb.deinit();

    try lb.addNode("node-1", "192.168.1.1:9000", 80);
    try lb.addNode("node-2", "192.168.1.2:9000", 20);

    var node1_count: u32 = 0;
    var node2_count: u32 = 0;

    // Make 100 requests
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const node = try lb.getNode(null);
        if (std.mem.eql(u8, node.id, "node-1")) {
            node1_count += 1;
        } else {
            node2_count += 1;
        }
        node.decrementConnections();
    }

    // Node-1 should get roughly 80% of requests
    try std.testing.expect(node1_count > node2_count);
}

test "node state scoring" {
    var node = NodeState.init("test", "localhost:9000", 100);

    // Healthy node with no connections
    const score1 = node.getScore();
    try std.testing.expect(score1 > 0);

    // Add some connections
    node.incrementConnections();
    node.incrementConnections();
    const score2 = node.getScore();
    try std.testing.expect(score2 < score1);

    // Unhealthy node should have 0 score
    node.is_healthy = false;
    const score3 = node.getScore();
    try std.testing.expectEqual(@as(f64, 0), score3);
}

