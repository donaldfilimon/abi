//! Comprehensive Network Module Tests
//!
//! Tests for the network module including:
//! - Raft consensus protocol
//! - Load balancer strategies
//! - Circuit breaker patterns
//! - Service discovery
//! - Integration tests for components working together

const std = @import("std");
const build_options = @import("build_options");
const network = @import("abi").network;

// =============================================================================
// Raft Consensus Tests
// =============================================================================

test "raft cluster creation and peer connectivity" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const node_ids = &[_][]const u8{ "node-1", "node-2", "node-3" };

    var cluster = try network.createRaftCluster(allocator, node_ids, .{
        .election_timeout_min_ms = 10,
        .election_timeout_max_ms = 20,
    });
    defer {
        for (cluster.items) |node| {
            node.deinit();
            allocator.destroy(node);
        }
        cluster.deinit(allocator);
    }

    // Each node should have 2 peers
    for (cluster.items) |node| {
        try std.testing.expectEqual(@as(usize, 2), node.peers.count());
    }
}

test "raft single node becomes leader immediately" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var node = try network.RaftNode.init(allocator, "solo-node", .{
        .election_timeout_min_ms = 5,
        .election_timeout_max_ms = 10,
    });
    defer node.deinit();

    // Trigger election timeout
    try node.tick(50);

    // Single node should become leader
    try std.testing.expectEqual(network.RaftState.leader, node.state);
    try std.testing.expect(node.isLeader());
}

test "raft term progression during elections" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var node = try network.RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 5,
        .election_timeout_max_ms = 10,
    });
    defer node.deinit();

    try std.testing.expectEqual(@as(u64, 0), node.current_term);

    // Trigger election
    try node.tick(50);

    // Term should have incremented
    try std.testing.expect(node.current_term >= 1);
}

test "raft vote rejection for outdated terms" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var node = try network.RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // Manually advance the node's term
    node.current_term = 5;

    // Request vote with older term should be rejected
    const request = network.RequestVoteRequest{
        .term = 2,
        .candidate_id = "node-2",
        .last_log_index = 0,
        .last_log_term = 0,
    };

    const response = try node.handleRequestVote(request);
    try std.testing.expect(!response.vote_granted);
    try std.testing.expectEqual(@as(u64, 5), response.term);
}

test "raft vote granted for higher term" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var node = try network.RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    const request = network.RequestVoteRequest{
        .term = 1,
        .candidate_id = "node-2",
        .last_log_index = 0,
        .last_log_term = 0,
    };

    const response = try node.handleRequestVote(request);
    try std.testing.expect(response.vote_granted);
    try std.testing.expectEqualStrings("node-2", node.voted_for.?);
}

test "raft append entries updates leader" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var follower = try network.RaftNode.init(allocator, "follower", .{});
    defer follower.deinit();

    const request = network.AppendEntriesRequest{
        .term = 1,
        .leader_id = "leader-node",
        .prev_log_index = 0,
        .prev_log_term = 0,
        .entries = &.{},
        .leader_commit = 0,
    };

    const response = try follower.handleAppendEntries(request);

    try std.testing.expect(response.success);
    try std.testing.expectEqual(network.RaftState.follower, follower.state);
    try std.testing.expectEqualStrings("leader-node", follower.leader_id.?);
}

test "raft log replication" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Create a leader
    var leader = try network.RaftNode.init(allocator, "leader", .{
        .election_timeout_min_ms = 5,
        .election_timeout_max_ms = 10,
    });
    defer leader.deinit();

    // Make it leader (single node)
    try leader.tick(50);
    try std.testing.expect(leader.isLeader());

    // Append commands
    const idx1 = try leader.appendCommand("cmd-1");
    const idx2 = try leader.appendCommand("cmd-2");

    // Log should have entries (plus the no-op from becoming leader)
    try std.testing.expect(leader.log.items.len >= 2);
    try std.testing.expect(idx2 > idx1);
}

test "raft non-leader cannot append commands" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var follower = try network.RaftNode.init(allocator, "follower", .{});
    defer follower.deinit();

    // Follower should not be able to append commands
    try std.testing.expectError(network.RaftError.NotLeader, follower.appendCommand("test"));
}

test "raft build append entries request" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var node = try network.RaftNode.init(allocator, "leader", .{
        .election_timeout_min_ms = 5,
        .election_timeout_max_ms = 10,
    });
    defer node.deinit();

    try node.addPeer("follower-1");

    // Become leader
    try node.tick(50);

    const request = node.buildAppendEntriesRequest("follower-1");
    try std.testing.expect(request != null);
    try std.testing.expectEqualStrings("leader", request.?.leader_id);
}

// =============================================================================
// Load Balancer Tests
// =============================================================================

test "load balancer empty pool returns error" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var lb = network.LoadBalancer.init(allocator, .{ .strategy = .round_robin });
    defer lb.deinit();

    try std.testing.expectError(network.LoadBalancerError.NoHealthyNodes, lb.getNode(null));
}

test "load balancer node addition and removal" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var lb = network.LoadBalancer.init(allocator, .{ .strategy = .round_robin });
    defer lb.deinit();

    try lb.addNode("node-1", "192.168.1.1:9000", 100);
    try lb.addNode("node-2", "192.168.1.2:9000", 100);

    try std.testing.expectEqual(@as(usize, 2), lb.nodes.items.len);

    const removed = lb.removeNode("node-1");
    try std.testing.expect(removed);
    try std.testing.expectEqual(@as(usize, 1), lb.nodes.items.len);
}

test "load balancer node update" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var lb = network.LoadBalancer.init(allocator, .{ .strategy = .round_robin });
    defer lb.deinit();

    try lb.addNode("node-1", "192.168.1.1:9000", 100);

    // Update same node with new address
    try lb.addNode("node-1", "192.168.1.1:9001", 150);

    // Should still be only one node
    try std.testing.expectEqual(@as(usize, 1), lb.nodes.items.len);

    // Weight should be updated
    try std.testing.expectEqual(@as(u32, 150), lb.nodes.items[0].weight);
}

test "load balancer random strategy" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var lb = network.LoadBalancer.init(allocator, .{ .strategy = .random });
    defer lb.deinit();

    try lb.addNode("node-1", "192.168.1.1:9000", 100);
    try lb.addNode("node-2", "192.168.1.2:9000", 100);

    // Should return a valid node
    const node = try lb.getNode(null);
    try std.testing.expect(node.is_healthy);
}

test "load balancer health weighted strategy" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var lb = network.LoadBalancer.init(allocator, .{ .strategy = .health_weighted });
    defer lb.deinit();

    try lb.addNode("node-1", "192.168.1.1:9000", 100);
    try lb.addNode("node-2", "192.168.1.2:9000", 50);

    // Higher weight node should have higher score
    const node = try lb.getNode(null);
    try std.testing.expect(node.is_healthy);
}

test "load balancer sticky sessions" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var lb = network.LoadBalancer.init(allocator, .{
        .strategy = .round_robin,
        .sticky_sessions = true,
    });
    defer lb.deinit();

    try lb.addNode("node-1", "192.168.1.1:9000", 100);
    try lb.addNode("node-2", "192.168.1.2:9000", 100);

    // Same client should get same node
    const client_id = "client-123";
    const node1 = try lb.getNode(client_id);
    node1.decrementConnections();

    const node2 = try lb.getNode(client_id);
    node2.decrementConnections();

    try std.testing.expectEqualStrings(node1.id, node2.id);
}

test "load balancer get stats" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var lb = network.LoadBalancer.init(allocator, .{ .strategy = .round_robin });
    defer lb.deinit();

    try lb.addNode("node-1", "192.168.1.1:9000", 100);

    const stats = try lb.getStats(allocator);
    defer allocator.free(stats);

    try std.testing.expectEqual(@as(usize, 1), stats.len);
    try std.testing.expectEqualStrings("node-1", stats[0].id);
}

test "load balancer response time tracking" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var lb = network.LoadBalancer.init(allocator, .{ .strategy = .round_robin });
    defer lb.deinit();

    try lb.addNode("node-1", "192.168.1.1:9000", 100);

    const node = try lb.getNode(null);
    lb.recordSuccess(node.id, 50);

    // Response time should be tracked
    try std.testing.expect(node.response_time_avg_ms > 0);
}

// =============================================================================
// Circuit Breaker Tests
// =============================================================================

test "circuit breaker force open" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var breaker = network.CircuitBreaker.init(allocator, .{
        .failure_threshold = 10,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    try std.testing.expectEqual(network.CircuitState.closed, breaker.getState());

    breaker.forceOpen();
    try std.testing.expectEqual(network.CircuitState.open, breaker.getState());
}

test "circuit breaker force closed" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var breaker = network.CircuitBreaker.init(allocator, .{
        .failure_threshold = 2,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    // Open the circuit
    breaker.recordFailure();
    breaker.recordFailure();
    try std.testing.expectEqual(network.CircuitState.open, breaker.getState());

    breaker.forceClosed();
    try std.testing.expectEqual(network.CircuitState.closed, breaker.getState());
}

test "circuit breaker consecutive tracking" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var breaker = network.CircuitBreaker.init(allocator, .{
        .failure_threshold = 10,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    breaker.recordSuccess();
    breaker.recordSuccess();
    breaker.recordSuccess();

    var stats = breaker.getStats();
    try std.testing.expectEqual(@as(u32, 3), stats.consecutive_successes);
    try std.testing.expectEqual(@as(u32, 0), stats.consecutive_failures);

    breaker.recordFailure();
    stats = breaker.getStats();
    try std.testing.expectEqual(@as(u32, 0), stats.consecutive_successes);
    try std.testing.expectEqual(@as(u32, 1), stats.consecutive_failures);
}

test "circuit breaker half open success recovery" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var breaker = network.CircuitBreaker.init(allocator, .{
        .failure_threshold = 2,
        .success_threshold = 2,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    // Open the circuit
    breaker.recordFailure();
    breaker.recordFailure();
    try std.testing.expectEqual(network.CircuitState.open, breaker.getState());

    // Manually transition to half-open for testing
    breaker.mutex.lock();
    breaker.state = .half_open;
    breaker.success_count = 0;
    breaker.mutex.unlock();

    // Record enough successes to close
    breaker.recordSuccess();
    breaker.recordSuccess();

    try std.testing.expectEqual(network.CircuitState.closed, breaker.getState());
}

test "circuit breaker half open failure returns to open" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var breaker = network.CircuitBreaker.init(allocator, .{
        .failure_threshold = 2,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    // Manually set to half-open
    breaker.mutex.lock();
    breaker.state = .half_open;
    breaker.mutex.unlock();

    // Any failure should return to open
    breaker.recordFailure();
    try std.testing.expectEqual(network.CircuitState.open, breaker.getState());
}

test "circuit breaker registry list breakers" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var registry_obj = network.CircuitRegistry.init(allocator);
    defer registry_obj.deinit();

    try registry_obj.register("service-a", .{});
    try registry_obj.register("service-b", .{});

    const names = try registry_obj.listBreakers();
    defer allocator.free(names);

    try std.testing.expectEqual(@as(usize, 2), names.len);
}

test "circuit breaker registry reset all" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var registry_obj = network.CircuitRegistry.init(allocator);
    defer registry_obj.deinit();

    const breaker = try registry_obj.getOrCreate("service-a");

    // Open the circuit
    breaker.forceOpen();
    try std.testing.expectEqual(network.CircuitState.open, breaker.getState());

    registry_obj.resetAll();
    try std.testing.expectEqual(network.CircuitState.closed, breaker.getState());
}

// =============================================================================
// Service Discovery Tests
// =============================================================================

test "service discovery id generation" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    const id = try network.ServiceDiscovery.discovery_types.generateServiceId(allocator, "my-service");
    defer allocator.free(id);

    try std.testing.expect(std.mem.startsWith(u8, id, "my-service-"));
    try std.testing.expect(id.len > "my-service-".len);
}

test "service discovery registration lifecycle" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var discovery = try network.ServiceDiscovery.init(allocator, .{
        .backend = .static,
        .service_name = "test-service",
        .service_address = "127.0.0.1",
        .service_port = 8080,
    });
    defer discovery.deinit();

    try std.testing.expect(!discovery.registered);

    try discovery.register();
    try std.testing.expect(discovery.registered);

    try discovery.deregister();
    try std.testing.expect(!discovery.registered);
}

test "service discovery base64 encoding" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    const original = "hello world";
    const encoded = try network.base64Encode(allocator, original);
    defer allocator.free(encoded);

    const decoded = try network.base64Decode(allocator, encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(original, decoded);
}

// =============================================================================
// Integration Tests
// =============================================================================

test "load balancer with discovery" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Create a node registry
    var reg = network.NodeRegistry.init(allocator);
    defer reg.deinit();

    try reg.register("node-1", "192.168.1.10:9000");
    try reg.register("node-2", "192.168.1.11:9001");

    // Sync to load balancer
    var lb = network.LoadBalancer.init(allocator, .{ .strategy = .round_robin });
    defer lb.deinit();

    try lb.syncFromRegistry(&reg);

    try std.testing.expectEqual(@as(usize, 2), lb.nodes.items.len);
}

test "network module initialization" {
    if (!network.isEnabled()) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    try network.initWithConfig(allocator, .{ .cluster_id = "test-cluster" });
    defer network.deinit();

    try std.testing.expect(network.isInitialized());

    const config = network.defaultConfig() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("test-cluster", config.cluster_id);
}

test "network context lifecycle" {
    if (!network.isEnabled()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const config_module = @import("../config/mod.zig");

    const ctx = try network.Context.init(allocator, config_module.NetworkConfig{});
    defer ctx.deinit();

    try std.testing.expectEqual(network.Context.State.disconnected, ctx.getState());

    try ctx.connect();
    try std.testing.expectEqual(network.Context.State.connected, ctx.getState());

    ctx.disconnect();
    try std.testing.expectEqual(network.Context.State.disconnected, ctx.getState());
}

// =============================================================================
// Raft Multi-Node Simulation Tests
// =============================================================================

test "raft simulated election with three nodes" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Create three nodes
    var node1 = try network.RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 50,
        .election_timeout_max_ms = 100,
    });
    defer node1.deinit();

    var node2 = try network.RaftNode.init(allocator, "node-2", .{
        .election_timeout_min_ms = 150,
        .election_timeout_max_ms = 200,
    });
    defer node2.deinit();

    var node3 = try network.RaftNode.init(allocator, "node-3", .{
        .election_timeout_min_ms = 250,
        .election_timeout_max_ms = 300,
    });
    defer node3.deinit();

    // Add peers
    try node1.addPeer("node-2");
    try node1.addPeer("node-3");
    try node2.addPeer("node-1");
    try node2.addPeer("node-3");
    try node3.addPeer("node-1");
    try node3.addPeer("node-2");

    // Node1 has shortest timeout, should start election first
    try node1.tick(100);

    // Node1 should be candidate
    try std.testing.expect(node1.state == .candidate or node1.state == .leader);
    try std.testing.expect(node1.current_term >= 1);

    // Build vote request
    const request = node1.buildRequestVoteRequest();

    // Node2 grants vote
    const response2 = try node2.handleRequestVote(request);
    try node1.handleRequestVoteResponse(response2);

    // Node3 grants vote
    const response3 = try node3.handleRequestVote(request);
    try node1.handleRequestVoteResponse(response3);

    // Node1 should now be leader (has majority)
    try std.testing.expectEqual(network.RaftState.leader, node1.state);
}
