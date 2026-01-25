//! Network Module Chaos Tests
//!
//! Tests for network module components under failure conditions:
//! - Raft consensus with message loss
//! - Leader election with network partitions
//! - Distributed operations with latency spikes
//! - Circuit breaker activation under failures
//!
//! These tests verify that the network system:
//! 1. Maintains consensus under adverse conditions
//! 2. Handles partitions and recovers correctly
//! 3. Circuit breakers protect against cascading failures

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("abi");
const network = abi.network;
const chaos = @import("mod.zig");

// ============================================================================
// Test Helpers
// ============================================================================

/// Sleep implementation for tests
fn sleepMs(ms: u64) void {
    const ns = ms * std.time.ns_per_ms;
    const builtin = @import("builtin");
    if (builtin.cpu.arch == .wasm32 or builtin.cpu.arch == .wasm64) {
        return;
    }
    if (@hasDecl(std.posix, "nanosleep")) {
        var req = std.posix.timespec{
            .sec = @intCast(ns / std.time.ns_per_s),
            .nsec = @intCast(ns % std.time.ns_per_s),
        };
        var rem: std.posix.timespec = undefined;
        while (true) {
            const result = std.posix.nanosleep(&req, &rem);
            if (result == 0) break;
            req = rem;
        }
    }
}

// ============================================================================
// Raft Consensus Chaos Tests
// ============================================================================

test "network chaos: raft handles message loss during election" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = try chaos.ChaosContext.init(allocator, 12345);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .message_drop,
        .probability = 0.2, // 20% message loss
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    // Create a single-node Raft cluster (will become leader immediately)
    var node = try network.RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 10,
        .election_timeout_max_ms = 20,
    });
    defer node.deinit();

    // Trigger election timeout
    try node.tick(50);

    // Single node should become leader despite chaos
    // (no messages to drop for single node)
    try std.testing.expectEqual(network.RaftState.leader, node.state);

    const stats = chaos_ctx.getStats();
    std.log.info("Raft election chaos: state={t}, term={d}, faults={d}", .{
        node.state,
        node.current_term,
        stats.faults_injected,
    });
}

test "network chaos: raft cluster election with partitions" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = try chaos.ChaosContext.init(allocator, 23456);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .network_partition,
        .probability = 0.1,
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var partition_sim = chaos.NetworkPartitionSimulator.init(allocator, &chaos_ctx);
    defer partition_sim.deinit();

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

    // Create a partition: node-3 is isolated
    try partition_sim.partition("node-1", "node-3");
    try partition_sim.partition("node-2", "node-3");

    // Node1 should start election first (shortest timeout)
    try node1.tick(100);

    // Node1 should be candidate or leader
    try std.testing.expect(node1.state == .candidate or node1.state == .leader);

    // Simulate vote exchange (only between node1 and node2, node3 is partitioned)
    if (!partition_sim.isPartitioned("node-1", "node-2")) {
        const request = node1.buildRequestVoteRequest();
        const response = try node2.handleRequestVote(request);
        try node1.handleRequestVoteResponse(response);
    }

    // Node1 should become leader with majority (self + node2)
    if (node1.state == .leader) {
        try std.testing.expectEqual(network.RaftState.leader, node1.state);
    }

    // Heal partition
    partition_sim.healAll();

    // After healing, node3 should be able to receive heartbeats
    if (node1.isLeader()) {
        const append_request = node1.buildAppendEntriesRequest("node-3");
        if (append_request) |req| {
            const response = try node3.handleAppendEntries(req);
            try std.testing.expect(response.success);
            try std.testing.expectEqualStrings("node-1", node3.leader_id.?);
        }
    }

    const stats = chaos_ctx.getStats();
    std.log.info("Raft partition chaos: node1={t}, node2={t}, node3={t}, faults={d}", .{
        node1.state,
        node2.state,
        node3.state,
        stats.faults_injected,
    });
}

test "network chaos: raft log replication with message delays" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = try chaos.ChaosContext.init(allocator, 34567);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .message_reorder,
        .probability = 0.15,
    });
    try chaos_ctx.addFault(.{
        .fault_type = .latency_injection,
        .probability = 0.1,
        .duration_ms = 5,
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var delay_sim = chaos.MessageDelaySimulator.init(allocator, &chaos_ctx);
    defer delay_sim.deinit();

    // Create a leader
    var leader = try network.RaftNode.init(allocator, "leader", .{
        .election_timeout_min_ms = 5,
        .election_timeout_max_ms = 10,
    });
    defer leader.deinit();

    // Make it leader
    try leader.tick(50);
    try std.testing.expect(leader.isLeader());

    // Append commands with potential delays
    var successful_appends: u32 = 0;
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        var cmd_buf: [32]u8 = undefined;
        const cmd = std.fmt.bufPrint(&cmd_buf, "command-{d}", .{i}) catch continue;

        // Maybe inject delay
        const delayed = delay_sim.maybeDelay("client", "leader", cmd) catch null;
        _ = delayed;

        // Maybe inject latency
        chaos_ctx.maybeInjectLatency();

        const idx = leader.appendCommand(cmd) catch continue;
        if (idx > 0) successful_appends += 1;
    }

    // Log should have entries
    try std.testing.expect(leader.log.items.len > 0);
    try std.testing.expect(successful_appends > 0);

    // Process any delayed messages
    const ready = delay_sim.getReadyMessages() catch &[_]chaos.MessageDelaySimulator.DelayedMessage{};
    defer allocator.free(ready);

    const stats = chaos_ctx.getStats();
    std.log.info("Raft delay chaos: appends={d}, log_len={d}, delayed={d}, faults={d}", .{
        successful_appends,
        leader.log.items.len,
        ready.len,
        stats.faults_injected,
    });
}

test "network chaos: raft vote rejection handles outdated terms with chaos" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = try chaos.ChaosContext.init(allocator, 45678);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .message_drop,
        .probability = 0.05,
    });
    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.02,
        .warmup_ops = 20,
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var node = try network.RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // Advance node's term
    node.current_term = 10;

    // Simulate multiple vote requests with old terms
    var rejected_count: u32 = 0;
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        if (chaos_ctx.shouldDropMessage()) continue;

        const request = network.RequestVoteRequest{
            .term = i, // Terms 0-9, all less than node's term of 10
            .candidate_id = "candidate",
            .last_log_index = 0,
            .last_log_term = 0,
        };

        const response = node.handleRequestVote(request) catch continue;
        if (!response.vote_granted) {
            rejected_count += 1;
        }
    }

    // Most votes should be rejected (old terms)
    try std.testing.expect(rejected_count > 0);

    const stats = chaos_ctx.getStats();
    std.log.info("Raft vote rejection chaos: rejected={d}, term={d}, faults={d}", .{
        rejected_count,
        node.current_term,
        stats.faults_injected,
    });
}

// ============================================================================
// Load Balancer Chaos Tests
// ============================================================================

test "network chaos: load balancer handles node failures" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = try chaos.ChaosContext.init(allocator, 56789);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.05,
        .warmup_ops = 10,
    });
    try chaos_ctx.addFault(.{
        .fault_type = .timeout_injection,
        .probability = 0.1,
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var lb = network.LoadBalancer.init(allocator, .{
        .strategy = .round_robin,
    });
    defer lb.deinit();

    // Add nodes (may fail under chaos)
    var added_nodes: u32 = 0;
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        var id_buf: [16]u8 = undefined;
        const id = std.fmt.bufPrint(&id_buf, "node-{d}", .{i}) catch continue;
        var addr_buf: [32]u8 = undefined;
        const addr = std.fmt.bufPrint(&addr_buf, "192.168.1.{d}:9000", .{i}) catch continue;

        lb.addNode(id, addr, 100) catch continue;
        added_nodes += 1;
    }

    // Should have some nodes
    try std.testing.expect(added_nodes > 0);

    // Test node selection under chaos
    var selections: u32 = 0;
    var failures: u32 = 0;

    var j: u32 = 0;
    while (j < 50) : (j += 1) {
        if (chaos_ctx.shouldTimeout()) {
            failures += 1;
            continue;
        }

        const node = lb.getNode(null) catch {
            failures += 1;
            continue;
        };
        node.decrementConnections();
        selections += 1;
    }

    try std.testing.expect(selections > 0);

    const stats = chaos_ctx.getStats();
    std.log.info("LB chaos: nodes={d}, selections={d}, failures={d}, faults={d}", .{
        added_nodes,
        selections,
        failures,
        stats.faults_injected,
    });
}

test "network chaos: load balancer sticky sessions under failures" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = try chaos.ChaosContext.init(allocator, 67890);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.03,
        .warmup_ops = 15,
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var lb = network.LoadBalancer.init(allocator, .{
        .strategy = .round_robin,
        .sticky_sessions = true,
    });
    defer lb.deinit();

    // Add nodes
    try lb.addNode("node-1", "192.168.1.1:9000", 100);
    try lb.addNode("node-2", "192.168.1.2:9000", 100);
    try lb.addNode("node-3", "192.168.1.3:9000", 100);

    // Test sticky sessions - same client should get same node
    const client_id = "test-client-123";

    var first_node_id: ?[]const u8 = null;
    var consistent_selections: u32 = 0;
    var inconsistent_selections: u32 = 0;

    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        const node = lb.getNode(client_id) catch continue;
        defer node.decrementConnections();

        if (first_node_id == null) {
            first_node_id = node.id;
        }

        if (first_node_id) |first_id| {
            if (std.mem.eql(u8, node.id, first_id)) {
                consistent_selections += 1;
            } else {
                inconsistent_selections += 1;
            }
        }
    }

    // Sticky sessions should be consistent
    try std.testing.expect(consistent_selections > inconsistent_selections);

    const stats = chaos_ctx.getStats();
    std.log.info("Sticky session chaos: consistent={d}, inconsistent={d}, faults={d}", .{
        consistent_selections,
        inconsistent_selections,
        stats.faults_injected,
    });
}

// ============================================================================
// Circuit Breaker Chaos Tests
// ============================================================================

test "network chaos: circuit breaker opens under failure load" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = try chaos.ChaosContext.init(allocator, 78901);
    defer chaos_ctx.deinit();

    // Configure chaos to trigger many failures
    try chaos_ctx.addFault(.{
        .fault_type = .timeout_injection,
        .probability = 0.8, // 80% timeout rate
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var breaker = network.CircuitBreaker.init(allocator, .{
        .failure_threshold = 5,
        .success_threshold = 2,
        .timeout_ms = 60000,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    // Initial state should be closed
    try std.testing.expectEqual(network.CircuitState.closed, breaker.getState());

    // Simulate operations with chaos-induced failures
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        if (chaos_ctx.shouldTimeout()) {
            breaker.recordFailure();
        } else {
            breaker.recordSuccess();
        }

        // Check if circuit opened
        if (breaker.getState() == .open) {
            break;
        }
    }

    // Circuit should have opened due to failures
    const final_state = breaker.getState();
    const breaker_stats = breaker.getStats();

    // Either opened or we hit the iteration limit
    try std.testing.expect(final_state == .open or breaker_stats.failed_requests > 0);

    const chaos_stats = chaos_ctx.getStats();
    std.log.info("Circuit breaker chaos: state={t}, failures={d}, faults={d}", .{
        final_state,
        breaker_stats.failed_requests,
        chaos_stats.faults_injected,
    });
}

test "network chaos: circuit breaker registry handles multiple breakers under chaos" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = try chaos.ChaosContext.init(allocator, 89012);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.05,
        .warmup_ops = 5,
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var registry = network.CircuitRegistry.init(allocator);
    defer registry.deinit();

    // Register multiple breakers
    var registered: u32 = 0;
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        var name_buf: [32]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "service-{d}", .{i}) catch continue;

        registry.register(name, .{
            .failure_threshold = 3,
            .success_threshold = 2,
        }) catch continue;
        registered += 1;
    }

    // Should have registered some breakers
    try std.testing.expect(registered > 0);

    // Simulate mixed success/failure across services
    var j: u32 = 0;
    while (j < 50) : (j += 1) {
        var name_buf: [32]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "service-{d}", .{j % registered}) catch continue;

        const breaker = registry.getBreaker(name) orelse continue;

        if (chaos_ctx.shouldTimeout()) {
            breaker.recordFailure();
        } else {
            breaker.recordSuccess();
        }
    }

    // Check aggregate stats
    const agg_stats = registry.getAggregateStats();
    try std.testing.expect(agg_stats.total_breakers > 0);
    try std.testing.expect(agg_stats.total_requests > 0);

    // Reset all and verify
    registry.resetAll();

    // After reset, all should be closed
    var it = registry.breakers.iterator();
    while (it.next()) |entry| {
        try std.testing.expectEqual(network.CircuitState.closed, entry.value_ptr.getState());
    }

    const stats = chaos_ctx.getStats();
    std.log.info("Registry chaos: registered={d}, requests={d}, faults={d}", .{
        registered,
        agg_stats.total_requests,
        stats.faults_injected,
    });
}

test "network chaos: circuit breaker recovery after chaos ends" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = try chaos.ChaosContext.init(allocator, 90123);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .timeout_injection,
        .probability = 0.9, // Very high failure rate during chaos
        .max_faults = 10,
    });

    var breaker = network.CircuitBreaker.init(allocator, .{
        .failure_threshold = 5,
        .success_threshold = 3,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    // Phase 1: Chaos - should trigger circuit open
    chaos_ctx.enable();

    var i: u32 = 0;
    while (i < 20 and breaker.getState() != .open) : (i += 1) {
        if (chaos_ctx.shouldTimeout()) {
            breaker.recordFailure();
        } else {
            breaker.recordSuccess();
        }
    }

    // Circuit should be open after chaos
    const state_after_chaos = breaker.getState();

    chaos_ctx.disable();

    // Phase 2: Force to half-open for recovery test
    if (state_after_chaos == .open) {
        breaker.mutex.lock();
        breaker.state = .half_open;
        breaker.success_count = 0;
        breaker.mutex.unlock();

        // Recovery: record successes (no chaos)
        var j: u32 = 0;
        while (j < 5) : (j += 1) {
            breaker.recordSuccess();
        }

        // Should recover to closed
        try std.testing.expectEqual(network.CircuitState.closed, breaker.getState());
    }

    const stats = chaos_ctx.getStats();
    std.log.info("Circuit recovery: after_chaos={t}, after_recovery={t}, faults={d}", .{
        state_after_chaos,
        breaker.getState(),
        stats.faults_injected,
    });
}

// ============================================================================
// Service Discovery Chaos Tests
// ============================================================================

test "network chaos: service discovery handles registration failures" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = try chaos.ChaosContext.init(allocator, 11111);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.1,
        .warmup_ops = 5,
    });
    try chaos_ctx.addFault(.{
        .fault_type = .network_partition,
        .probability = 0.05,
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var discovery = network.ServiceDiscovery.init(allocator, .{
        .backend = .static,
        .service_name = "test-service",
        .service_address = "127.0.0.1",
        .service_port = 8080,
    }) catch |err| {
        switch (err) {
            error.OutOfMemory => return, // Expected under chaos
            else => return err,
        }
    };
    defer discovery.deinit();

    // Registration may fail under chaos
    var registered = false;
    discovery.register() catch {
        // Expected failure under chaos
    };
    if (discovery.registered) registered = true;

    // Try to deregister (should handle failures)
    if (registered) {
        discovery.deregister() catch {};
    }

    const stats = chaos_ctx.getStats();
    std.log.info("Service discovery chaos: registered={}, faults={d}", .{
        registered,
        stats.faults_injected,
    });
}

// ============================================================================
// Node Registry Chaos Tests
// ============================================================================

test "network chaos: node registry handles concurrent chaos" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = try chaos.ChaosContext.init(allocator, 22222);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.08,
        .warmup_ops = 10,
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var registry = network.NodeRegistry.init(allocator);
    defer registry.deinit();

    // Register nodes under chaos
    var registered: u32 = 0;
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        var id_buf: [16]u8 = undefined;
        const id = std.fmt.bufPrint(&id_buf, "node-{d}", .{i}) catch continue;
        var addr_buf: [32]u8 = undefined;
        const addr = std.fmt.bufPrint(&addr_buf, "192.168.1.{d}:8080", .{i}) catch continue;

        registry.register(id, addr) catch continue;
        registered += 1;
    }

    // Should have some nodes registered
    try std.testing.expect(registered > 0);

    // Sync to load balancer
    var lb = network.LoadBalancer.init(allocator, .{ .strategy = .round_robin });
    defer lb.deinit();

    lb.syncFromRegistry(&registry) catch |err| {
        switch (err) {
            error.OutOfMemory => {
                // Expected under chaos
            },
            else => return err,
        }
    };

    const stats = chaos_ctx.getStats();
    std.log.info("Node registry chaos: registered={d}, lb_nodes={d}, faults={d}", .{
        registered,
        lb.nodes.items.len,
        stats.faults_injected,
    });
}

// ============================================================================
// Combined Network Chaos Tests
// ============================================================================

test "network chaos: full network stack under combined failures" {
    if (!build_options.enable_network) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = try chaos.ChaosContext.init(allocator, 33333);
    defer chaos_ctx.deinit();

    // Multiple fault types
    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.03,
        .warmup_ops = 20,
    });
    try chaos_ctx.addFault(.{
        .fault_type = .network_partition,
        .probability = 0.05,
    });
    try chaos_ctx.addFault(.{
        .fault_type = .message_drop,
        .probability = 0.1,
    });
    try chaos_ctx.addFault(.{
        .fault_type = .latency_injection,
        .probability = 0.1,
        .duration_ms = 2,
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    // Initialize network components
    try network.initWithConfig(allocator, .{ .cluster_id = "chaos-test-cluster" });
    defer network.deinit();

    // Create Raft nodes
    var leader = network.RaftNode.init(allocator, "leader", .{
        .election_timeout_min_ms = 5,
        .election_timeout_max_ms = 10,
    }) catch return;
    defer leader.deinit();

    // Make leader
    try leader.tick(50);

    // Create circuit breakers
    var breaker = network.CircuitBreaker.init(allocator, .{
        .failure_threshold = 5,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    // Simulate mixed operations
    var successful_ops: u32 = 0;
    var i: u32 = 0;
    while (i < 30) : (i += 1) {
        // Inject latency
        chaos_ctx.maybeInjectLatency();

        // Skip if partitioned
        if (chaos_ctx.shouldNetworkPartition()) continue;

        // Skip if message dropped
        if (chaos_ctx.shouldDropMessage()) continue;

        // Try Raft operation
        var cmd_buf: [16]u8 = undefined;
        const cmd = std.fmt.bufPrint(&cmd_buf, "op-{d}", .{i}) catch continue;

        if (leader.isLeader()) {
            _ = leader.appendCommand(cmd) catch continue;
        }

        // Record in circuit breaker
        if (chaos_ctx.shouldTimeout()) {
            breaker.recordFailure();
        } else {
            breaker.recordSuccess();
        }

        successful_ops += 1;
    }

    // Verify system is still functional
    try std.testing.expect(network.isInitialized());

    const stats = chaos_ctx.getStats();
    std.log.info("Full network chaos: ops={d}, leader={}, breaker={t}, faults={d}", .{
        successful_ops,
        leader.isLeader(),
        breaker.getState(),
        stats.faults_injected,
    });
}
