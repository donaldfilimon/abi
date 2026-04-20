//! Tests for Raft Consensus Protocol
//!
//! Covers node initialization, peer management, vote handling,
//! append entries, leader election, stats, persistence,
//! pre-vote protocol, partition tolerance, and fault injection.

const std = @import("std");
const testing = std.testing;
const raft = @import("raft.zig");
const raft_persistence = @import("raft_persistence.zig");

const RaftNode = raft.RaftNode;
const RaftState = raft.RaftState;
const RequestVoteRequest = raft.RequestVoteRequest;
const RequestVoteResponse = raft.RequestVoteResponse;
const AppendEntriesRequest = raft.AppendEntriesRequest;
const AppendEntriesResponse = raft.AppendEntriesResponse;
const RaftPersistence = raft_persistence.RaftPersistence;
const FaultInjector = raft.FaultInjector;
const initIoBackend = raft.initIoBackend;

// ============================================================================
// Original tests
// ============================================================================

test "raft node initialization" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    try testing.expectEqualStrings("node-1", node.node_id);
    try testing.expectEqual(RaftState.follower, node.state);
    try testing.expectEqual(@as(u64, 0), node.current_term);
}

test "raft add peers" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    try node.addPeer("node-2");
    try node.addPeer("node-3");

    try testing.expectEqual(@as(usize, 2), node.peers.count());
}

test "raft vote handling" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // Request vote with higher term
    const request = RequestVoteRequest{
        .term = 1,
        .candidate_id = "node-2",
        .last_log_index = 0,
        .last_log_term = 0,
    };

    const response = try node.handleRequestVote(request);

    try testing.expect(response.vote_granted);
    try testing.expectEqual(@as(u64, 1), node.current_term);
    try testing.expectEqualStrings("node-2", node.voted_for.?);
}

test "raft append entries heartbeat" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    const request = AppendEntriesRequest{
        .term = 1,
        .leader_id = "node-2",
        .prev_log_index = 0,
        .prev_log_term = 0,
        .entries = &.{},
        .leader_commit = 0,
    };

    const response = try node.handleAppendEntries(request);

    try testing.expect(response.success);
    try testing.expectEqual(RaftState.follower, node.state);
    try testing.expectEqualStrings("node-2", node.leader_id.?);
}

test "raft leader election single node" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 10,
        .election_timeout_max_ms = 20,
    });
    defer node.deinit();

    // Trigger election timeout
    try node.tick(100);

    // Single node should become leader
    try testing.expectEqual(RaftState.leader, node.state);
    try testing.expect(node.isLeader());
}

test "raft stats" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    try node.addPeer("node-2");

    const stats = node.getStats();

    try testing.expectEqual(@as(usize, 1), stats.peer_count);
    try testing.expectEqual(RaftState.follower, stats.state);
    try testing.expectEqual(@as(u64, 0), stats.current_term);
}

test "raft persistence save and load" {
    const allocator = testing.allocator;

    // Create a node with some state
    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // Add some state
    node.state = .leader;
    node.current_term = 5;
    _ = try node.appendCommand("test-command-1");
    _ = try node.appendCommand("test-command-2");

    // Save to persistence
    var persistence = try RaftPersistence.init(allocator, "test_raft_state.bin");
    defer persistence.deinit();
    try persistence.save(&node);

    // Create new node and load
    var node2 = try RaftNode.init(allocator, "node-1", .{});
    defer node2.deinit();

    try persistence.load(&node2);

    // Verify state was restored
    try testing.expectEqual(@as(u64, 5), node2.current_term);
    try testing.expectEqual(@as(usize, 2), node2.log.items.len);

    // Clean up test file
    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();
    std.Io.Dir.cwd().deleteFile(io, "test_raft_state.bin") catch {};
}

// ============================================================================
// Pre-vote protocol tests
// ============================================================================

test "raft pre-vote state toString" {
    try testing.expectEqualStrings("pre_candidate", RaftState.pre_candidate.toString());
}

test "raft pre-vote request handling does not change voter state" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // Send a pre-vote request — should not change node's term or voted_for
    const request = RequestVoteRequest{
        .term = 5,
        .candidate_id = "node-2",
        .last_log_index = 0,
        .last_log_term = 0,
        .is_pre_vote = true,
    };

    const response = try node.handleRequestVote(request);

    // Pre-vote should be granted (candidate log is up-to-date, proposed term >= ours)
    try testing.expect(response.vote_granted);
    // But our term should NOT have changed
    try testing.expectEqual(@as(u64, 0), node.current_term);
    // And we should NOT have recorded a voted_for
    try testing.expect(node.voted_for == null);
}

test "raft pre-vote rejected when candidate log is stale" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // Make the node a leader so it can append, then step it down
    node.state = .leader;
    node.current_term = 3;
    _ = try node.appendCommand("data-1");
    node.state = .follower;

    // Pre-vote from a candidate with older log
    const request = RequestVoteRequest{
        .term = 4,
        .candidate_id = "node-2",
        .last_log_index = 0,
        .last_log_term = 0,
        .is_pre_vote = true,
    };

    const response = try node.handleRequestVote(request);

    // Should be rejected — candidate's log is behind ours
    try testing.expect(!response.vote_granted);
    // Term unchanged
    try testing.expectEqual(@as(u64, 3), node.current_term);
}

test "raft pre-vote prevents term inflation from partitioned follower" {
    const allocator = testing.allocator;

    // Create a follower node with pre-vote enabled and peers
    var node = try RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 10,
        .election_timeout_max_ms = 20,
        .enable_pre_vote = true,
    });
    defer node.deinit();

    try node.addPeer("node-2");
    try node.addPeer("node-3");

    const initial_term = node.current_term;

    // Simulate election timeout — node should enter pre_candidate, NOT candidate
    try node.tick(100);

    // With pre-vote enabled and no responses, node enters pre_candidate
    try testing.expectEqual(RaftState.pre_candidate, node.state);
    // Term should NOT have increased (pre-vote doesn't increment term)
    try testing.expectEqual(initial_term, node.current_term);

    // Tick again — pre-vote times out, restarts pre-vote, still no term increase
    try node.tick(100);
    try testing.expectEqual(RaftState.pre_candidate, node.state);
    try testing.expectEqual(initial_term, node.current_term);
}

test "raft pre-vote succeeds with quorum then proceeds to election" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 10,
        .election_timeout_max_ms = 20,
        .enable_pre_vote = true,
    });
    defer node.deinit();

    try node.addPeer("node-2");
    try node.addPeer("node-3");

    // Trigger pre-vote
    try node.tick(100);
    try testing.expectEqual(RaftState.pre_candidate, node.state);

    // Simulate pre-vote response from node-2 (granted)
    try node.handlePreVoteResponse(RequestVoteResponse{
        .term = 0,
        .vote_granted = true,
        .voter_id = "node-2",
    });

    // With 2/3 pre-votes (self + node-2), quorum reached.
    // Node should have transitioned to candidate via startElection().
    try testing.expectEqual(RaftState.candidate, node.state);
    // Term should now be incremented (real election started)
    try testing.expectEqual(@as(u64, 1), node.current_term);
}

test "raft build pre-vote request uses proposed term" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    node.current_term = 5;

    const request = node.buildPreVoteRequest();

    // Pre-vote request should propose term + 1 without actually changing it
    try testing.expectEqual(@as(u64, 6), request.term);
    try testing.expect(request.is_pre_vote);
    try testing.expectEqualStrings("node-1", request.candidate_id);
    // Actual term unchanged
    try testing.expectEqual(@as(u64, 5), node.current_term);
}

test "raft single node with pre-vote skips to election" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 10,
        .election_timeout_max_ms = 20,
        .enable_pre_vote = true,
    });
    defer node.deinit();

    // No peers — single node cluster
    // Tick should skip pre-vote and go straight to leader
    try node.tick(100);

    try testing.expectEqual(RaftState.leader, node.state);
    try testing.expectEqual(@as(u64, 1), node.current_term);
}

// ============================================================================
// Partition tolerance tests
// ============================================================================

test "raft partitioned leader steps down after losing quorum" {
    const allocator = testing.allocator;

    // Create a leader node
    var node = try RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 100,
        .election_timeout_max_ms = 200,
        .heartbeat_interval_ms = 10,
        .enable_pre_vote = false,
    });
    defer node.deinit();

    try node.addPeer("node-2");
    try node.addPeer("node-3");

    // Force to leader state
    node.current_term = 1;
    node.state = .leader;
    if (node.leader_id) |lid| allocator.free(lid);
    node.leader_id = try allocator.dupe(u8, "node-1");

    try testing.expectEqual(RaftState.leader, node.state);

    // Simulate partition: tick enough for peer contact to exceed election timeout.
    // Each tick ages peer contacts. Use large tick to exceed the election timeout range.
    try node.tick(300);

    // Leader should have stepped down because quorum was lost
    try testing.expectEqual(RaftState.follower, node.state);
}

test "raft leader remains leader when quorum is maintained" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 100,
        .election_timeout_max_ms = 200,
        .heartbeat_interval_ms = 10,
        .enable_pre_vote = false,
    });
    defer node.deinit();

    try node.addPeer("node-2");
    try node.addPeer("node-3");

    // Force leader state
    node.current_term = 1;
    node.state = .leader;
    if (node.leader_id) |lid| allocator.free(lid);
    node.leader_id = try allocator.dupe(u8, "node-1");

    // Tick a small amount
    try node.tick(5);

    // Simulate response from node-2 (resets its contact timer)
    try node.handleAppendEntriesResponse(AppendEntriesResponse{
        .term = 1,
        .success = true,
        .match_index = 0,
        .follower_id = "node-2",
    });

    // Tick again — node-2 just responded so quorum is maintained (self + node-2 = 2/3)
    try node.tick(5);

    // Should still be leader
    try testing.expectEqual(RaftState.leader, node.state);
}

test "raft append entries response resets peer contact timer" {
    const allocator = testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{
        .heartbeat_interval_ms = 10,
        .enable_pre_vote = false,
    });
    defer node.deinit();

    try node.addPeer("node-2");

    // Force leader state
    node.current_term = 1;
    node.state = .leader;

    // Age the peer contact
    if (node.peers.getPtr("node-2")) |peer| {
        peer.last_contact_ms = 500;
    }

    // Receive response from node-2
    try node.handleAppendEntriesResponse(AppendEntriesResponse{
        .term = 1,
        .success = true,
        .match_index = 0,
        .follower_id = "node-2",
    });

    // Contact timer should be reset to 0
    const peer = node.peers.get("node-2").?;
    try testing.expectEqual(@as(u64, 0), peer.last_contact_ms);
}

// ============================================================================
// Fault injection tests
// ============================================================================

test "raft fault injector simulate partition and heal" {
    const allocator = testing.allocator;

    var fi = FaultInjector.init(allocator);
    defer fi.deinit();

    // Initially no partition
    try testing.expect(!fi.isBlocked("node-1", "node-2"));
    try testing.expect(!fi.isBlocked("node-2", "node-1"));

    // Simulate partition
    try fi.simulatePartition("node-1", "node-2");

    try testing.expect(fi.isBlocked("node-1", "node-2"));
    try testing.expect(fi.isBlocked("node-2", "node-1"));
    // Other routes unaffected
    try testing.expect(!fi.isBlocked("node-1", "node-3"));

    // Heal partition
    fi.simulateHeal("node-1", "node-2");

    try testing.expect(!fi.isBlocked("node-1", "node-2"));
    try testing.expect(!fi.isBlocked("node-2", "node-1"));
}

test "raft fault injector multiple partitions" {
    const allocator = testing.allocator;

    var fi = FaultInjector.init(allocator);
    defer fi.deinit();

    try fi.simulatePartition("node-1", "node-2");
    try fi.simulatePartition("node-1", "node-3");

    try testing.expect(fi.isBlocked("node-1", "node-2"));
    try testing.expect(fi.isBlocked("node-1", "node-3"));
    try testing.expect(!fi.isBlocked("node-2", "node-3"));

    // Heal only one partition
    fi.simulateHeal("node-1", "node-2");
    try testing.expect(!fi.isBlocked("node-1", "node-2"));
    try testing.expect(fi.isBlocked("node-1", "node-3"));
}
