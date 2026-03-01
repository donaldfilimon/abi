//! Network Raft Consensus Tests — Extended Coverage
//!
//! Tests multi-node election, log replication, vote rejection,
//! append entries with entries, stats tracking, and edge cases.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const raft = if (build_options.enable_network) abi.features.network.raft else struct {};
const RaftNode = if (build_options.enable_network) raft.RaftNode else struct {};
const RaftState = if (build_options.enable_network) raft.RaftState else struct {};
const RequestVoteRequest = if (build_options.enable_network) raft.RequestVoteRequest else struct {};
const AppendEntriesRequest = if (build_options.enable_network) raft.AppendEntriesRequest else struct {};
const LogEntry = if (build_options.enable_network) raft.LogEntry else struct {};

// ============================================================================
// RaftState Tests
// ============================================================================

test "raft: RaftState toString" {
    if (!build_options.enable_network) return error.SkipZigTest;

    try std.testing.expectEqualStrings("follower", RaftState.follower.toString());
    try std.testing.expectEqualStrings("candidate", RaftState.candidate.toString());
    try std.testing.expectEqualStrings("leader", RaftState.leader.toString());
}

// ============================================================================
// Vote Rejection Tests
// ============================================================================

test "raft: reject vote with lower term" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // Manually set term to 5
    node.current_term = 5;

    // Request vote with lower term (2) — should be rejected
    const request = RequestVoteRequest{
        .term = 2,
        .candidate_id = "node-2",
        .last_log_index = 0,
        .last_log_term = 0,
    };

    const response = try node.handleRequestVote(request);

    try std.testing.expect(!response.vote_granted);
    try std.testing.expectEqual(@as(u64, 5), response.term);
}

test "raft: reject second vote in same term" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // First vote — should be granted
    const request1 = RequestVoteRequest{
        .term = 1,
        .candidate_id = "node-2",
        .last_log_index = 0,
        .last_log_term = 0,
    };
    const response1 = try node.handleRequestVote(request1);
    try std.testing.expect(response1.vote_granted);

    // Second vote in same term from different candidate — should be rejected
    const request2 = RequestVoteRequest{
        .term = 1,
        .candidate_id = "node-3",
        .last_log_index = 0,
        .last_log_term = 0,
    };
    const response2 = try node.handleRequestVote(request2);
    try std.testing.expect(!response2.vote_granted);
}

// ============================================================================
// Peer Management Tests
// ============================================================================

test "raft: addPeer self is no-op" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // Adding self as peer should be silently ignored
    try node.addPeer("node-1");
    try std.testing.expectEqual(@as(usize, 0), node.peers.count());
}

test "raft: removePeer removes existing peer" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    try node.addPeer("node-2");
    try node.addPeer("node-3");
    try std.testing.expectEqual(@as(usize, 2), node.peers.count());

    node.removePeer("node-2");
    try std.testing.expectEqual(@as(usize, 1), node.peers.count());
}

test "raft: removePeer nonexistent is no-op" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    try node.addPeer("node-2");

    // Remove nonexistent peer — should not crash
    node.removePeer("node-99");
    try std.testing.expectEqual(@as(usize, 1), node.peers.count());
}

// ============================================================================
// AppendEntries with Entries Tests
// ============================================================================

test "raft: appendEntries with log entries" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    const entries = [_]LogEntry{
        .{ .term = 1, .index = 1, .data = "cmd-1", .entry_type = .command },
        .{ .term = 1, .index = 2, .data = "cmd-2", .entry_type = .command },
    };

    const request = AppendEntriesRequest{
        .term = 1,
        .leader_id = "leader-1",
        .prev_log_index = 0,
        .prev_log_term = 0,
        .entries = &entries,
        .leader_commit = 0,
    };

    const response = try node.handleAppendEntries(request);

    try std.testing.expect(response.success);
    try std.testing.expectEqual(@as(u64, 2), response.match_index);
    try std.testing.expectEqual(@as(usize, 2), node.log.items.len);
}

test "raft: appendEntries updates commit index" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // First append some entries
    const entries = [_]LogEntry{
        .{ .term = 1, .index = 1, .data = "cmd-1", .entry_type = .command },
        .{ .term = 1, .index = 2, .data = "cmd-2", .entry_type = .command },
    };

    _ = try node.handleAppendEntries(.{
        .term = 1,
        .leader_id = "leader-1",
        .prev_log_index = 0,
        .prev_log_term = 0,
        .entries = &entries,
        .leader_commit = 0,
    });

    // Now send heartbeat with updated commit index
    const response = try node.handleAppendEntries(.{
        .term = 1,
        .leader_id = "leader-1",
        .prev_log_index = 2,
        .prev_log_term = 1,
        .entries = &.{},
        .leader_commit = 2,
    });

    try std.testing.expect(response.success);
    try std.testing.expectEqual(@as(u64, 2), node.commit_index);
}

test "raft: appendEntries rejects stale term" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // Set node to term 5
    node.current_term = 5;

    const request = AppendEntriesRequest{
        .term = 3, // lower than current
        .leader_id = "old-leader",
        .prev_log_index = 0,
        .prev_log_term = 0,
        .entries = &.{},
        .leader_commit = 0,
    };

    const response = try node.handleAppendEntries(request);
    try std.testing.expect(!response.success);
    try std.testing.expectEqual(@as(u64, 5), response.term);
}

// ============================================================================
// Leader Command Append Tests
// ============================================================================

test "raft: appendCommand fails when not leader" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // Node starts as follower — appendCommand should fail
    const result = node.appendCommand("test-command");
    try std.testing.expectError(error.NotLeader, result);
}

test "raft: appendCommand succeeds as leader" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Single-node cluster becomes leader via election
    var node = try RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 10,
        .election_timeout_max_ms = 20,
    });
    defer node.deinit();

    // Trigger election
    try node.tick(100);
    try std.testing.expect(node.isLeader());

    // Now append commands (election adds a no-op entry, so indices start after that)
    const idx1 = try node.appendCommand("cmd-1");
    const idx2 = try node.appendCommand("cmd-2");

    // Indices should be sequential (starting after any no-op entries from election)
    try std.testing.expect(idx1 >= 1);
    try std.testing.expect(idx2 == idx1 + 1);
    try std.testing.expect(node.log.items.len >= 2);
}

// ============================================================================
// Stats Tests
// ============================================================================

test "raft: stats track election and log info" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 10,
        .election_timeout_max_ms = 20,
    });
    defer node.deinit();

    try node.addPeer("node-2");
    try node.addPeer("node-3");

    var stats = node.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.peer_count);
    try std.testing.expectEqual(RaftState.follower, stats.state);
    try std.testing.expectEqual(@as(usize, 0), stats.log_length);
    try std.testing.expectEqual(@as(u64, 0), stats.commit_index);
}

test "raft: getLeader returns null initially" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    try std.testing.expect(node.getLeader() == null);
    try std.testing.expect(!node.isLeader());
}

test "raft: getLeader after heartbeat" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    // Receive heartbeat from leader
    _ = try node.handleAppendEntries(.{
        .term = 1,
        .leader_id = "leader-99",
        .prev_log_index = 0,
        .prev_log_term = 0,
        .entries = &.{},
        .leader_commit = 0,
    });

    const leader = node.getLeader();
    try std.testing.expect(leader != null);
    try std.testing.expectEqualStrings("leader-99", leader.?);
}

// ============================================================================
// Step Down on Higher Term Tests
// ============================================================================

test "raft: higher term vote request causes step down" {
    if (!build_options.enable_network) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Single node → becomes leader
    var node = try RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 10,
        .election_timeout_max_ms = 20,
    });
    defer node.deinit();

    try node.tick(100);
    try std.testing.expect(node.isLeader());

    // Receive vote request with higher term and up-to-date log → should step down and grant vote
    const last_log_idx = node.log.items.len;
    const last_log_term = if (last_log_idx > 0) node.log.items[last_log_idx - 1].term else 0;

    const response = try node.handleRequestVote(.{
        .term = node.current_term + 10,
        .candidate_id = "node-2",
        .last_log_index = @intCast(last_log_idx),
        .last_log_term = last_log_term,
    });

    // Higher term forces step down to follower regardless of vote outcome
    try std.testing.expectEqual(RaftState.follower, node.state);
    // Vote should be granted since candidate's log is at least as up-to-date
    try std.testing.expect(response.vote_granted);
}
