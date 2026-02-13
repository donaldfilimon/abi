//! Tests for Raft Consensus Protocol
//!
//! Covers node initialization, peer management, vote handling,
//! append entries, leader election, stats, and persistence.

const std = @import("std");
const testing = std.testing;
const raft = @import("raft.zig");
const raft_persistence = @import("raft_persistence.zig");

const RaftNode = raft.RaftNode;
const RaftState = raft.RaftState;
const RequestVoteRequest = raft.RequestVoteRequest;
const AppendEntriesRequest = raft.AppendEntriesRequest;
const RaftPersistence = raft_persistence.RaftPersistence;
const initIoBackend = raft.initIoBackend;

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
    node.current_term = 5;
    try node.appendCommand("test-command-1");
    try node.appendCommand("test-command-2");

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
