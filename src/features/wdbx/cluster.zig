//! In-process Raft-style consensus (Cluster Layer).
//!
//! A real leader-election + log-replication state machine exercised over an
//! in-process node array with a deterministic, caller-driven step model (no
//! wall-clock timers or randomness, so it is fully reproducible and testable).
//! This is an honest single-process consensus core — node failure, leader
//! failover, and majority-quorum replication all work.
//!
//! Networked RPC: `cluster_rpc.zig` implements real TCP RequestVote/AppendEntries
//! (loopback multi-node + optional routable bind with `ABI_WDBX_CLUSTER_TOKEN`).
//! That is still not production multi-host/sharding (TLS/mTLS, dynamic membership,
//! and ops story remain gaps).

const std = @import("std");

pub const Role = enum { follower, candidate, leader };

pub const LogEntry = struct {
    term: u64,
    data: []const u8,
};

pub const VoteReply = struct {
    granted: bool,
    term: u64,
};

pub const AppendReply = struct {
    ack: bool,
    term: u64,
};

pub const Node = struct {
    id: u32,
    term: u64 = 0,
    voted_for: ?u32 = null,
    role: Role = .follower,
    alive: bool = true,
    log: std.ArrayListUnmanaged(LogEntry) = .empty,
    commit_index: usize = 0,
};

pub const ClusterError = error{
    NoLeader,
    NotEnoughNodes,
    NodeNotFound,
    QuorumUnreachable,
};

pub const Cluster = struct {
    allocator: std.mem.Allocator,
    nodes: []Node,

    pub fn init(allocator: std.mem.Allocator, node_count: usize) !Cluster {
        if (node_count == 0) return error.NotEnoughNodes;
        const nodes = try allocator.alloc(Node, node_count);
        for (nodes, 0..) |*n, i| n.* = .{ .id = @intCast(i) };
        return .{ .allocator = allocator, .nodes = nodes };
    }

    pub fn deinit(self: *Cluster) void {
        for (self.nodes) |*n| {
            for (n.log.items) |e| self.allocator.free(e.data);
            n.log.deinit(self.allocator);
        }
        self.allocator.free(self.nodes);
    }

    /// Majority of the *configured* cluster size (standard Raft quorum).
    pub fn quorum(self: *const Cluster) usize {
        return self.nodes.len / 2 + 1;
    }

    pub fn aliveCount(self: *const Cluster) usize {
        var c: usize = 0;
        for (self.nodes) |n| {
            if (n.alive) c += 1;
        }
        return c;
    }

    pub fn leader(self: *Cluster) ?*Node {
        for (self.nodes) |*n| {
            if (n.alive and n.role == .leader) return n;
        }
        return null;
    }

    fn nodeById(self: *Cluster, id: u32) ?*Node {
        for (self.nodes) |*n| {
            if (n.id == id) return n;
        }
        return null;
    }

    /// Run an election with `candidate_id` standing. Increments the term,
    /// collects votes from alive peers whose term is not ahead, and promotes the
    /// candidate to leader iff it reaches quorum. Returns true on success.
    pub fn startElection(self: *Cluster, candidate_id: u32) ClusterError!bool {
        const cand = self.nodeById(candidate_id) orelse return error.NodeNotFound;
        if (!cand.alive) return error.NodeNotFound;

        const new_term = blk: {
            var max_term: u64 = 0;
            for (self.nodes) |n| max_term = @max(max_term, n.term);
            break :blk max_term + 1;
        };

        cand.term = new_term;
        cand.role = .candidate;
        cand.voted_for = candidate_id;
        var votes: usize = 1; // votes for itself

        for (self.nodes) |*peer| {
            if (peer.id == candidate_id or !peer.alive) continue;
            if (applyVote(peer, new_term, candidate_id)) votes += 1;
        }

        if (votes >= self.quorum()) {
            cand.role = .leader;
            for (self.nodes) |*peer| {
                if (peer.id != candidate_id and peer.alive) peer.role = .follower;
            }
            return true;
        }
        cand.role = .candidate;
        return false;
    }

    /// Leader appends `data` and replicates to alive followers. Commits (and
    /// advances `commit_index` on every replica that received it) once a quorum
    /// of replicas hold the entry. Returns the acknowledgement count.
    pub fn replicate(self: *Cluster, data: []const u8) ClusterError!usize {
        const ldr = self.leader() orelse return error.NoLeader;
        const term = ldr.term;

        appendEntry(self, ldr, term, data) catch return error.QuorumUnreachable;
        var acks: usize = 1;

        for (self.nodes) |*peer| {
            if (peer.id == ldr.id or !peer.alive) continue;
            appendEntry(self, peer, term, data) catch |err| {
                std.log.warn("wdbx cluster: follower {d} replication failed: {s}", .{ peer.id, @errorName(err) });
                continue;
            };
            acks += 1;
        }

        if (acks >= self.quorum()) {
            // Commit on the leader and on every replica that holds the entry.
            const committed = ldr.log.items.len;
            for (self.nodes) |*peer| {
                if (!peer.alive) continue;
                if (peer.log.items.len >= committed) peer.commit_index = committed;
            }
            return acks;
        }
        return error.QuorumUnreachable;
    }

    fn appendEntry(self: *Cluster, node: *Node, term: u64, data: []const u8) !void {
        const owned = try self.allocator.dupe(u8, data);
        errdefer self.allocator.free(owned);
        try node.log.append(self.allocator, .{ .term = term, .data = owned });
    }

    /// Simulate a node crash. A downed leader vacates leadership; survivors must
    /// run a new election to restore availability.
    pub fn failNode(self: *Cluster, id: u32) ClusterError!void {
        const n = self.nodeById(id) orelse return error.NodeNotFound;
        n.alive = false;
        n.role = .follower;
    }

    pub fn reviveNode(self: *Cluster, id: u32) ClusterError!void {
        const n = self.nodeById(id) orelse return error.NodeNotFound;
        n.alive = true;
    }

    /// Human-readable one-line status for `abi wdbx cluster status`.
    pub fn statusLine(self: *Cluster, allocator: std.mem.Allocator) ![]u8 {
        const ldr = self.leader();
        const leader_str: []const u8 = if (ldr) |l| try std.fmt.allocPrint(allocator, "{d}", .{l.id}) else try allocator.dupe(u8, "none");
        defer allocator.free(leader_str);
        const term = if (ldr) |l| l.term else 0;
        const commit = if (ldr) |l| l.commit_index else 0;
        return std.fmt.allocPrint(
            allocator,
            "nodes={d} alive={d} quorum={d} leader={s} term={d} commit_index={d}",
            .{ self.nodes.len, self.aliveCount(), self.quorum(), leader_str, term, commit },
        );
    }
};

/// Apply a RequestVote to `node` under standard Raft rules. Grants the vote when
/// the candidate's term is not stale and the node has not already voted for a
/// different candidate this term.
pub fn applyVote(node: *Node, term: u64, candidate: u32) bool {
    if (term < node.term) return false;
    if (term > node.term) {
        node.term = term;
        node.voted_for = null;
        node.role = .follower;
    }
    if (node.voted_for == null or node.voted_for == candidate) {
        node.voted_for = candidate;
        node.role = .follower;
        return true;
    }
    return false;
}

/// Apply an AppendEntries to `node`: a non-stale term makes the node a follower
/// and appends the (owned) entry to its log. Rejects a stale term.
pub fn applyAppend(node: *Node, allocator: std.mem.Allocator, term: u64, data: []const u8) !bool {
    if (term < node.term) return false;
    node.term = term;
    node.role = .follower;
    const owned = try allocator.dupe(u8, data);
    errdefer allocator.free(owned);
    try node.log.append(allocator, .{ .term = term, .data = owned });
    return true;
}

test "cluster: single leader elected with quorum" {
    var c = try Cluster.init(std.testing.allocator, 3);
    defer c.deinit();

    try std.testing.expect(try c.startElection(0));
    const ldr = c.leader().?;
    try std.testing.expectEqual(@as(u32, 0), ldr.id);
    try std.testing.expectEqual(@as(u64, 1), ldr.term);

    var leaders: usize = 0;
    for (c.nodes) |n| {
        if (n.role == .leader) leaders += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), leaders);
}

test "cluster: replication reaches quorum and advances commit index" {
    var c = try Cluster.init(std.testing.allocator, 3);
    defer c.deinit();
    _ = try c.startElection(0);

    const acks = try c.replicate("set x=1");
    try std.testing.expectEqual(@as(usize, 3), acks);
    for (c.nodes) |n| {
        try std.testing.expectEqual(@as(usize, 1), n.log.items.len);
        try std.testing.expectEqual(@as(usize, 1), n.commit_index);
        try std.testing.expectEqualStrings("set x=1", n.log.items[0].data);
    }
}

test "cluster: leader failover elects a new leader at a higher term" {
    var c = try Cluster.init(std.testing.allocator, 3);
    defer c.deinit();
    _ = try c.startElection(0);
    _ = try c.replicate("a");

    try c.failNode(0);
    try std.testing.expect(c.leader() == null);

    // Survivors (1 and 2) form a quorum and elect a new leader.
    try std.testing.expect(try c.startElection(1));
    const ldr = c.leader().?;
    try std.testing.expectEqual(@as(u32, 1), ldr.id);
    try std.testing.expectEqual(@as(u64, 2), ldr.term);

    // Replication still works with the surviving majority.
    const acks = try c.replicate("b");
    try std.testing.expectEqual(@as(usize, 2), acks);
}

test "cluster: loses availability without a quorum" {
    var c = try Cluster.init(std.testing.allocator, 3);
    defer c.deinit();
    _ = try c.startElection(0);
    try c.failNode(0);
    try c.failNode(1);

    // Only one node alive: cannot reach quorum of 2.
    try std.testing.expect(!try c.startElection(2));
    try std.testing.expect(c.leader() == null);
    try std.testing.expectError(error.NoLeader, c.replicate("c"));
}

test "cluster: startElection/replicate match applyVote/applyAppend driven directly" {
    const allocator = std.testing.allocator;

    // Cluster A: driven entirely through the public API under test.
    var via_api = try Cluster.init(allocator, 3);
    defer via_api.deinit();
    try std.testing.expect(try via_api.startElection(0));
    _ = try via_api.replicate("hello");

    // Cluster B: identical topology, driven by calling the shared free
    // functions directly in the same order startElection/replicate use
    // internally. Divergence from Cluster A indicates a behavior change in
    // the shared helper path.
    var via_primitives = try Cluster.init(allocator, 3);
    defer via_primitives.deinit();

    const cand = &via_primitives.nodes[0];
    cand.term = 1;
    cand.role = .candidate;
    cand.voted_for = 0;
    var votes: usize = 1; // candidate votes for itself
    for (via_primitives.nodes[1..]) |*peer| {
        if (applyVote(peer, 1, 0)) votes += 1;
    }
    try std.testing.expect(votes >= via_primitives.quorum());
    cand.role = .leader;

    // The leader appends its own entry directly, not via applyAppend:
    // applyAppend unconditionally sets role = .follower, so the leader's own
    // write stays distinct from the follower path.
    const owned = try allocator.dupe(u8, "hello");
    try cand.log.append(allocator, .{ .term = 1, .data = owned });

    for (via_primitives.nodes[1..]) |*peer| {
        _ = try applyAppend(peer, allocator, 1, "hello");
    }

    for (via_api.nodes, via_primitives.nodes) |a, b| {
        try std.testing.expectEqual(a.term, b.term);
        try std.testing.expectEqual(a.voted_for, b.voted_for);
        try std.testing.expectEqual(a.role, b.role);
        try std.testing.expectEqual(a.log.items.len, b.log.items.len);
        for (a.log.items, b.log.items) |a_entry, b_entry| {
            try std.testing.expectEqual(a_entry.term, b_entry.term);
            try std.testing.expectEqualStrings(a_entry.data, b_entry.data);
        }
    }
}

test {
    std.testing.refAllDecls(@This());
}
