//! Raft consensus protocol implementation.
//!
//! Provides leader election and log replication for distributed
//! systems requiring strong consistency guarantees.
//!
//! Key components:
//! - Leader election with randomized timeouts
//! - Log replication with AppendEntries RPC
//! - Commit tracking and state machine application
//!
//! Usage:
//!   var node = try RaftNode.init(allocator, "node-1", .{});
//!   defer node.deinit();
//!   try node.addPeer("node-2");
//!   try node.tick(); // Process timeouts and state transitions

const std = @import("std");

/// Raft node state.
pub const RaftState = enum {
    follower,
    candidate,
    leader,

    pub fn toString(self: RaftState) []const u8 {
        return switch (self) {
            .follower => "follower",
            .candidate => "candidate",
            .leader => "leader",
        };
    }
};

/// Error types for Raft operations.
pub const RaftError = error{
    NotLeader,
    LogInconsistent,
    TermOutdated,
    ElectionFailed,
    NoQuorum,
    InvalidLogIndex,
    PeerNotFound,
    OutOfMemory,
    CommitFailed,
};

/// Configuration for Raft node.
pub const RaftConfig = struct {
    /// Minimum election timeout in milliseconds.
    election_timeout_min_ms: u64 = 150,
    /// Maximum election timeout in milliseconds.
    election_timeout_max_ms: u64 = 300,
    /// Heartbeat interval in milliseconds.
    heartbeat_interval_ms: u64 = 50,
    /// Maximum entries per AppendEntries RPC.
    max_entries_per_append: usize = 100,
    /// Enable pre-vote protocol (prevents disruption from partitioned nodes).
    enable_pre_vote: bool = true,
    /// Enable leader lease (allows reads without quorum during lease).
    enable_leader_lease: bool = false,
    /// Leader lease duration in milliseconds.
    leader_lease_ms: u64 = 100,
};

/// Entry in the replicated log.
pub const LogEntry = struct {
    /// Term when entry was created.
    term: u64,
    /// Log index (1-based).
    index: u64,
    /// Command data.
    data: []const u8,
    /// Entry type.
    entry_type: EntryType = .command,

    pub const EntryType = enum {
        command,
        config_change,
        no_op,
    };
};

/// Request for vote RPC.
pub const RequestVoteRequest = struct {
    /// Candidate's term.
    term: u64,
    /// Candidate requesting vote.
    candidate_id: []const u8,
    /// Index of candidate's last log entry.
    last_log_index: u64,
    /// Term of candidate's last log entry.
    last_log_term: u64,
    /// Whether this is a pre-vote request.
    is_pre_vote: bool = false,
};

/// Response to vote request.
pub const RequestVoteResponse = struct {
    /// Current term for candidate to update.
    term: u64,
    /// True if candidate received vote.
    vote_granted: bool,
    /// ID of responding node.
    voter_id: []const u8,
};

/// Append entries RPC request.
pub const AppendEntriesRequest = struct {
    /// Leader's term.
    term: u64,
    /// Leader ID so follower can redirect clients.
    leader_id: []const u8,
    /// Index of log entry immediately preceding new ones.
    prev_log_index: u64,
    /// Term of prev_log_index entry.
    prev_log_term: u64,
    /// Log entries to store (empty for heartbeat).
    entries: []const LogEntry,
    /// Leader's commit index.
    leader_commit: u64,
};

/// Response to append entries.
pub const AppendEntriesResponse = struct {
    /// Current term for leader to update.
    term: u64,
    /// True if follower contained entry matching prev_log_index/term.
    success: bool,
    /// Follower's last log index (for optimized backtracking).
    match_index: u64,
    /// ID of responding node.
    follower_id: []const u8,
};

/// Peer node tracking information.
pub const PeerState = struct {
    /// Peer node ID.
    node_id: []const u8,
    /// Index of next log entry to send.
    next_index: u64,
    /// Index of highest log entry known to be replicated.
    match_index: u64,
    /// Last time heard from this peer.
    last_contact_ms: u64,
    /// Whether vote was granted in current election.
    vote_granted: bool,
    /// Whether pre-vote was granted.
    pre_vote_granted: bool,
};

/// Statistics about Raft node operation.
pub const RaftStats = struct {
    current_term: u64,
    state: RaftState,
    commit_index: u64,
    last_applied: u64,
    log_length: usize,
    peer_count: usize,
    leader_id: ?[]const u8,
    votes_received: usize,
    elections_started: u64,
    elections_won: u64,
    heartbeats_sent: u64,
    entries_replicated: u64,
};

/// Raft consensus node.
pub const RaftNode = struct {
    allocator: std.mem.Allocator,
    /// This node's unique ID.
    node_id: []const u8,
    /// Configuration.
    config: RaftConfig,

    // Persistent state (must persist to stable storage)
    /// Current term (latest term seen).
    current_term: u64,
    /// CandidateId that received vote in current term.
    voted_for: ?[]const u8,
    /// Log entries.
    log: std.ArrayListUnmanaged(LogEntry),

    // Volatile state on all servers
    /// Index of highest log entry known to be committed.
    commit_index: u64,
    /// Index of highest log entry applied to state machine.
    last_applied: u64,
    /// Current state (follower/candidate/leader).
    state: RaftState,
    /// Current leader ID (null if unknown).
    leader_id: ?[]const u8,

    // Volatile state on leaders (reinitialized after election)
    /// For each peer, index of next log entry to send.
    peers: std.StringHashMapUnmanaged(PeerState),

    // Timing
    /// Election timeout (randomized).
    election_timeout_ms: u64,
    /// Time since last heartbeat/vote.
    time_since_last_heartbeat_ms: u64,
    /// Random number generator.
    rng: std.Random.DefaultPrng,

    // Statistics
    elections_started: u64,
    elections_won: u64,
    heartbeats_sent: u64,
    entries_replicated: u64,

    /// Callback type for applying committed entries.
    pub const ApplyCallback = *const fn (entry: LogEntry, user_data: ?*anyopaque) void;

    /// Initialize a new Raft node.
    pub fn init(allocator: std.mem.Allocator, node_id: []const u8, config: RaftConfig) !RaftNode {
        const id_copy = try allocator.dupe(u8, node_id);
        errdefer allocator.free(id_copy);

        var rng = std.Random.DefaultPrng.init(blk: {
            var seed: u64 = 0;
            for (node_id) |c| {
                seed = seed *% 31 +% c;
            }
            // Mix in some entropy from timer
            var timer = std.time.Timer.start() catch break :blk seed;
            seed ^= timer.read();
            break :blk seed;
        });

        const election_timeout = config.election_timeout_min_ms +
            rng.random().uintLessThan(u64, config.election_timeout_max_ms - config.election_timeout_min_ms);

        return RaftNode{
            .allocator = allocator,
            .node_id = id_copy,
            .config = config,
            .current_term = 0,
            .voted_for = null,
            .log = .{},
            .commit_index = 0,
            .last_applied = 0,
            .state = .follower,
            .leader_id = null,
            .peers = .{},
            .election_timeout_ms = election_timeout,
            .time_since_last_heartbeat_ms = 0,
            .rng = rng,
            .elections_started = 0,
            .elections_won = 0,
            .heartbeats_sent = 0,
            .entries_replicated = 0,
        };
    }

    /// Clean up resources.
    pub fn deinit(self: *RaftNode) void {
        // Free log entries
        for (self.log.items) |entry| {
            self.allocator.free(entry.data);
        }
        self.log.deinit(self.allocator);

        // Free peers
        var iter = self.peers.valueIterator();
        while (iter.next()) |peer| {
            self.allocator.free(peer.node_id);
        }
        self.peers.deinit(self.allocator);

        // Free voted_for if set
        if (self.voted_for) |vf| {
            self.allocator.free(vf);
        }

        // Free leader_id if set
        if (self.leader_id) |lid| {
            self.allocator.free(lid);
        }

        self.allocator.free(self.node_id);
        self.* = undefined;
    }

    /// Add a peer node.
    pub fn addPeer(self: *RaftNode, peer_id: []const u8) !void {
        if (std.mem.eql(u8, peer_id, self.node_id)) {
            return; // Don't add self as peer
        }

        const id_copy = try self.allocator.dupe(u8, peer_id);
        errdefer self.allocator.free(id_copy);

        const peer_state = PeerState{
            .node_id = id_copy,
            .next_index = self.getLastLogIndex() + 1,
            .match_index = 0,
            .last_contact_ms = 0,
            .vote_granted = false,
            .pre_vote_granted = false,
        };

        try self.peers.put(self.allocator, id_copy, peer_state);
    }

    /// Remove a peer node.
    pub fn removePeer(self: *RaftNode, peer_id: []const u8) void {
        if (self.peers.fetchRemove(peer_id)) |kv| {
            self.allocator.free(kv.value.node_id);
        }
    }

    /// Get current leader ID.
    pub fn getLeader(self: *const RaftNode) ?[]const u8 {
        return self.leader_id;
    }

    /// Check if this node is the leader.
    pub fn isLeader(self: *const RaftNode) bool {
        return self.state == .leader;
    }

    /// Get node statistics.
    pub fn getStats(self: *const RaftNode) RaftStats {
        return RaftStats{
            .current_term = self.current_term,
            .state = self.state,
            .commit_index = self.commit_index,
            .last_applied = self.last_applied,
            .log_length = self.log.items.len,
            .peer_count = self.peers.count(),
            .leader_id = self.leader_id,
            .votes_received = self.countVotes(),
            .elections_started = self.elections_started,
            .elections_won = self.elections_won,
            .heartbeats_sent = self.heartbeats_sent,
            .entries_replicated = self.entries_replicated,
        };
    }

    /// Append a command to the log (leader only).
    pub fn appendCommand(self: *RaftNode, data: []const u8) !u64 {
        if (self.state != .leader) {
            return RaftError.NotLeader;
        }

        const data_copy = try self.allocator.dupe(u8, data);
        errdefer self.allocator.free(data_copy);

        const entry = LogEntry{
            .term = self.current_term,
            .index = self.getLastLogIndex() + 1,
            .data = data_copy,
            .entry_type = .command,
        };

        try self.log.append(self.allocator, entry);

        return entry.index;
    }

    /// Process a tick (call periodically for timeouts).
    pub fn tick(self: *RaftNode, elapsed_ms: u64) !void {
        self.time_since_last_heartbeat_ms += elapsed_ms;

        switch (self.state) {
            .follower, .candidate => {
                if (self.time_since_last_heartbeat_ms >= self.election_timeout_ms) {
                    try self.startElection();
                }
            },
            .leader => {
                if (self.time_since_last_heartbeat_ms >= self.config.heartbeat_interval_ms) {
                    try self.sendHeartbeats();
                    self.time_since_last_heartbeat_ms = 0;
                }
            },
        }
    }

    /// Handle RequestVote RPC.
    pub fn handleRequestVote(self: *RaftNode, request: RequestVoteRequest) !RequestVoteResponse {
        var response = RequestVoteResponse{
            .term = self.current_term,
            .vote_granted = false,
            .voter_id = self.node_id,
        };

        // If request term < current term, reject
        if (request.term < self.current_term) {
            return response;
        }

        // If request term > current term, update and convert to follower
        if (request.term > self.current_term) {
            try self.stepDown(request.term);
        }

        response.term = self.current_term;

        // Check if we can grant vote
        const can_vote = self.voted_for == null or
            std.mem.eql(u8, self.voted_for.?, request.candidate_id);

        // Check if candidate's log is at least as up-to-date
        const log_ok = self.isLogUpToDate(request.last_log_index, request.last_log_term);

        if (can_vote and log_ok) {
            // Grant vote
            if (self.voted_for) |vf| {
                self.allocator.free(vf);
            }
            self.voted_for = try self.allocator.dupe(u8, request.candidate_id);
            response.vote_granted = true;

            // Reset election timeout
            self.resetElectionTimeout();
        }

        return response;
    }

    /// Handle RequestVote response.
    pub fn handleRequestVoteResponse(self: *RaftNode, response: RequestVoteResponse) !void {
        if (self.state != .candidate) {
            return; // Ignore if not candidate
        }

        if (response.term > self.current_term) {
            try self.stepDown(response.term);
            return;
        }

        if (response.term == self.current_term and response.vote_granted) {
            // Record vote
            if (self.peers.getPtr(response.voter_id)) |peer| {
                peer.vote_granted = true;
            }

            // Check for majority
            if (self.hasQuorum()) {
                try self.becomeLeader();
            }
        }
    }

    /// Handle AppendEntries RPC.
    pub fn handleAppendEntries(self: *RaftNode, request: AppendEntriesRequest) !AppendEntriesResponse {
        var response = AppendEntriesResponse{
            .term = self.current_term,
            .success = false,
            .match_index = 0,
            .follower_id = self.node_id,
        };

        // Reply false if term < currentTerm
        if (request.term < self.current_term) {
            return response;
        }

        // Update term if request has higher term
        if (request.term > self.current_term) {
            try self.stepDown(request.term);
        }

        response.term = self.current_term;

        // Update leader info
        if (self.leader_id) |lid| {
            if (!std.mem.eql(u8, lid, request.leader_id)) {
                self.allocator.free(lid);
                self.leader_id = try self.allocator.dupe(u8, request.leader_id);
            }
        } else {
            self.leader_id = try self.allocator.dupe(u8, request.leader_id);
        }

        // Reset election timeout (we heard from leader)
        self.resetElectionTimeout();
        self.state = .follower;

        // Check if log contains an entry at prevLogIndex with prevLogTerm
        if (request.prev_log_index > 0) {
            if (request.prev_log_index > self.log.items.len) {
                // Log too short
                response.match_index = self.getLastLogIndex();
                return response;
            }

            const prev_entry = self.log.items[request.prev_log_index - 1];
            if (prev_entry.term != request.prev_log_term) {
                // Term mismatch - delete conflicting entries
                self.truncateLog(request.prev_log_index - 1);
                response.match_index = self.getLastLogIndex();
                return response;
            }
        }

        // Append new entries (if any)
        for (request.entries) |entry| {
            const log_idx = entry.index - 1;
            if (log_idx < self.log.items.len) {
                // Entry exists - check for conflict
                if (self.log.items[log_idx].term != entry.term) {
                    self.truncateLog(log_idx);
                    const data_copy = try self.allocator.dupe(u8, entry.data);
                    const new_entry = LogEntry{
                        .term = entry.term,
                        .index = entry.index,
                        .data = data_copy,
                        .entry_type = entry.entry_type,
                    };
                    try self.log.append(self.allocator, new_entry);
                }
            } else {
                // New entry
                const data_copy = try self.allocator.dupe(u8, entry.data);
                const new_entry = LogEntry{
                    .term = entry.term,
                    .index = entry.index,
                    .data = data_copy,
                    .entry_type = entry.entry_type,
                };
                try self.log.append(self.allocator, new_entry);
            }
        }

        // Update commit index
        if (request.leader_commit > self.commit_index) {
            self.commit_index = @min(request.leader_commit, self.getLastLogIndex());
        }

        response.success = true;
        response.match_index = self.getLastLogIndex();
        return response;
    }

    /// Handle AppendEntries response.
    pub fn handleAppendEntriesResponse(self: *RaftNode, response: AppendEntriesResponse) !void {
        if (self.state != .leader) {
            return;
        }

        if (response.term > self.current_term) {
            try self.stepDown(response.term);
            return;
        }

        const peer = self.peers.getPtr(response.follower_id) orelse return;

        if (response.success) {
            // Update next_index and match_index
            peer.match_index = response.match_index;
            peer.next_index = response.match_index + 1;
            self.entries_replicated += 1;

            // Check if we can advance commit_index
            self.advanceCommitIndex();
        } else {
            // Decrement next_index and retry
            if (peer.next_index > 1) {
                // Use match_index hint for faster backtracking
                if (response.match_index > 0) {
                    peer.next_index = response.match_index + 1;
                } else {
                    peer.next_index -= 1;
                }
            }
        }
    }

    /// Build AppendEntries request for a peer.
    pub fn buildAppendEntriesRequest(self: *RaftNode, peer_id: []const u8) ?AppendEntriesRequest {
        const peer = self.peers.get(peer_id) orelse return null;

        const prev_log_index = if (peer.next_index > 1) peer.next_index - 1 else 0;
        const prev_log_term = if (prev_log_index > 0 and prev_log_index <= self.log.items.len)
            self.log.items[prev_log_index - 1].term
        else
            0;

        // Get entries to send
        const start_idx = peer.next_index;
        const end_idx = @min(start_idx + self.config.max_entries_per_append, self.getLastLogIndex() + 1);

        var entries: []const LogEntry = &.{};
        if (start_idx <= self.log.items.len and end_idx > start_idx) {
            entries = self.log.items[start_idx - 1 .. end_idx - 1];
        }

        return AppendEntriesRequest{
            .term = self.current_term,
            .leader_id = self.node_id,
            .prev_log_index = prev_log_index,
            .prev_log_term = prev_log_term,
            .entries = entries,
            .leader_commit = self.commit_index,
        };
    }

    /// Build RequestVote request.
    pub fn buildRequestVoteRequest(self: *const RaftNode) RequestVoteRequest {
        return RequestVoteRequest{
            .term = self.current_term,
            .candidate_id = self.node_id,
            .last_log_index = self.getLastLogIndex(),
            .last_log_term = self.getLastLogTerm(),
            .is_pre_vote = false,
        };
    }

    /// Apply committed entries to state machine.
    pub fn applyCommitted(self: *RaftNode, callback: ApplyCallback, user_data: ?*anyopaque) void {
        while (self.last_applied < self.commit_index) {
            self.last_applied += 1;
            if (self.last_applied <= self.log.items.len) {
                const entry = self.log.items[self.last_applied - 1];
                callback(entry, user_data);
            }
        }
    }

    /// Get list of peer IDs.
    pub fn getPeerIds(self: *const RaftNode) !std.ArrayListUnmanaged([]const u8) {
        var list = std.ArrayListUnmanaged([]const u8){};
        errdefer list.deinit(self.allocator);

        var iter = self.peers.keyIterator();
        while (iter.next()) |key| {
            try list.append(self.allocator, key.*);
        }

        return list;
    }

    // Private methods

    fn startElection(self: *RaftNode) !void {
        self.current_term += 1;
        self.state = .candidate;
        self.elections_started += 1;

        // Vote for self
        if (self.voted_for) |vf| {
            self.allocator.free(vf);
        }
        self.voted_for = try self.allocator.dupe(u8, self.node_id);

        // Reset votes from peers
        var iter = self.peers.valueIterator();
        while (iter.next()) |peer| {
            peer.vote_granted = false;
            peer.pre_vote_granted = false;
        }

        // Reset election timeout
        self.resetElectionTimeout();

        // Check if we already have quorum (single node cluster)
        if (self.hasQuorum()) {
            try self.becomeLeader();
        }
    }

    fn becomeLeader(self: *RaftNode) !void {
        self.state = .leader;
        self.elections_won += 1;

        // Update leader_id
        if (self.leader_id) |lid| {
            if (!std.mem.eql(u8, lid, self.node_id)) {
                self.allocator.free(lid);
                self.leader_id = try self.allocator.dupe(u8, self.node_id);
            }
        } else {
            self.leader_id = try self.allocator.dupe(u8, self.node_id);
        }

        // Initialize peer tracking
        const last_log_idx = self.getLastLogIndex();
        var iter = self.peers.valueIterator();
        while (iter.next()) |peer| {
            peer.next_index = last_log_idx + 1;
            peer.match_index = 0;
        }

        // Append no-op entry
        const noop_data = try self.allocator.dupe(u8, "");
        const noop = LogEntry{
            .term = self.current_term,
            .index = self.getLastLogIndex() + 1,
            .data = noop_data,
            .entry_type = .no_op,
        };
        try self.log.append(self.allocator, noop);

        // Send initial heartbeats
        try self.sendHeartbeats();
    }

    fn stepDown(self: *RaftNode, new_term: u64) !void {
        self.current_term = new_term;
        self.state = .follower;

        // Clear voted_for
        if (self.voted_for) |vf| {
            self.allocator.free(vf);
            self.voted_for = null;
        }

        self.resetElectionTimeout();
    }

    fn sendHeartbeats(self: *RaftNode) !void {
        self.heartbeats_sent += 1;
        self.time_since_last_heartbeat_ms = 0;
        // Heartbeat requests are built via buildAppendEntriesRequest
        // The caller is responsible for actually sending them
    }

    fn resetElectionTimeout(self: *RaftNode) void {
        self.time_since_last_heartbeat_ms = 0;
        self.election_timeout_ms = self.config.election_timeout_min_ms +
            self.rng.random().uintLessThan(u64, self.config.election_timeout_max_ms - self.config.election_timeout_min_ms);
    }

    fn getLastLogIndex(self: *const RaftNode) u64 {
        return self.log.items.len;
    }

    fn getLastLogTerm(self: *const RaftNode) u64 {
        if (self.log.items.len == 0) return 0;
        return self.log.items[self.log.items.len - 1].term;
    }

    fn isLogUpToDate(self: *const RaftNode, last_index: u64, last_term: u64) bool {
        const my_last_term = self.getLastLogTerm();
        const my_last_index = self.getLastLogIndex();

        if (last_term != my_last_term) {
            return last_term > my_last_term;
        }
        return last_index >= my_last_index;
    }

    fn countVotes(self: *const RaftNode) usize {
        var votes: usize = 1; // Self vote
        var iter = self.peers.valueIterator();
        while (iter.next()) |peer| {
            if (peer.vote_granted) votes += 1;
        }
        return votes;
    }

    fn hasQuorum(self: *const RaftNode) bool {
        const votes = self.countVotes();
        const total_nodes = self.peers.count() + 1;
        return votes > total_nodes / 2;
    }

    fn advanceCommitIndex(self: *RaftNode) void {
        // Find the highest N such that a majority of matchIndex[i] >= N
        // and log[N].term == currentTerm
        const last_log_idx = self.getLastLogIndex();
        if (last_log_idx == 0) return;

        var n = last_log_idx;
        while (n > self.commit_index) : (n -= 1) {
            if (n <= self.log.items.len and self.log.items[n - 1].term == self.current_term) {
                var count: usize = 1; // Self
                var iter = self.peers.valueIterator();
                while (iter.next()) |peer| {
                    if (peer.match_index >= n) count += 1;
                }
                const total = self.peers.count() + 1;
                if (count > total / 2) {
                    self.commit_index = n;
                    return;
                }
            }
            if (n == 0) break;
        }
    }

    fn truncateLog(self: *RaftNode, from_index: usize) void {
        while (self.log.items.len > from_index) {
            const entry = self.log.pop();
            self.allocator.free(entry.data);
        }
    }
};

/// Create a Raft cluster for testing/simulation.
pub fn createCluster(allocator: std.mem.Allocator, node_ids: []const []const u8, config: RaftConfig) !std.ArrayListUnmanaged(*RaftNode) {
    var nodes = std.ArrayListUnmanaged(*RaftNode){};
    errdefer {
        for (nodes.items) |node| {
            node.deinit();
            allocator.destroy(node);
        }
        nodes.deinit(allocator);
    }

    // Create nodes
    for (node_ids) |id| {
        const node = try allocator.create(RaftNode);
        errdefer allocator.destroy(node);
        node.* = try RaftNode.init(allocator, id, config);
        try nodes.append(allocator, node);
    }

    // Connect peers
    for (nodes.items) |node| {
        for (node_ids) |peer_id| {
            if (!std.mem.eql(u8, peer_id, node.node_id)) {
                try node.addPeer(peer_id);
            }
        }
    }

    return nodes;
}

test "raft node initialization" {
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    try std.testing.expectEqualStrings("node-1", node.node_id);
    try std.testing.expectEqual(RaftState.follower, node.state);
    try std.testing.expectEqual(@as(u64, 0), node.current_term);
}

test "raft add peers" {
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    try node.addPeer("node-2");
    try node.addPeer("node-3");

    try std.testing.expectEqual(@as(usize, 2), node.peers.count());
}

test "raft vote handling" {
    const allocator = std.testing.allocator;

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

    try std.testing.expect(response.vote_granted);
    try std.testing.expectEqual(@as(u64, 1), node.current_term);
    try std.testing.expectEqualStrings("node-2", node.voted_for.?);
}

test "raft append entries heartbeat" {
    const allocator = std.testing.allocator;

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

    try std.testing.expect(response.success);
    try std.testing.expectEqual(RaftState.follower, node.state);
    try std.testing.expectEqualStrings("node-2", node.leader_id.?);
}

test "raft leader election single node" {
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{
        .election_timeout_min_ms = 10,
        .election_timeout_max_ms = 20,
    });
    defer node.deinit();

    // Trigger election timeout
    try node.tick(100);

    // Single node should become leader
    try std.testing.expectEqual(RaftState.leader, node.state);
    try std.testing.expect(node.isLeader());
}

test "raft stats" {
    const allocator = std.testing.allocator;

    var node = try RaftNode.init(allocator, "node-1", .{});
    defer node.deinit();

    try node.addPeer("node-2");

    const stats = node.getStats();

    try std.testing.expectEqual(@as(usize, 1), stats.peer_count);
    try std.testing.expectEqual(RaftState.follower, stats.state);
    try std.testing.expectEqual(@as(u64, 0), stats.current_term);
}

// ============================================================================
// Raft Persistence Layer
// ============================================================================

/// Persistent state that must survive restarts.
pub const PersistentState = struct {
    /// Current term.
    current_term: u64,
    /// Candidate that received vote in current term (null-terminated).
    voted_for_len: u32,
    voted_for: [256]u8,
    /// Number of log entries.
    log_count: u32,
};

/// Log entry header for persistence.
pub const PersistentLogEntry = struct {
    term: u64,
    index: u64,
    entry_type: u8,
    data_len: u32,
};

/// File format magic number.
const RAFT_MAGIC: u32 = 0x52414654; // "RAFT"
/// Current format version.
const RAFT_VERSION: u16 = 1;

/// Raft persistence manager for durable state storage.
pub const RaftPersistence = struct {
    allocator: std.mem.Allocator,
    path: []const u8,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !RaftPersistence {
        return .{
            .allocator = allocator,
            .path = try allocator.dupe(u8, path),
        };
    }

    pub fn deinit(self: *RaftPersistence) void {
        self.allocator.free(self.path);
    }

    /// Save Raft node state to disk.
    pub fn save(self: *RaftPersistence, node: *const RaftNode) !void {
        // Calculate total size needed
        var total_size: usize = 4 + 2 + @sizeOf(PersistentState); // magic + version + state
        for (node.log.items) |entry| {
            total_size += @sizeOf(PersistentLogEntry) + entry.data.len;
        }

        // Allocate buffer
        const buffer = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(buffer);

        var offset: usize = 0;

        // Write magic
        std.mem.writeInt(u32, buffer[offset..][0..4], RAFT_MAGIC, .little);
        offset += 4;

        // Write version
        std.mem.writeInt(u16, buffer[offset..][0..2], RAFT_VERSION, .little);
        offset += 2;

        // Write persistent state
        var state = PersistentState{
            .current_term = node.current_term,
            .voted_for_len = 0,
            .voted_for = undefined,
            .log_count = @intCast(node.log.items.len),
        };
        @memset(&state.voted_for, 0);
        if (node.voted_for) |vf| {
            const len = @min(vf.len, state.voted_for.len);
            @memcpy(state.voted_for[0..len], vf[0..len]);
            state.voted_for_len = @intCast(len);
        }

        const state_bytes = std.mem.asBytes(&state);
        @memcpy(buffer[offset..][0..state_bytes.len], state_bytes);
        offset += state_bytes.len;

        // Write log entries
        for (node.log.items) |entry| {
            const entry_header = PersistentLogEntry{
                .term = entry.term,
                .index = entry.index,
                .entry_type = @intFromEnum(entry.entry_type),
                .data_len = @intCast(entry.data.len),
            };
            const header_bytes = std.mem.asBytes(&entry_header);
            @memcpy(buffer[offset..][0..header_bytes.len], header_bytes);
            offset += header_bytes.len;

            @memcpy(buffer[offset..][0..entry.data.len], entry.data);
            offset += entry.data.len;
        }

        // Write to file using std.Io
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        var file = try std.Io.Dir.cwd().createFile(io, self.path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, buffer[0..offset]);
    }

    /// Load Raft node state from disk.
    pub fn load(self: *RaftPersistence, node: *RaftNode) !void {
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        const buffer = std.Io.Dir.cwd().readFileAlloc(
            io,
            self.path,
            self.allocator,
            .limited(64 * 1024 * 1024),
        ) catch |err| {
            if (err == error.FileNotFound) return; // No state to load
            return err;
        };
        defer self.allocator.free(buffer);

        if (buffer.len < 6) return error.InvalidFormat;

        var offset: usize = 0;

        // Verify magic
        const magic = std.mem.readInt(u32, buffer[offset..][0..4], .little);
        if (magic != RAFT_MAGIC) return error.InvalidFormat;
        offset += 4;

        // Verify version
        const version = std.mem.readInt(u16, buffer[offset..][0..2], .little);
        if (version != RAFT_VERSION) return error.UnsupportedVersion;
        offset += 2;

        // Read persistent state
        if (buffer.len < offset + @sizeOf(PersistentState)) return error.InvalidFormat;
        const state: *const PersistentState = @ptrCast(@alignCast(buffer[offset..].ptr));
        offset += @sizeOf(PersistentState);

        // Apply state
        node.current_term = state.current_term;

        // Free old voted_for if any
        if (node.voted_for) |vf| {
            node.allocator.free(vf);
            node.voted_for = null;
        }
        if (state.voted_for_len > 0) {
            node.voted_for = try node.allocator.dupe(u8, state.voted_for[0..state.voted_for_len]);
        }

        // Clear existing log
        for (node.log.items) |entry| {
            node.allocator.free(entry.data);
        }
        node.log.clearRetainingCapacity();

        // Read log entries
        var i: u32 = 0;
        while (i < state.log_count) : (i += 1) {
            if (buffer.len < offset + @sizeOf(PersistentLogEntry)) return error.InvalidFormat;
            const entry_header: *const PersistentLogEntry = @ptrCast(@alignCast(buffer[offset..].ptr));
            offset += @sizeOf(PersistentLogEntry);

            if (buffer.len < offset + entry_header.data_len) return error.InvalidFormat;
            const data = try node.allocator.dupe(u8, buffer[offset..][0..entry_header.data_len]);
            errdefer node.allocator.free(data);
            offset += entry_header.data_len;

            try node.log.append(node.allocator, LogEntry{
                .term = entry_header.term,
                .index = entry_header.index,
                .entry_type = @enumFromInt(entry_header.entry_type),
                .data = data,
            });
        }
    }

    /// Check if persistence file exists.
    pub fn exists(self: *RaftPersistence) bool {
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        if (std.Io.Dir.cwd().openFile(io, self.path, .{})) |file| {
            file.close(io);
            return true;
        } else |_| {
            return false;
        }
    }

    pub const LoadError = error{
        InvalidFormat,
        UnsupportedVersion,
    } || std.mem.Allocator.Error || std.Io.Dir.ReadFileAllocError;
};

// ============================================================================
// Raft Snapshots for Log Compaction
// ============================================================================

/// Snapshot metadata.
pub const SnapshotMetadata = struct {
    /// Last included index in the snapshot.
    last_included_index: u64,
    /// Term of last included index.
    last_included_term: u64,
    /// Configuration at snapshot time (serialized peer list).
    config_size: u32,
};

/// Snapshot file magic number.
const SNAPSHOT_MAGIC: u32 = 0x534E4150; // "SNAP"
/// Snapshot format version.
const SNAPSHOT_VERSION: u16 = 1;

/// Raft snapshot manager for log compaction.
pub const RaftSnapshotManager = struct {
    allocator: std.mem.Allocator,
    snapshot_dir: []const u8,
    /// Minimum log entries to keep before compaction.
    min_log_entries: usize,
    /// Threshold for triggering automatic snapshot.
    snapshot_threshold: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        snapshot_dir: []const u8,
        config: SnapshotConfig,
    ) !RaftSnapshotManager {
        return .{
            .allocator = allocator,
            .snapshot_dir = try allocator.dupe(u8, snapshot_dir),
            .min_log_entries = config.min_log_entries,
            .snapshot_threshold = config.snapshot_threshold,
        };
    }

    pub fn deinit(self: *RaftSnapshotManager) void {
        self.allocator.free(self.snapshot_dir);
    }

    /// Create a snapshot of the current state machine.
    pub fn createSnapshot(
        self: *RaftSnapshotManager,
        node: *RaftNode,
        state_machine_data: []const u8,
    ) !void {
        if (node.commit_index == 0) return; // Nothing to snapshot

        const last_included_index = node.commit_index;
        const last_included_term = if (last_included_index > 0 and last_included_index <= node.log.items.len)
            node.log.items[last_included_index - 1].term
        else
            0;

        // Build configuration data (peer list)
        var config_data = std.ArrayListUnmanaged(u8){};
        defer config_data.deinit(self.allocator);

        var iter = node.peers.keyIterator();
        while (iter.next()) |key| {
            try config_data.appendSlice(self.allocator, key.*);
            try config_data.append(self.allocator, '\n');
        }

        // Calculate total size
        const total_size = 4 + 2 + @sizeOf(SnapshotMetadata) + config_data.items.len + state_machine_data.len;

        // Allocate buffer
        const buffer = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(buffer);

        var offset: usize = 0;

        // Write magic
        std.mem.writeInt(u32, buffer[offset..][0..4], SNAPSHOT_MAGIC, .little);
        offset += 4;

        // Write version
        std.mem.writeInt(u16, buffer[offset..][0..2], SNAPSHOT_VERSION, .little);
        offset += 2;

        // Write metadata
        const metadata = SnapshotMetadata{
            .last_included_index = last_included_index,
            .last_included_term = last_included_term,
            .config_size = @intCast(config_data.items.len),
        };
        const metadata_bytes = std.mem.asBytes(&metadata);
        @memcpy(buffer[offset..][0..metadata_bytes.len], metadata_bytes);
        offset += metadata_bytes.len;

        // Write config
        @memcpy(buffer[offset..][0..config_data.items.len], config_data.items);
        offset += config_data.items.len;

        // Write state machine data
        @memcpy(buffer[offset..][0..state_machine_data.len], state_machine_data);

        // Generate snapshot filename
        var filename_buf: [256]u8 = undefined;
        const filename = std.fmt.bufPrint(
            &filename_buf,
            "{s}/snapshot-{d}-{d}.snap",
            .{ self.snapshot_dir, last_included_index, last_included_term },
        ) catch return error.PathTooLong;

        // Write to file
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        // Ensure directory exists
        std.Io.Dir.cwd().makePath(io, self.snapshot_dir) catch |err| {
            std.log.warn("Failed to create snapshot directory '{s}': {t}", .{ self.snapshot_dir, err });
        };

        var file = try std.Io.Dir.cwd().createFile(io, filename, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, buffer[0..offset]);

        // Compact the log
        try self.compactLog(node, last_included_index);
    }

    /// Compact the log by removing entries before the snapshot index.
    fn compactLog(self: *RaftSnapshotManager, node: *RaftNode, snapshot_index: u64) !void {
        if (snapshot_index == 0 or snapshot_index > node.log.items.len) return;

        // Keep at least min_log_entries
        const entries_to_remove = if (snapshot_index > self.min_log_entries)
            snapshot_index - self.min_log_entries
        else
            0;

        if (entries_to_remove == 0) return;

        // Free data for removed entries
        for (node.log.items[0..entries_to_remove]) |entry| {
            node.allocator.free(entry.data);
        }

        // Shift remaining entries
        const remaining = node.log.items.len - entries_to_remove;
        std.mem.copyForwards(
            LogEntry,
            node.log.items[0..remaining],
            node.log.items[entries_to_remove..],
        );
        node.log.shrinkRetainingCapacity(remaining);
    }

    /// Load the latest snapshot.
    pub fn loadLatestSnapshot(
        self: *RaftSnapshotManager,
        node: *RaftNode,
    ) !?[]u8 {
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        // Find latest snapshot file
        var latest_index: u64 = 0;
        var latest_term: u64 = 0;
        var latest_filename: ?[]const u8 = null;

        var dir = std.Io.Dir.cwd().openDir(io, self.snapshot_dir, .{ .iterate = true }) catch |err| {
            if (err == error.FileNotFound) return null;
            return err;
        };
        defer dir.close(io);

        var dir_iter = dir.iterate();
        while (try dir_iter.next(io)) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".snap")) {
                // Parse index and term from filename
                if (parseSnapshotFilename(entry.name)) |parsed| {
                    if (parsed.index > latest_index or
                        (parsed.index == latest_index and parsed.term > latest_term))
                    {
                        latest_index = parsed.index;
                        latest_term = parsed.term;
                        if (latest_filename) |f| self.allocator.free(f);
                        latest_filename = try self.allocator.dupe(u8, entry.name);
                    }
                }
            }
        }

        if (latest_filename == null) return null;
        defer self.allocator.free(latest_filename.?);

        // Load the snapshot
        var path_buf: [512]u8 = undefined;
        const full_path = std.fmt.bufPrint(
            &path_buf,
            "{s}/{s}",
            .{ self.snapshot_dir, latest_filename.? },
        ) catch return error.PathTooLong;

        const buffer = try std.Io.Dir.cwd().readFileAlloc(
            io,
            full_path,
            self.allocator,
            .limited(256 * 1024 * 1024),
        );
        defer self.allocator.free(buffer);

        if (buffer.len < 6 + @sizeOf(SnapshotMetadata)) return error.InvalidSnapshot;

        var offset: usize = 0;

        // Verify magic
        const magic = std.mem.readInt(u32, buffer[offset..][0..4], .little);
        if (magic != SNAPSHOT_MAGIC) return error.InvalidSnapshot;
        offset += 4;

        // Verify version
        const version = std.mem.readInt(u16, buffer[offset..][0..2], .little);
        if (version != SNAPSHOT_VERSION) return error.UnsupportedVersion;
        offset += 2;

        // Read metadata (copy to aligned struct to avoid alignment issues)
        var metadata: SnapshotMetadata = undefined;
        const metadata_bytes = std.mem.asBytes(&metadata);
        if (buffer.len < offset + metadata_bytes.len) return error.InvalidSnapshot;
        @memcpy(metadata_bytes, buffer[offset..][0..metadata_bytes.len]);
        offset += @sizeOf(SnapshotMetadata);

        // Skip config for now
        offset += metadata.config_size;

        // Extract state machine data
        const state_data_len = buffer.len - offset;
        const state_data = try self.allocator.dupe(u8, buffer[offset..][0..state_data_len]);

        // Update node state
        node.commit_index = @max(node.commit_index, metadata.last_included_index);
        node.last_applied = @max(node.last_applied, metadata.last_included_index);

        return state_data;
    }

    /// Check if snapshot is needed based on log size.
    pub fn shouldSnapshot(self: *const RaftSnapshotManager, node: *const RaftNode) bool {
        return node.log.items.len >= self.snapshot_threshold;
    }

    /// Get list of available snapshot files.
    pub fn listSnapshots(self: *RaftSnapshotManager) !std.ArrayListUnmanaged(SnapshotInfo) {
        var snapshots = std.ArrayListUnmanaged(SnapshotInfo){};
        errdefer {
            for (snapshots.items) |s| self.allocator.free(s.filename);
            snapshots.deinit(self.allocator);
        }

        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        var dir = std.Io.Dir.cwd().openDir(io, self.snapshot_dir, .{ .iterate = true }) catch |err| {
            if (err == error.FileNotFound) return snapshots;
            return err;
        };
        defer dir.close(io);

        var dir_iter = dir.iterate();
        while (try dir_iter.next(io)) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".snap")) {
                if (parseSnapshotFilename(entry.name)) |parsed| {
                    try snapshots.append(self.allocator, .{
                        .filename = try self.allocator.dupe(u8, entry.name),
                        .last_included_index = parsed.index,
                        .last_included_term = parsed.term,
                    });
                }
            }
        }

        return snapshots;
    }

    const ParsedSnapshot = struct { index: u64, term: u64 };

    fn parseSnapshotFilename(name: []const u8) ?ParsedSnapshot {
        // Expected format: snapshot-{index}-{term}.snap
        if (!std.mem.startsWith(u8, name, "snapshot-")) return null;
        if (!std.mem.endsWith(u8, name, ".snap")) return null;

        const content = name["snapshot-".len .. name.len - ".snap".len];
        var parts = std.mem.splitScalar(u8, content, '-');

        const index_str = parts.next() orelse return null;
        const term_str = parts.next() orelse return null;

        const index = std.fmt.parseInt(u64, index_str, 10) catch return null;
        const term = std.fmt.parseInt(u64, term_str, 10) catch return null;

        return .{ .index = index, .term = term };
    }

    pub const SnapshotError = error{
        InvalidSnapshot,
        UnsupportedVersion,
        PathTooLong,
    } || std.mem.Allocator.Error || std.Io.Dir.OpenDirError || std.Io.Dir.MakePathError ||
        std.Io.Dir.CreateFileError || std.Io.Dir.ReadFileAllocError;
};

/// Snapshot configuration options.
pub const SnapshotConfig = struct {
    /// Minimum number of log entries to keep after compaction.
    min_log_entries: usize = 100,
    /// Log size threshold to trigger automatic snapshot.
    snapshot_threshold: usize = 10000,
};

/// Information about a snapshot file.
pub const SnapshotInfo = struct {
    filename: []const u8,
    last_included_index: u64,
    last_included_term: u64,
};

// ============================================================================
// InstallSnapshot RPC for Raft
// ============================================================================

/// InstallSnapshot RPC request.
pub const InstallSnapshotRequest = struct {
    /// Leader's term.
    term: u64,
    /// Leader ID so follower can redirect clients.
    leader_id: []const u8,
    /// Last included index in snapshot.
    last_included_index: u64,
    /// Term of last included index.
    last_included_term: u64,
    /// Byte offset in snapshot chunk.
    offset: u64,
    /// Snapshot chunk data.
    data: []const u8,
    /// True if this is the last chunk.
    done: bool,
};

/// InstallSnapshot RPC response.
pub const InstallSnapshotResponse = struct {
    /// Current term for leader to update.
    term: u64,
    /// True if chunk was accepted.
    success: bool,
    /// Follower's current offset for next chunk.
    next_offset: u64,
};

// ============================================================================
// Membership Change Protocol
// ============================================================================

/// Configuration change entry types.
pub const ConfigChangeType = enum(u8) {
    add_node = 0,
    remove_node = 1,
    promote_learner = 2, // Learner -> Voting member
};

/// Configuration change request.
pub const ConfigChangeRequest = struct {
    change_type: ConfigChangeType,
    node_id: []const u8,
    node_address: ?[]const u8,
};

/// Apply a configuration change to the Raft node.
pub fn applyConfigChange(node: *RaftNode, change: ConfigChangeRequest) !void {
    switch (change.change_type) {
        .add_node => {
            try node.addPeer(change.node_id);
        },
        .remove_node => {
            node.removePeer(change.node_id);
        },
        .promote_learner => {
            // Learner promotion would update peer flags
            // For now, learners are not tracked separately
        },
    }
}

test "raft persistence save and load" {
    const allocator = std.testing.allocator;

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
    try std.testing.expectEqual(@as(u64, 5), node2.current_term);
    try std.testing.expectEqual(@as(usize, 2), node2.log.items.len);

    // Clean up test file
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();
    std.Io.Dir.cwd().deleteFile(io, "test_raft_state.bin") catch {};
}
