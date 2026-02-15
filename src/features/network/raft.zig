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
const time = @import("../../services/shared/time.zig");
const sync = @import("../../services/shared/sync.zig");

// Re-export persistence and snapshot types for backward compatibility
const raft_persistence = @import("raft_persistence.zig");
pub const PersistentState = raft_persistence.PersistentState;
pub const PersistentLogEntry = raft_persistence.PersistentLogEntry;
pub const RaftPersistence = raft_persistence.RaftPersistence;

const raft_snapshot = @import("raft_snapshot.zig");
pub const SnapshotMetadata = raft_snapshot.SnapshotMetadata;
pub const RaftSnapshotManager = raft_snapshot.RaftSnapshotManager;
pub const SnapshotConfig = raft_snapshot.SnapshotConfig;
pub const SnapshotInfo = raft_snapshot.SnapshotInfo;
pub const InstallSnapshotRequest = raft_snapshot.InstallSnapshotRequest;
pub const InstallSnapshotResponse = raft_snapshot.InstallSnapshotResponse;
pub const ConfigChangeType = raft_snapshot.ConfigChangeType;
pub const ConfigChangeRequest = raft_snapshot.ConfigChangeRequest;
pub const applyConfigChange = raft_snapshot.applyConfigChange;

pub fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

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
            var timer = time.Timer.start() catch break :blk seed;
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
                const new_leader = try self.allocator.dupe(u8, request.leader_id);
                self.allocator.free(lid);
                self.leader_id = new_leader;
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
                    errdefer self.allocator.free(data_copy);
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
                errdefer self.allocator.free(data_copy);
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
        var list = std.ArrayListUnmanaged([]const u8).empty;
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
                const new_leader = try self.allocator.dupe(u8, self.node_id);
                self.allocator.free(lid);
                self.leader_id = new_leader;
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
        errdefer self.allocator.free(noop_data);
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
            if (self.log.pop()) |entry| {
                self.allocator.free(entry.data);
            }
        }
    }
};

/// Create a Raft cluster for testing/simulation.
pub fn createCluster(allocator: std.mem.Allocator, node_ids: []const []const u8, config: RaftConfig) !std.ArrayListUnmanaged(*RaftNode) {
    var nodes = std.ArrayListUnmanaged(*RaftNode).empty;
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

test {
    _ = @import("raft_persistence.zig");
    _ = @import("raft_snapshot.zig");
    _ = @import("raft_test.zig");
}
