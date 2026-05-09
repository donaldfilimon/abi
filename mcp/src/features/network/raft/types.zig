//! Raft protocol types and configurations.

const std = @import("std");

/// Raft node state.
pub const RaftState = enum {
    follower,
    pre_candidate,
    candidate,
    leader,

    pub fn toString(self: RaftState) []const u8 {
        return switch (self) {
            .follower => "follower",
            .pre_candidate => "pre_candidate",
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

/// Callback type for applying committed entries.
pub const ApplyCallback = *const fn (entry: LogEntry, user_data: ?*anyopaque) void;
