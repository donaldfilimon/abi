const std = @import("std");

pub const RaftNode = struct {
    allocator: std.mem.Allocator,
    node_id: []const u8,
    config: RaftConfig,

    pub fn init(_: std.mem.Allocator, _: []const u8, _: RaftConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}

    pub fn getStats(_: *const @This()) RaftStats {
        return .{};
    }

    pub fn getLeader(_: *const @This()) ?[]const u8 {
        return null;
    }

    pub fn isLeader(_: *const @This()) bool {
        return false;
    }

    pub const ApplyCallback = *const fn (entry: LogEntry, user_data: ?*anyopaque) void;
};

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

pub const RaftConfig = struct {
    node_id: []const u8 = "",
    election_timeout_min_ms: u64 = 150,
    election_timeout_max_ms: u64 = 300,
    heartbeat_interval_ms: u64 = 50,
    max_entries_per_append: usize = 100,
    enable_pre_vote: bool = true,
    enable_leader_lease: bool = false,
    leader_lease_ms: u64 = 100,
};

pub const RaftError = error{
    NetworkDisabled,
    NotLeader,
    ElectionFailed,
    LogInconsistency,
    NoQuorum,
    InvalidLogIndex,
    PeerNotFound,
    OutOfMemory,
    CommitFailed,
};

pub const RaftStats = struct {
    current_term: u64 = 0,
    state: RaftState = .follower,
    commit_index: u64 = 0,
    last_applied: u64 = 0,
    log_length: usize = 0,
    peer_count: usize = 0,
    leader_id: ?[]const u8 = null,
    votes_received: usize = 0,
    elections_started: u64 = 0,
    elections_won: u64 = 0,
    heartbeats_sent: u64 = 0,
    entries_replicated: u64 = 0,
};

pub const LogEntry = struct {
    term: u64 = 0,
    index: u64 = 0,
    data: []const u8 = "",
};

pub const RequestVoteRequest = struct {
    term: u64 = 0,
    candidate_id: []const u8 = "",
    last_log_index: u64 = 0,
    last_log_term: u64 = 0,
};

pub const RequestVoteResponse = struct {
    term: u64 = 0,
    vote_granted: bool = false,
};

pub const AppendEntriesRequest = struct {
    term: u64 = 0,
    leader_id: []const u8 = "",
    prev_log_index: u64 = 0,
    prev_log_term: u64 = 0,
    entries: []const LogEntry = &.{},
    leader_commit: u64 = 0,
};

pub const AppendEntriesResponse = struct {
    term: u64 = 0,
    success: bool = false,
    match_index: u64 = 0,
    follower_id: []const u8 = "",
};

pub const PeerState = struct {
    node_id: []const u8 = "",
    next_index: u64 = 0,
    match_index: u64 = 0,
    last_contact_ms: u64 = 0,
    vote_granted: bool = false,
    pre_vote_granted: bool = false,
};

pub fn createCluster(_: std.mem.Allocator, _: []const []const u8) !void {
    return error.NetworkDisabled;
}

// Persistence
pub const RaftPersistence = struct {
    pub fn init(_: std.mem.Allocator, _: []const u8) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const PersistentState = struct {
    current_term: u64 = 0,
    voted_for: ?[]const u8 = null,
};

// Snapshots
pub const RaftSnapshotManager = struct {
    pub fn init(_: std.mem.Allocator, _: SnapshotConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const SnapshotConfig = struct {
    snapshot_interval: u64 = 10000,
    snapshot_dir: []const u8 = "snapshots",
};

pub const SnapshotMetadata = struct {
    last_included_index: u64 = 0,
    last_included_term: u64 = 0,
};

pub const SnapshotInfo = struct {
    metadata: SnapshotMetadata = .{},
    size: u64 = 0,
};

pub const InstallSnapshotRequest = struct {
    term: u64 = 0,
    leader_id: []const u8 = "",
    last_included_index: u64 = 0,
    last_included_term: u64 = 0,
    offset: u64 = 0,
    data: []const u8 = "",
    done: bool = false,
};

pub const InstallSnapshotResponse = struct {
    term: u64 = 0,
};

// Membership changes
pub const ConfigChangeType = enum { add_node, remove_node };

pub const ConfigChangeRequest = struct {
    change_type: ConfigChangeType = .add_node,
    node_id: []const u8 = "",
    address: []const u8 = "",
};

pub fn applyConfigChange(_: *RaftNode, _: ConfigChangeRequest) !void {
    return error.NetworkDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
