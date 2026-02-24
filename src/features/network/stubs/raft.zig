const std = @import("std");

pub const RaftNode = struct {
    pub fn init(_: std.mem.Allocator, _: RaftConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const RaftState = enum { follower, candidate, leader };

pub const RaftConfig = struct {
    node_id: []const u8 = "",
    election_timeout_min_ms: u64 = 150,
    election_timeout_max_ms: u64 = 300,
    heartbeat_interval_ms: u64 = 50,
};

pub const RaftError = error{
    NetworkDisabled,
    NotLeader,
    ElectionFailed,
    LogInconsistency,
};

pub const RaftStats = struct {
    state: RaftState = .follower,
    term: u64 = 0,
    commit_index: u64 = 0,
    last_applied: u64 = 0,
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
};

pub const PeerState = struct {
    id: []const u8 = "",
    next_index: u64 = 0,
    match_index: u64 = 0,
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
