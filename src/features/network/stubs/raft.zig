const std = @import("std");

pub const RaftNode = struct {
    allocator: std.mem.Allocator,
    node_id: []const u8,
    config: RaftConfig,
    current_term: u64 = 0,
    commit_index: u64 = 0,
    last_applied: u64 = 0,
    state: RaftState = .follower,
    leader_id: ?[]const u8 = null,
    peers: std.StringHashMapUnmanaged(PeerState) = .{},
    log: std.ArrayListUnmanaged(LogEntry) = .empty,

    pub fn init(allocator: std.mem.Allocator, node_id: []const u8, config: RaftConfig) !@This() {
        return .{
            .allocator = allocator,
            .node_id = node_id,
            .config = config,
        };
    }
    pub fn deinit(self: *@This()) void {
        for (self.log.items) |entry| {
            self.allocator.free(entry.data);
        }
        self.log.deinit(self.allocator);

        var peer_iter = self.peers.iterator();
        while (peer_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.peers.deinit(self.allocator);

        if (self.leader_id) |leader_id| {
            if (leader_id.ptr != self.node_id.ptr) {
                self.allocator.free(leader_id);
            }
        }

        self.* = undefined;
    }

    pub fn getStats(self: *const @This()) RaftStats {
        return .{
            .current_term = self.current_term,
            .state = self.state,
            .commit_index = self.commit_index,
            .last_applied = self.last_applied,
            .log_length = self.log.items.len,
            .peer_count = self.peers.count(),
            .leader_id = self.leader_id,
        };
    }

    pub fn getLeader(self: *const @This()) ?[]const u8 {
        return self.leader_id;
    }

    pub fn isLeader(self: *const @This()) bool {
        return self.state == .leader;
    }

    pub fn addPeer(self: *@This(), peer_id: []const u8) !void {
        if (self.peers.contains(peer_id)) return;

        const peer_copy = try self.allocator.dupe(u8, peer_id);
        errdefer self.allocator.free(peer_copy);

        try self.peers.put(self.allocator, peer_copy, .{
            .node_id = peer_copy,
            .next_index = @as(u64, @intCast(self.log.items.len + 1)),
        });
    }

    pub fn appendCommand(self: *@This(), data: []const u8) !u64 {
        const index: u64 = @intCast(self.log.items.len + 1);
        const data_copy = try self.allocator.dupe(u8, data);
        errdefer self.allocator.free(data_copy);

        try self.log.append(self.allocator, .{
            .term = self.current_term,
            .index = index,
            .data = data_copy,
            .entry_type = .command,
        });
        return index;
    }

    pub fn handleRequestVote(_: *@This(), _: RequestVoteRequest) !RequestVoteResponse {
        return error.NetworkDisabled;
    }

    pub fn handlePreVoteResponse(_: *@This(), _: RequestVoteResponse) !void {
        return error.NetworkDisabled;
    }

    pub fn buildPreVoteRequest(_: *const @This()) RequestVoteRequest {
        return .{};
    }

    pub fn buildRequestVoteRequest(_: *const @This()) RequestVoteRequest {
        return .{};
    }

    pub const ApplyCallback = *const fn (entry: LogEntry, user_data: ?*anyopaque) void;
};

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
    entry_type: EntryType = .command,

    pub const EntryType = enum {
        command,
        config_change,
        no_op,
    };
};

pub const RequestVoteRequest = struct {
    term: u64 = 0,
    candidate_id: []const u8 = "",
    last_log_index: u64 = 0,
    last_log_term: u64 = 0,
    is_pre_vote: bool = false,
};

pub const RequestVoteResponse = struct {
    term: u64 = 0,
    vote_granted: bool = false,
    voter_id: []const u8 = "",
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

pub fn createCluster(_: std.mem.Allocator, _: []const []const u8, _: RaftConfig) !void {
    return error.NetworkDisabled;
}

/// Fault injection stub (no-op when network is disabled).
pub const FaultInjector = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) FaultInjector {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *FaultInjector) void {}
    pub fn simulatePartition(_: *FaultInjector, _: []const u8, _: []const u8) !void {
        return error.NetworkDisabled;
    }
    pub fn simulateHeal(_: *FaultInjector, _: []const u8, _: []const u8) void {}
    pub fn isBlocked(_: *const FaultInjector, _: []const u8, _: []const u8) bool {
        return false;
    }
};

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
