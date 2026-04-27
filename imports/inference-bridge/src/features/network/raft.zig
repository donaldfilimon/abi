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
//! Sub-modules:
//! - `raft/types.zig` — Protocol types and configurations
//! - `raft/node.zig` — Raft node implementation
//! - `raft/fault_injector.zig` — Fault injection for testing
//! - `raft/cluster.zig` — Cluster management for testing
//! - `raft/io.zig` — IO backend helpers

const std = @import("std");

// Re-export persistence and snapshot types
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

// Re-exports from sub-modules
const types = @import("raft/types.zig");
pub const RaftState = types.RaftState;
pub const RaftError = types.RaftError;
pub const RaftConfig = types.RaftConfig;
pub const LogEntry = types.LogEntry;
pub const RequestVoteRequest = types.RequestVoteRequest;
pub const RequestVoteResponse = types.RequestVoteResponse;
pub const AppendEntriesRequest = types.AppendEntriesRequest;
pub const AppendEntriesResponse = types.AppendEntriesResponse;
pub const PeerState = types.PeerState;
pub const RaftStats = types.RaftStats;
pub const ApplyCallback = types.ApplyCallback;

pub const RaftNode = @import("raft/node.zig").RaftNode;
pub const FaultInjector = @import("raft/fault_injector.zig").FaultInjector;
pub const createCluster = @import("raft/cluster.zig").createCluster;
pub const initIoBackend = @import("raft/io.zig").initIoBackend;

// =============================================================================
// Tests
// =============================================================================

test {
    _ = @import("raft_persistence.zig");
    _ = @import("raft_snapshot.zig");
    _ = @import("raft_test.zig");
    _ = @import("raft/types.zig");
    _ = @import("raft/node.zig");
    _ = @import("raft/fault_injector.zig");
    _ = @import("raft/cluster.zig");
    _ = @import("raft/io.zig");
}

test {
    std.testing.refAllDecls(@This());
}
