//! Multi-Region Replication Manager
//!
//! Handles data replication across multiple nodes and regions:
//! - Synchronous and asynchronous replication modes
//! - Automatic leader election
//! - Conflict resolution
//! - Lag monitoring

// Re-export all public types from sub-modules
pub const ReplicationConfig = @import("state.zig").ReplicationConfig;
pub const ReplicationMode = @import("state.zig").ReplicationMode;
pub const ReplicationState = @import("state.zig").ReplicationState;
pub const ReplicationEvent = @import("state.zig").ReplicationEvent;
pub const DisconnectReason = @import("state.zig").DisconnectReason;
pub const ConflictResolution = @import("state.zig").ConflictResolution;
pub const QueueEntry = @import("state.zig").QueueEntry;
pub const ReplicationQueue = @import("state.zig").ReplicationQueue;

pub const ReplicationManager = @import("membership.zig").ReplicationManager;

const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
