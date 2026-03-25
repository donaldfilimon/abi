//! Multi-Region Replication Manager
//!
//! Handles data replication across multiple nodes and regions:
//! - Synchronous and asynchronous replication modes
//! - Automatic leader election
//! - Conflict resolution
//! - Lag monitoring
//!
//! Implementation decomposed into replication/ subdirectory.

const mod = @import("replication/mod.zig");

pub const ReplicationConfig = mod.ReplicationConfig;
pub const ReplicationMode = mod.ReplicationMode;
pub const ReplicationState = mod.ReplicationState;
pub const ReplicationEvent = mod.ReplicationEvent;
pub const DisconnectReason = mod.DisconnectReason;
pub const ConflictResolution = mod.ConflictResolution;
pub const QueueEntry = mod.QueueEntry;
pub const ReplicationQueue = mod.ReplicationQueue;

pub const ReplicationManager = mod.ReplicationManager;

const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
