//! Distributed Node Interface for HNSW Graph
//!
//! Provides the abstraction layer for nodes that may reside on different
//! physical storage nodes (partitions) in a distributed HNSW deployment.

const std = @import("std");

/// Uniquely identifies a node within the global HNSW graph.
/// Nodes are partitioned by range or hash of their ID.
pub const NodeDescriptor = struct {
    id: u32,
    partition_id: u32,
};

/// Handle to a remote node's neighbors at a specific layer.
pub const RemoteNeighborHandle = struct {
    descriptor: NodeDescriptor,
    layer: usize,
};

/// Atomic primitive for coordinating neighbor list updates across partitions.
pub const SyncBarrier = struct {
    active_insertions: std.atomic.Value(usize) = .init(0),

    pub fn acquire(self: *SyncBarrier) void {
        _ = self.active_insertions.fetchAdd(1, .acquire);
    }

    pub fn release(self: *SyncBarrier) void {
        _ = self.active_insertions.fetchSub(1, .release);
    }

    pub fn wait(self: *SyncBarrier) void {
        while (self.active_insertions.load(.acquire) > 0) {
            std.atomic.spinLoopHint();
        }
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "SyncBarrier tracks active insertions" {
    var barrier = SyncBarrier{};

    try std.testing.expectEqual(@as(usize, 0), barrier.active_insertions.load(.acquire));

    barrier.acquire();
    try std.testing.expectEqual(@as(usize, 1), barrier.active_insertions.load(.acquire));

    barrier.acquire();
    try std.testing.expectEqual(@as(usize, 2), barrier.active_insertions.load(.acquire));

    barrier.release();
    try std.testing.expectEqual(@as(usize, 1), barrier.active_insertions.load(.acquire));

    barrier.release();
    barrier.wait();
    try std.testing.expectEqual(@as(usize, 0), barrier.active_insertions.load(.acquire));
}
