//! Shared types for the federated learning module.
//!
//! Used by both mod.zig and stub.zig to prevent type drift between
//! enabled and disabled paths.
//!
//! Source of truth: mod.zig definitions.

const std = @import("std");

/// Information about a registered federated learning node.
pub const NodeInfo = struct {
    id: []const u8,
    last_update: i64,
};

/// Errors produced by the federated coordinator.
pub const CoordinatorError = error{
    InsufficientUpdates,
    InvalidUpdate,
};

/// Strategy for aggregating model updates from multiple nodes.
pub const AggregationStrategy = enum {
    mean,
    weighted_mean,
};

/// Read-only view of a model update submission (caller-owned slices).
pub const ModelUpdateView = struct {
    node_id: []const u8,
    step: u64,
    weights: []const f32,
    sample_count: u32 = 1,
};

/// Owned model update stored by the coordinator.
pub const ModelUpdate = struct {
    node_id: []const u8,
    step: u64,
    timestamp: u64,
    weights: []f32,
    sample_count: u32,

    pub fn deinit(self: *ModelUpdate, allocator: std.mem.Allocator) void {
        allocator.free(self.node_id);
        allocator.free(self.weights);
        self.* = undefined;
    }
};

/// Configuration for the federated coordinator.
pub const CoordinatorConfig = struct {
    min_updates: usize = 1,
    max_updates: usize = 64,
    max_staleness_seconds: u64 = 300,
    strategy: AggregationStrategy = .mean,
};

test {
    std.testing.refAllDecls(@This());
}
