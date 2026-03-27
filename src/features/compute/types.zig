//! Shared types for the compute feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

const std = @import("std");

/// Errors returned by compute operations.
pub const ComputeError = error{
    MeshUnavailable,
    NodeUnreachable,
    TaskFailed,
    OutOfMemory,
};

pub const Error = ComputeError;

/// Compute context for framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,
};
