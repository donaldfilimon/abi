//! Engine Stub for WASM/Freestanding Targets
//!
//! Provides stub implementations when threading is not available.
//! All operations return ThreadsUnavailable error.

const std = @import("std");
const types = @import("types.zig");

pub const EngineError = error{
    ThreadsUnavailable,
    NotSupported,
};

/// Stub execution context
pub const ExecutionContext = struct {
    allocator: std.mem.Allocator,
};

/// Stub workload hints
pub const WorkloadHints = struct {
    priority: u8 = 0,
};

/// Stub workload vtable
pub const WorkloadVTable = struct {
    execute: *const fn (*anyopaque, *ExecutionContext) anyerror!void,
};

/// Stub GPU workload vtable
pub const GPUWorkloadVTable = struct {
    execute: *const fn (*anyopaque) anyerror!void,
};

/// Stub result handle
pub const ResultHandle = struct {
    id: u64 = 0,
};

/// Stub result vtable
pub const ResultVTable = struct {
    get: *const fn (*anyopaque) ?[]const u8,
};

/// Stub work item
pub const WorkItem = struct {
    id: u64 = 0,
};

/// Stub distributed compute engine - no threading support
pub const DistributedComputeEngine = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: types.EngineConfig) EngineError!DistributedComputeEngine {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *DistributedComputeEngine) void {
        _ = self;
    }

    pub fn submitTask(self: *DistributedComputeEngine, task_fn: anytype, args: anytype) EngineError!types.TaskId {
        _ = self;
        _ = task_fn;
        _ = args;
        return EngineError.ThreadsUnavailable;
    }

    pub fn waitForResult(self: *DistributedComputeEngine, task_id: types.TaskId) EngineError![]const u8 {
        _ = self;
        _ = task_id;
        return EngineError.ThreadsUnavailable;
    }

    pub fn getStats(self: *const DistributedComputeEngine) Stats {
        _ = self;
        return .{};
    }

    pub const Stats = struct {
        tasks_submitted: usize = 0,
        tasks_completed: usize = 0,
        tasks_failed: usize = 0,
    };
};

/// Stub run work item
pub fn runWorkItem(item: *WorkItem) void {
    _ = item;
}

test {
    std.testing.refAllDecls(@This());
}
