//! Type Definitions and Utilities for the Distributed Compute Engine
//!
//! This module provides core types, error definitions, and utility functions
//! used throughout the distributed compute engine.
//!
//! ## Error Types
//!
//! Two main error sets are provided:
//!
//! - `EngineError`: High-level errors returned to users (timeout, queue full, etc.)
//! - `TaskExecuteError`: Errors that can occur during task execution
//!
//! ## Configuration
//!
//! `EngineConfig` controls engine behavior:
//!
//! | Field | Default | Description |
//! |-------|---------|-------------|
//! | `max_tasks` | 1024 | Maximum in-flight tasks |
//! | `worker_count` | null (auto) | Number of worker threads |
//! | `numa_enabled` | false | Enable NUMA-aware scheduling |
//! | `cpu_affinity_enabled` | false | Pin workers to specific CPUs |
//!
//! ## Result Handling
//!
//! Results are serialized into `ResultBlob` for storage:
//!
//! - `ResultKind.value`: Fixed-size types (ints, floats, structs)
//! - `ResultKind.owned_slice`: Variable-length byte slices
//! - `ResultKind.task_error`: Task execution failed with error
//!
//! The `encodeResult` and type-checking utilities handle serialization.
//!
//! ## Backoff Strategy
//!
//! The `Backoff` struct implements exponential backoff for spin-wait loops:
//!
//! ```zig
//! var backoff = Backoff{};
//! while (condition) {
//!     // Do work...
//!     if (failed) {
//!         backoff.spin(); // Exponential backoff
//!     } else {
//!         backoff.reset();
//!     }
//! }
//! ```
//!
//! Backoff phases:
//! 1. First 16 spins: CPU spin-loop hint (pause instruction)
//! 2. After 16 spins: Yield to OS scheduler

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");

/// Backoff strategy for spin-wait loops in the engine.
pub const Backoff = struct {
    spins: usize = 0,

    pub fn reset(self: *Backoff) void {
        self.spins = 0;
    }

    pub fn spin(self: *Backoff) void {
        self.spins += 1;
        if (self.spins <= 16) {
            std.atomic.spinLoopHint();
            return;
        }
        // Thread yield failure is non-critical; log at debug level and continue
        std.Thread.yield() catch |err| {
            std.log.debug("Thread yield failed during engine backoff spin: {t}", .{err});
        };
    }

    pub fn wait(self: *Backoff) void {
        self.spins += 1;
        const iterations = @min(self.spins, 64);
        var i: usize = 0;
        while (i < iterations) {
            std.atomic.spinLoopHint();
            i += 1;
        }
        if (self.spins > 32) {
            // Thread yield failure is non-critical; log at debug level and continue
            std.Thread.yield() catch |err| {
                std.log.debug("Thread yield failed during engine backoff wait: {t}", .{err});
            };
        }
    }
};

var global_timer: ?time.Timer = null;
var timer_init_mutex = sync.Mutex{};

/// Get current time in milliseconds using a global timer.
/// Thread-safe: uses mutex for one-time initialization.
pub fn nowMilliseconds() i64 {
    // Double-checked locking pattern for thread-safe initialization
    if (global_timer == null) {
        timer_init_mutex.lock();
        defer timer_init_mutex.unlock();
        // Re-check after acquiring lock
        if (global_timer == null) {
            global_timer = time.Timer.start() catch null;
        }
    }
    if (global_timer) |*timer| {
        const ns = timer.read();
        const ms = @as(f64, @floatFromInt(ns)) / @as(f64, std.time.ns_per_ms);
        return @as(i64, @intFromFloat(ms));
    }
    return 0;
}

pub const EngineError = error{
    ResultNotFound,
    Timeout,
    UnsupportedResultType,
    QueueFull,
    TaskFailed,
};

/// Error set for task execution functions.
/// Combines common allocation errors with task-specific failures.
pub const TaskExecuteError = error{
    OutOfMemory,
    ExecutionFailed,
    Cancelled,
    InvalidInput,
    BufferMismatch,
    GpuUnavailable,
    Timeout,
};

/// Function pointer type for task execution.
pub const TaskExecuteFn = *const fn (std.mem.Allocator, *anyopaque) TaskExecuteError!ResultBlob;

pub const TaskId = u64;

pub const DEFAULT_MAX_TASKS: usize = 1024;

pub const EngineConfig = struct {
    max_tasks: usize = DEFAULT_MAX_TASKS,
    worker_count: ?usize = null,
    numa_enabled: bool = false,
    cpu_affinity_enabled: bool = false,
    numa_topology: ?*@import("numa.zig").CpuTopology = null,
};

pub const ResultKind = enum {
    value,
    owned_slice,
    task_error,
};

pub const ResultBlob = struct {
    kind: ResultKind,
    bytes: []u8,
    size: usize,
    error_code: u16 = 0,
};

pub const TaskNode = struct {
    id: TaskId,
    execute: TaskExecuteFn,
    destroy: *const fn (std.mem.Allocator, *anyopaque) void,
    payload: *anyopaque,
};

/// Check if a type is a byte slice ([]u8 or []const u8).
pub fn isByteSlice(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .pointer => |pointer| pointer.size == .slice and pointer.child == u8,
        else => false,
    };
}

/// Encode a result value into a ResultBlob.
pub fn encodeResult(
    allocator: std.mem.Allocator,
    comptime ResultType: type,
    result: ResultType,
) !ResultBlob {
    if (comptime isByteSlice(ResultType)) {
        const slice: []const u8 = result;
        const copy = try allocator.dupe(u8, slice);
        return .{
            .kind = .owned_slice,
            .bytes = copy,
            .size = copy.len,
        };
    }

    const size = @sizeOf(ResultType);
    const copy = try allocator.alloc(u8, size);
    std.mem.copyForwards(u8, copy, std.mem.asBytes(&result));
    return .{
        .kind = .value,
        .bytes = copy,
        .size = size,
    };
}

/// Call a task and return its result.
pub fn callTask(comptime ResultType: type, task: anytype, allocator: std.mem.Allocator) !ResultType {
    const TaskType = @TypeOf(task);
    switch (@typeInfo(TaskType)) {
        .@"fn" => return task(allocator),
        .pointer => |pointer| {
            if (@typeInfo(pointer.child) == .@"fn") {
                return task.*(allocator);
            }
        },
        else => {},
    }

    if (@hasDecl(TaskType, "execute")) {
        return task.execute(allocator);
    }

    @compileError("Task must be a function or type with execute(allocator)");
}

test {
    std.testing.refAllDecls(@This());
}
