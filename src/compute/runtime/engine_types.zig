//! Type definitions and utilities for the distributed compute engine.
//!
//! Contains error types, configuration, result handling types,
//! and utility structs like Backoff.

const std = @import("std");

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

var global_timer: ?std.time.Timer = null;

/// Get current time in milliseconds using a global timer.
pub fn nowMilliseconds() i64 {
    if (global_timer == null) {
        global_timer = std.time.Timer.start() catch null;
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
