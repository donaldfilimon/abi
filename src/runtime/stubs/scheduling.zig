const std = @import("std");
const types = @import("types.zig");
const engine_mod = @import("engine.zig");

pub fn Future(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator) types.Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn get(_: *Self) types.Error!T {
            return error.RuntimeDisabled;
        }

        pub fn getState(_: *const Self) FutureState {
            return .pending;
        }

        pub fn cancel(_: *Self) void {}
    };
}

pub const FutureState = enum { pending, ready, cancelled, failed };
pub const FutureResult = struct {};

pub fn Promise(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator) types.Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn set(_: *Self, _: T) void {}

        pub fn setError(_: *Self, _: anyerror) void {}
    };
}

pub fn all(_: std.mem.Allocator, _: anytype) types.Error!void {
    return error.RuntimeDisabled;
}

pub fn race(_: std.mem.Allocator, _: anytype) types.Error!void {
    return error.RuntimeDisabled;
}

pub fn delay(_: u64) types.Error!void {
    return error.RuntimeDisabled;
}

// Cancellation
pub const CancellationToken = struct {
    pub fn isCancelled(_: *const CancellationToken) bool {
        return false;
    }

    pub fn getState(_: *const CancellationToken) CancellationState {
        return .none;
    }
};

pub const CancellationSource = struct {
    pub fn init(_: std.mem.Allocator) types.Error!CancellationSource {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *CancellationSource) void {}

    pub fn cancel(_: *CancellationSource) void {}

    pub fn token(_: *CancellationSource) CancellationToken {
        return .{};
    }
};

pub const CancellationState = enum { none, requested, acknowledged };
pub const CancellationReason = enum { user_requested, timeout, error_occurred };
pub const LinkedCancellation = struct {};
pub const ScopedCancellation = struct {};

// Task groups
pub const TaskGroup = struct {
    pub fn init(_: std.mem.Allocator, _: TaskGroupConfig) types.Error!TaskGroup {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *TaskGroup) void {}

    pub fn spawn(_: *TaskGroup, _: anytype) types.Error!void {
        return error.RuntimeDisabled;
    }

    pub fn wait(_: *TaskGroup) types.Error!void {
        return error.RuntimeDisabled;
    }

    pub fn getStats(_: *const TaskGroup) GroupStats {
        return .{};
    }
};

pub const TaskGroupConfig = struct {
    max_concurrent: ?usize = null,
    cancellation_token: ?CancellationToken = null,
};

pub const TaskGroupBuilder = struct {
    pub fn init(_: std.mem.Allocator) TaskGroupBuilder {
        return .{};
    }

    pub fn withMaxConcurrent(_: *TaskGroupBuilder, _: usize) *TaskGroupBuilder {
        return undefined;
    }

    pub fn build(_: *TaskGroupBuilder) types.Error!TaskGroup {
        return error.RuntimeDisabled;
    }
};

pub const ScopedTaskGroup = struct {};

pub const TaskContext = struct {
    allocator: std.mem.Allocator,
    cancellation_token: ?CancellationToken = null,
};

pub const TaskFn = *const fn (*TaskContext) anyerror!void;
pub const TaskState = enum { pending, running, completed, failed, cancelled };
pub const TaskResult = struct {};
pub const TaskInfo = struct {
    id: engine_mod.TaskId = 0,
    state: TaskState = .pending,
};
pub const GroupStats = struct {
    tasks_submitted: usize = 0,
    tasks_completed: usize = 0,
    tasks_failed: usize = 0,
};

pub fn parallelForEach(_: std.mem.Allocator, _: anytype, _: anytype) types.Error!void {
    return error.RuntimeDisabled;
}

// Async runtime
pub const AsyncRuntime = struct {
    pub fn init(_: std.mem.Allocator, _: AsyncRuntimeOptions) types.Error!AsyncRuntime {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *AsyncRuntime) void {}

    pub fn spawn(_: *AsyncRuntime, _: anytype) types.Error!TaskHandle {
        return error.RuntimeDisabled;
    }

    pub fn run(_: *AsyncRuntime) types.Error!void {
        return error.RuntimeDisabled;
    }
};

pub const AsyncRuntimeOptions = struct {
    thread_pool_size: ?usize = null,
};

pub const TaskHandle = struct {
    id: engine_mod.TaskId = 0,

    pub fn wait(_: *TaskHandle) types.Error!void {
        return error.RuntimeDisabled;
    }

    pub fn cancel(_: *TaskHandle) void {}
};

pub const AsyncTaskGroup = struct {};
pub const AsyncError = types.Error;
