//! Scheduling Abstractions for Runtime
//!
//! This module provides async execution primitives including:
//!
//! - `Future` / `Promise` - Async results with chaining
//! - `CancellationToken` - Cooperative cancellation
//! - `TaskGroup` - Hierarchical task organization
//! - `AsyncRuntime` - High-level async task execution

const std = @import("std");

// Local imports (implementation files)
pub const future_mod = @import("future.zig");
pub const cancellation_mod = @import("cancellation.zig");
pub const task_group_mod = @import("task_group.zig");
pub const async_mod = @import("async.zig");
pub const thread_pool_mod = @import("thread_pool.zig");
pub const dag_pipeline_mod = @import("dag_pipeline.zig");

// Future/Promise pattern
pub const Future = future_mod.Future;
pub const FutureState = future_mod.FutureState;
pub const FutureResult = future_mod.FutureResult;
pub const Promise = future_mod.Promise;
pub const all = future_mod.all;
pub const race = future_mod.race;
pub const delay = future_mod.delay;

// Cancellation
pub const CancellationToken = cancellation_mod.CancellationToken;
pub const CancellationSource = cancellation_mod.CancellationSource;
pub const CancellationState = cancellation_mod.CancellationState;
pub const CancellationReason = cancellation_mod.CancellationReason;
pub const LinkedCancellation = cancellation_mod.LinkedCancellation;
pub const ScopedCancellation = cancellation_mod.ScopedCancellation;

// Task groups
pub const TaskGroup = task_group_mod.TaskGroup;
pub const TaskGroupConfig = task_group_mod.TaskGroupConfig;
pub const TaskGroupBuilder = task_group_mod.TaskGroupBuilder;
pub const ScopedTaskGroup = task_group_mod.ScopedTaskGroup;
pub const TaskContext = task_group_mod.TaskContext;
pub const TaskFn = task_group_mod.TaskFn;
pub const TaskState = task_group_mod.TaskState;
pub const TaskResult = task_group_mod.TaskResult;
pub const TaskInfo = task_group_mod.TaskInfo;
pub const GroupStats = task_group_mod.GroupStats;
pub const parallelForEach = task_group_mod.parallelForEach;

// Async runtime
pub const AsyncRuntime = async_mod.AsyncRuntime;
pub const AsyncRuntimeOptions = async_mod.AsyncRuntimeOptions;
pub const TaskHandle = async_mod.TaskHandle;
pub const AsyncTaskGroup = async_mod.TaskGroup;
pub const AsyncError = async_mod.AsyncError;

// Work-stealing thread pool (v2)
pub const ThreadPool = thread_pool_mod.ThreadPool;
pub const ThreadPoolTask = thread_pool_mod.Task;
pub const parallelFor = thread_pool_mod.parallelFor;

// DAG pipeline scheduler (v2)
pub const Pipeline = dag_pipeline_mod.Pipeline;
pub const PipelineResult = dag_pipeline_mod.PipelineResult;
pub const PipelineStage = dag_pipeline_mod.Stage;
pub const StageStatus = dag_pipeline_mod.StageStatus;
pub const createInferencePipeline = dag_pipeline_mod.createInferencePipeline;

// ============================================================================
// Tests
// ============================================================================

test "TaskGroup initialization" {
    var group = TaskGroup.init(std.testing.allocator, .{});
    defer group.deinit();

    const stats = group.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.failed);
}

test "CancellationSource basic" {
    var source = CancellationSource.init(std.testing.allocator);
    defer source.deinit();

    const token = source.getToken();
    try std.testing.expect(!token.isCancelled());
}

test {
    std.testing.refAllDecls(@This());
}
