const std = @import("std");
pub const engine = @import("engine.zig");
pub const async = @import("async.zig");
pub const benchmark = @import("benchmark.zig");
pub const workload = @import("workload.zig");

pub const DistributedComputeEngine = engine.DistributedComputeEngine;
pub const EngineConfig = engine.EngineConfig;
pub const EngineError = engine.EngineError;
pub const TaskId = engine.TaskId;
pub const BenchmarkResult = benchmark.BenchmarkResult;
pub const runBenchmarks = benchmark.runBenchmarks;
pub const ExecutionContext = workload.ExecutionContext;
pub const WorkloadHints = workload.WorkloadHints;
pub const WorkloadVTable = workload.WorkloadVTable;
pub const ResultHandle = workload.ResultHandle;
pub const ResultVTable = workload.ResultVTable;
pub const WorkItem = workload.WorkItem;
pub const GPUWorkloadVTable = workload.GPUWorkloadVTable;
pub const runWorkItem = workload.runWorkItem;
pub const matMul = workload.matMul;
pub const dense = workload.dense;
pub const relu = workload.relu;
pub const MatrixMultiplyTask = workload.MatrixMultiplyTask;
pub const MlpTask = workload.MlpTask;
pub const AsyncRuntime = async.AsyncRuntime;
pub const AsyncRuntimeOptions = async.AsyncRuntimeOptions;
pub const TaskHandle = async.TaskHandle;
pub const TaskGroup = async.TaskGroup;
pub const AsyncError = async.AsyncError;

pub fn createEngine(allocator: std.mem.Allocator, config: EngineConfig) !DistributedComputeEngine {
    return engine.DistributedComputeEngine.init(allocator, config);
}

pub fn submitTask(engine_instance: *DistributedComputeEngine, comptime ResultType: type, task: anytype) !TaskId {
    return engine_instance.submit_task(ResultType, task);
}

pub fn waitForResult(engine_instance: *DistributedComputeEngine, comptime ResultType: type, id: TaskId, timeout_ms: u64) !ResultType {
    return engine_instance.wait_for_result(ResultType, id, timeout_ms);
}

pub fn runTask(engine_instance: *DistributedComputeEngine, comptime ResultType: type, task: anytype, timeout_ms: u64) !ResultType {
    const id = try submitTask(engine_instance, ResultType, task);
    return waitForResult(engine_instance, ResultType, id, timeout_ms);
}

test "create engine helper" {
    var engine_instance = try createEngine(std.testing.allocator, .{ .max_tasks = 4 });
    defer engine_instance.deinit();
    try std.testing.expectEqual(@as(TaskId, 1), engine_instance.nextId());
}

test "run task helper" {
    var engine_instance = try createEngine(std.testing.allocator, .{ .max_tasks = 4 });
    defer engine_instance.deinit();

    const result = try runTask(&engine_instance, u32, sampleTask, 1000);
    try std.testing.expectEqual(@as(u32, 7), result);
}

fn sampleTask(_: std.mem.Allocator) !u32 {
    return 7;
}
