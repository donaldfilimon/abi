const std = @import("std");

pub const engine = @import("engine.zig");
pub const benchmark = @import("benchmark.zig");

pub const DistributedComputeEngine = engine.DistributedComputeEngine;
pub const EngineConfig = engine.EngineConfig;
pub const EngineError = engine.EngineError;
pub const TaskId = engine.TaskId;
pub const BenchmarkResult = benchmark.BenchmarkResult;
pub const runBenchmarks = benchmark.runBenchmarks;

pub fn createEngine(
    allocator: std.mem.Allocator,
    config: EngineConfig,
) !DistributedComputeEngine {
    return engine.DistributedComputeEngine.init(allocator, config);
}

pub fn submitTask(
    engine_instance: *DistributedComputeEngine,
    comptime ResultType: type,
    task: anytype,
) !TaskId {
    return engine_instance.submit_task(ResultType, task);
}

pub fn waitForResult(
    engine_instance: *DistributedComputeEngine,
    comptime ResultType: type,
    id: TaskId,
    timeout_ms: u64,
) !ResultType {
    return engine_instance.wait_for_result(ResultType, id, timeout_ms);
}

pub fn runTask(
    engine_instance: *DistributedComputeEngine,
    comptime ResultType: type,
    task: anytype,
    timeout_ms: u64,
) !ResultType {
    const id = try submitTask(engine_instance, ResultType, task);
    return waitForResult(engine_instance, ResultType, id, timeout_ms);
}

test "create engine helper" {
    var engine_instance = try createEngine(std.testing.allocator, .{ .max_tasks = 4 });
    defer engine_instance.deinit();
    try std.testing.expect(engine_instance.next_id == 1);
}

test "run task helper" {
    var engine_instance = try createEngine(std.testing.allocator, .{ .max_tasks = 4 });
    defer engine_instance.deinit();
    const result = try runTask(&engine_instance, u32, sampleTask, 0);
    try std.testing.expectEqual(@as(u32, 7), result);
}

fn sampleTask(_: std.mem.Allocator) !u32 {
    return 7;
}
