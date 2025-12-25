const std = @import("std");

pub const runtime = @import("runtime/mod.zig");
pub const memory = @import("memory/mod.zig");
pub const concurrency = @import("concurrency/mod.zig");

pub const DistributedComputeEngine = runtime.DistributedComputeEngine;
pub const EngineConfig = runtime.EngineConfig;
pub const EngineError = runtime.EngineError;
pub const TaskId = runtime.TaskId;
pub const BenchmarkResult = runtime.BenchmarkResult;
pub const runBenchmarks = runtime.runBenchmarks;

var initialized: bool = false;

pub fn createEngine(allocator: std.mem.Allocator, config: EngineConfig) !DistributedComputeEngine {
    return runtime.createEngine(allocator, config);
}

pub fn createDefaultEngine(allocator: std.mem.Allocator) !DistributedComputeEngine {
    return runtime.createEngine(allocator, .{});
}

pub fn submitTask(
    engine_instance: *DistributedComputeEngine,
    comptime ResultType: type,
    task: anytype,
) !TaskId {
    return runtime.submitTask(engine_instance, ResultType, task);
}

pub fn waitForResult(
    engine_instance: *DistributedComputeEngine,
    comptime ResultType: type,
    id: TaskId,
    timeout_ms: u64,
) !ResultType {
    return runtime.waitForResult(engine_instance, ResultType, id, timeout_ms);
}

pub fn runTask(
    engine_instance: *DistributedComputeEngine,
    comptime ResultType: type,
    task: anytype,
    timeout_ms: u64,
) !ResultType {
    return runtime.runTask(engine_instance, ResultType, task, timeout_ms);
}

pub fn init(_: std.mem.Allocator) !void {
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isInitialized() bool {
    return initialized;
}

test "compute init toggles state" {
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}

test "create default engine" {
    var engine_instance = try createDefaultEngine(std.testing.allocator);
    defer engine_instance.deinit();
    try std.testing.expect(engine_instance.config.max_tasks > 0);
}
