//! Compute module providing high-performance runtime, memory management, and concurrency.
//!
//! This module is the core execution engine for ABI, offering:
//! - Work-stealing scheduler with worker thread pool
//! - Lock-free data structures for concurrent operations
//! - Memory management with stable allocators and worker arenas
//! - GPU acceleration support with CPU fallback
//! - Result caching with metadata
//!
//! Usage:
//! ```zig
//! var engine = try abi.compute.createDefaultEngine(allocator);
//! defer engine.deinit();
//!
//! const result = try abi.compute.runTask(&engine, u32, myTask, 1000);
//! ```
const std = @import("std");
const build_options = @import("build_options");

pub const runtime = @import("runtime/mod.zig");
pub const memory = @import("memory/mod.zig");
pub const concurrency = @import("concurrency/mod.zig");
pub const gpu = @import("gpu/mod.zig");
pub const simd = @import("../shared/simd.zig");

const network_module = if (build_options.enable_network)
    @import("network/mod.zig")
else
    @import("network/stub.zig");

const profiling_module = if (build_options.enable_profiling)
    @import("profiling/mod.zig")
else
    @import("profiling/stub.zig");

pub const network = network_module;
pub const profiling = profiling_module;

pub const DistributedComputeEngine = runtime.DistributedComputeEngine;
pub const EngineConfig = runtime.EngineConfig;
pub const EngineError = runtime.EngineError;
pub const TaskId = runtime.TaskId;
pub const BenchmarkResult = runtime.BenchmarkResult;
pub const runBenchmarks = runtime.runBenchmarks;

pub const ExecutionContext = runtime.ExecutionContext;
pub const WorkloadHints = runtime.WorkloadHints;
pub const WorkloadVTable = runtime.WorkloadVTable;
pub const GPUWorkloadVTable = runtime.GPUWorkloadVTable;
pub const ResultHandle = runtime.ResultHandle;
pub const ResultVTable = runtime.ResultVTable;
pub const WorkItem = runtime.WorkItem;
pub const runWorkItem = runtime.runWorkItem;
pub const matMul = runtime.matMul;
pub const dense = runtime.dense;
pub const relu = runtime.relu;
pub const MatrixMultiplyTask = runtime.MatrixMultiplyTask;
pub const MlpTask = runtime.MlpTask;
pub const AsyncRuntime = runtime.AsyncRuntime;
pub const AsyncRuntimeOptions = runtime.AsyncRuntimeOptions;
pub const TaskHandle = runtime.TaskHandle;
pub const TaskGroup = runtime.TaskGroup;
pub const AsyncError = runtime.AsyncError;

// SIMD functions available through abi.simd

pub const NetworkEngine = network.NetworkEngine;
pub const NetworkConfig = network.NetworkConfig;
pub const NodeRegistry = network.NodeRegistry;
pub const NodeInfo = network.NodeInfo;
pub const TaskMessage = network.TaskMessage;
pub const ResultMessage = network.ResultMessage;
pub const serializeTask = network.serializeTask;
pub const deserializeTask = network.deserializeTask;
pub const serializeResult = network.serializeResult;
pub const deserializeResult = network.deserializeResult;
pub const DEFAULT_NETWORK_CONFIG = network.DEFAULT_NETWORK_CONFIG;
pub const SerializationFormat = network.SerializationFormat;

pub const MetricsCollector = profiling.MetricsCollector;
pub const MetricsConfig = profiling.MetricsConfig;
pub const MetricsSummary = profiling.MetricsSummary;
pub const DEFAULT_METRICS_CONFIG = profiling.DEFAULT_METRICS_CONFIG;

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

/// Alias for runTask() - runs a workload and waits for the result
pub fn runWorkload(
    engine_instance: *DistributedComputeEngine,
    comptime ResultType: type,
    workload: anytype,
    timeout_ms: u64,
) !ResultType {
    return runtime.runWorkload(engine_instance, ResultType, workload, timeout_ms);
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
    try std.testing.expect(engine_instance.config().max_tasks > 0);
}
