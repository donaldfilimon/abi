const std = @import("std");

<<<<<<< HEAD
pub const runtime = @import("runtime/mod.zig");
=======
const build_options = @import("build_options");

const network_module = if (build_options.enable_network)
    @import("network/mod.zig")
else
    @import("network/disabled.zig");

const profiling_module = if (build_options.enable_profiling)
    @import("profiling/mod.zig")
else
    @import("profiling/disabled.zig");

pub const simd = @import("simd/mod.zig");
>>>>>>> 4bc375bf29812920975da719c10a1b8ba159d095
pub const memory = @import("memory/mod.zig");
pub const concurrency = @import("concurrency/mod.zig");

<<<<<<< HEAD
pub const DistributedComputeEngine = runtime.DistributedComputeEngine;
pub const EngineConfig = runtime.EngineConfig;
pub const EngineError = runtime.EngineError;
pub const TaskId = runtime.TaskId;
=======
pub const gpu = if (build_options.enable_gpu)
    @import("gpu/mod.zig")
else
    struct {
        pub const GPUBackend = void;
        pub const GPUManager = void;
        pub const GPUWorkloadVTable = void;
        pub const GPUExecutionContext = void;
        pub const GPUWorkloadHints = void;
    };

pub const network = network_module;
pub const profiling = profiling_module;

pub const VectorOps = simd.VectorOps;
pub const ComputeVector = simd.ComputeVector;

pub const CacheAlignedBuffer = memory.CacheAlignedBuffer;
pub const PoolAllocator = memory.PoolAllocator;

pub const ChaseLevDeque = concurrency.ChaseLevDeque;
pub const InjectionQueue = concurrency.InjectionQueue;
pub const ShardedMap = concurrency.ShardedMap;

pub const Engine = runtime.Engine;
pub const WorkloadVTable = runtime.WorkloadVTable;
pub const WorkItem = runtime.WorkItem;
pub const ResultHandle = runtime.ResultHandle;
pub const ResultVTable = runtime.ResultVTable;
pub const ExecutionContext = runtime.ExecutionContext;
pub const WorkloadHints = runtime.WorkloadHints;
pub const DEFAULT_HINTS = runtime.DEFAULT_HINTS;
pub const EngineConfig = runtime.config.EngineConfig;
pub const DEFAULT_CONFIG = runtime.config.DEFAULT_CONFIG;
pub const MetricsCollector = profiling.MetricsCollector;
pub const MetricsConfig = profiling.MetricsConfig;
pub const MetricsSummary = profiling.MetricsSummary;
pub const DEFAULT_METRICS_CONFIG = profiling.DEFAULT_METRICS_CONFIG;
pub const config = runtime.config;

pub const Matrix = workloads.Matrix;
pub const MatrixMultiplication = workloads.MatrixMultiplication;
pub const NeuralInference = workloads.NeuralInference;

pub const ComputeBenchmark = runtime.ComputeBenchmark;
>>>>>>> 4bc375bf29812920975da719c10a1b8ba159d095
pub const BenchmarkResult = runtime.BenchmarkResult;
pub const runBenchmarks = runtime.runBenchmarks;

<<<<<<< HEAD
var initialized: bool = false;
=======
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
>>>>>>> 4bc375bf29812920975da719c10a1b8ba159d095

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
