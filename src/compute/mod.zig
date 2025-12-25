//! Compute Runtime Module
//!
//! High-performance compute runtime with SIMD, memory management,
//! concurrency primitives, and workload execution.

const std = @import("std");

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
pub const memory = @import("memory/mod.zig");
pub const concurrency = @import("concurrency/mod.zig");
pub const runtime = @import("runtime/mod.zig");
pub const workloads = @import("workloads/mod.zig");

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
pub const BenchmarkResult = runtime.BenchmarkResult;
pub const runBenchmark = runtime.runBenchmark;
pub const printBenchmarkResults = runtime.printBenchmarkResults;
pub const MatrixMultBenchmark = runtime.MatrixMultBenchmark;
pub const MemoryAllocationBenchmark = runtime.MemoryAllocationBenchmark;
pub const FibonacciBenchmark = runtime.FibonacciBenchmark;
pub const HashingBenchmark = runtime.HashingBenchmark;

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

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
}

pub fn deinit() void {}
