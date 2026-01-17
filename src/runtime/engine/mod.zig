//! Task Execution Engine
//!
//! This module provides the core task execution engine with:
//!
//! - `Engine` - Work-stealing distributed compute engine
//! - `EngineConfig` - Engine configuration options
//! - NUMA-aware task scheduling
//! - Result handling and task lifecycle

const std = @import("std");

// Local imports (implementation files)
pub const engine_impl = @import("engine.zig");
pub const types = @import("types.zig");
pub const numa = @import("numa.zig");
pub const benchmark = @import("benchmark.zig");

// Core engine types
pub const Engine = engine_impl.DistributedComputeEngine;
pub const DistributedComputeEngine = engine_impl.DistributedComputeEngine;
pub const EngineConfig = types.EngineConfig;
pub const EngineError = types.EngineError;
pub const TaskId = types.TaskId;
pub const DEFAULT_MAX_TASKS = types.DEFAULT_MAX_TASKS;

// Task execution types
pub const TaskExecuteError = types.TaskExecuteError;
pub const TaskExecuteFn = types.TaskExecuteFn;
pub const TaskNode = types.TaskNode;

// Result handling
pub const ResultKind = types.ResultKind;
pub const ResultBlob = types.ResultBlob;
pub const encodeResult = types.encodeResult;
pub const callTask = types.callTask;

// Utilities
pub const Backoff = types.Backoff;
pub const nowMilliseconds = types.nowMilliseconds;
pub const isByteSlice = types.isByteSlice;

// NUMA support
pub const CpuTopology = numa.CpuTopology;
pub const NumaNode = numa.NumaNode;
pub const CpuInfo = numa.CpuInfo;

// Benchmarking
pub const BenchmarkResult = benchmark.BenchmarkResult;
pub const runBenchmarks = benchmark.runBenchmarks;

// Workload types (from engine_impl)
pub const ExecutionContext = engine_impl.ExecutionContext;
pub const WorkloadHints = engine_impl.WorkloadHints;
pub const WorkloadVTable = engine_impl.WorkloadVTable;
pub const GPUWorkloadVTable = engine_impl.GPUWorkloadVTable;
pub const ResultHandle = engine_impl.ResultHandle;
pub const ResultVTable = engine_impl.ResultVTable;
pub const WorkItem = engine_impl.WorkItem;
pub const runWorkItem = engine_impl.runWorkItem;

/// Create an engine with default configuration.
pub fn createEngine(allocator: std.mem.Allocator) !Engine {
    return Engine.init(allocator, .{});
}

/// Create an engine with custom configuration.
pub fn createEngineWithConfig(allocator: std.mem.Allocator, config: EngineConfig) !Engine {
    return Engine.init(allocator, config);
}

// ============================================================================
// Tests
// ============================================================================

test "EngineConfig defaults" {
    const config = EngineConfig{};
    try std.testing.expectEqual(@as(usize, DEFAULT_MAX_TASKS), config.max_tasks);
    try std.testing.expect(config.worker_count == null);
    try std.testing.expect(!config.numa_enabled);
}

test "Backoff utility" {
    var backoff = Backoff{};
    try std.testing.expectEqual(@as(usize, 0), backoff.spins);
    backoff.spin();
    try std.testing.expectEqual(@as(usize, 1), backoff.spins);
    backoff.reset();
    try std.testing.expectEqual(@as(usize, 0), backoff.spins);
}
