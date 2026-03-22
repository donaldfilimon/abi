//! Task Execution Engine
//!
//! This module provides the core task execution engine with:
//!
//! - `Engine` - Work-stealing distributed compute engine
//! - `EngineConfig` - Engine configuration options
//! - NUMA-aware task scheduling
//! - Result handling and task lifecycle
//! - `ResultCache` - High-performance result caching for fast-path completion
//!
//! Note: On WASM/freestanding targets without thread support, the engine
//! provides stub implementations that return appropriate errors.

const std = @import("std");
const builtin = @import("builtin");

// Platform detection - threads not available on WASM/freestanding
const has_threads = !(@import("builtin").cpu.arch == .wasm32 or
    @import("builtin").cpu.arch == .wasm64 or
    @import("builtin").os.tag == .freestanding);

// Local imports (implementation files) - conditionally compiled
pub const engine_impl = if (has_threads) @import("engine.zig") else @import("engine_stub.zig");
pub const types = @import("types.zig");
pub const numa = if (has_threads) @import("numa.zig") else @import("numa_stub.zig");
pub const benchmark = if (has_threads) @import("benchmark.zig") else @import("benchmark_stub.zig");
pub const result_cache = @import("result_cache.zig");
pub const steal_policy = if (has_threads) @import("steal_policy.zig") else struct {
    // Stub for platforms without thread support
    pub const NumaStealPolicy = struct {
        pub fn init(_: anytype, _: anytype, _: anytype, _: anytype) !@This() {
            return error.ThreadsNotSupported;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const RoundRobinStealPolicy = struct {
        pub fn init(_: anytype, _: anytype) !@This() {
            return error.ThreadsNotSupported;
        }
        pub fn deinit(_: *@This()) void {}
    };
};

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

// Benchmarking
pub const BenchmarkResult = benchmark.BenchmarkResult;
pub const runBenchmarks = benchmark.runBenchmarks;

// Result caching
pub const ResultCache = result_cache.ResultCache;
pub const CacheConfig = result_cache.CacheConfig;
pub const CacheStats = result_cache.CacheStats;
pub const Memoize = result_cache.Memoize;

// Work-stealing policies
pub const NumaStealPolicy = steal_policy.NumaStealPolicy;
pub const RoundRobinStealPolicy = steal_policy.RoundRobinStealPolicy;
pub const StealPolicyConfig = if (has_threads) steal_policy.StealPolicyConfig else struct {};
pub const StealStats = if (has_threads) steal_policy.StealStats else struct {};

// Workload types — defined in engine_stub.zig only; provide stubs when threaded.
const stub_types = @import("engine_stub.zig");
pub const ExecutionContext = stub_types.ExecutionContext;
pub const WorkloadHints = stub_types.WorkloadHints;
pub const WorkloadVTable = stub_types.WorkloadVTable;
pub const GPUWorkloadVTable = stub_types.GPUWorkloadVTable;
pub const ResultHandle = stub_types.ResultHandle;
pub const ResultVTable = stub_types.ResultVTable;
pub const WorkItem = stub_types.WorkItem;
pub const runWorkItem = stub_types.runWorkItem;

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
