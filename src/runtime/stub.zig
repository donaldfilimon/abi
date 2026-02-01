//! Runtime Stub Module
//!
//! Provides stub implementations when runtime is disabled at compile time.
//! All operations return error.RuntimeDisabled.

const std = @import("std");

// ============================================================================
// Local Stubs Imports
// ============================================================================

pub const types = @import("stubs/types.zig");
pub const engine_mod = @import("stubs/engine.zig");
pub const scheduling_mod = @import("stubs/scheduling.zig");
pub const concurrency_mod = @import("stubs/concurrency.zig");
pub const memory_mod = @import("stubs/memory.zig");
pub const workload_mod = @import("stubs/workload.zig");

// ============================================================================
// Re-exports
// ============================================================================

pub const Error = types.Error;
pub const EngineError = types.EngineError;
pub const SchedulingError = types.SchedulingError;
pub const ConcurrencyError = types.ConcurrencyError;
pub const MemoryError = types.MemoryError;

// Engine Types
pub const Engine = engine_mod.Engine;
pub const DistributedComputeEngine = engine_mod.DistributedComputeEngine;
pub const EngineConfig = engine_mod.EngineConfig;
pub const EngineStats = engine_mod.EngineStats;
pub const TaskId = engine_mod.TaskId;
pub const ResultCache = engine_mod.ResultCache;
pub const CacheConfig = engine_mod.CacheConfig;
pub const CacheStats = engine_mod.CacheStats;
pub const Memoize = engine_mod.Memoize;
pub const NumaStealPolicy = engine_mod.NumaStealPolicy;
pub const RoundRobinStealPolicy = engine_mod.RoundRobinStealPolicy;
pub const StealPolicyConfig = engine_mod.StealPolicyConfig;
pub const StealStats = engine_mod.StealStats;
pub const BenchmarkResult = engine_mod.BenchmarkResult;
pub const runBenchmarks = engine_mod.runBenchmarks;

// Workload Types
pub const ExecutionContext = workload_mod.ExecutionContext;
pub const WorkloadHints = workload_mod.WorkloadHints;
pub const Priority = workload_mod.Priority;
pub const WorkloadVTable = workload_mod.WorkloadVTable;
pub const GPUWorkloadVTable = workload_mod.GPUWorkloadVTable;
pub const ResultHandle = workload_mod.ResultHandle;
pub const ResultVTable = workload_mod.ResultVTable;
pub const WorkItem = workload_mod.WorkItem;
pub const runWorkItem = workload_mod.runWorkItem;

// Scheduling Types
pub const Future = scheduling_mod.Future;
pub const FutureState = scheduling_mod.FutureState;
pub const FutureResult = scheduling_mod.FutureResult;
pub const Promise = scheduling_mod.Promise;
pub const all = scheduling_mod.all;
pub const race = scheduling_mod.race;
pub const delay = scheduling_mod.delay;
pub const CancellationToken = scheduling_mod.CancellationToken;
pub const CancellationSource = scheduling_mod.CancellationSource;
pub const CancellationState = scheduling_mod.CancellationState;
pub const CancellationReason = scheduling_mod.CancellationReason;
pub const LinkedCancellation = scheduling_mod.LinkedCancellation;
pub const ScopedCancellation = scheduling_mod.ScopedCancellation;
pub const TaskGroup = scheduling_mod.TaskGroup;
pub const TaskGroupConfig = scheduling_mod.TaskGroupConfig;
pub const TaskGroupBuilder = scheduling_mod.TaskGroupBuilder;
pub const ScopedTaskGroup = scheduling_mod.ScopedTaskGroup;
pub const TaskContext = scheduling_mod.TaskContext;
pub const TaskFn = scheduling_mod.TaskFn;
pub const TaskState = scheduling_mod.TaskState;
pub const TaskResult = scheduling_mod.TaskResult;
pub const TaskInfo = scheduling_mod.TaskInfo;
pub const GroupStats = scheduling_mod.GroupStats;
pub const parallelForEach = scheduling_mod.parallelForEach;
pub const AsyncRuntime = scheduling_mod.AsyncRuntime;
pub const AsyncRuntimeOptions = scheduling_mod.AsyncRuntimeOptions;
pub const TaskHandle = scheduling_mod.TaskHandle;
pub const AsyncTaskGroup = scheduling_mod.AsyncTaskGroup;
pub const AsyncError = scheduling_mod.AsyncError;

// Concurrency Types
pub const WorkStealingQueue = concurrency_mod.WorkStealingQueue;
pub const WorkQueue = concurrency_mod.WorkQueue;
pub const LockFreeQueue = concurrency_mod.LockFreeQueue;
pub const LockFreeStack = concurrency_mod.LockFreeStack;
pub const ShardedMap = concurrency_mod.ShardedMap;
pub const PriorityQueue = concurrency_mod.PriorityQueue;
pub const Backoff = concurrency_mod.Backoff;
pub const ChaseLevDeque = concurrency_mod.ChaseLevDeque;
pub const WorkStealingScheduler = concurrency_mod.WorkStealingScheduler;
pub const EpochReclamation = concurrency_mod.EpochReclamation;
pub const LockFreeStackEBR = concurrency_mod.LockFreeStackEBR;
pub const MpmcQueue = concurrency_mod.MpmcQueue;
pub const BlockingMpmcQueue = concurrency_mod.BlockingMpmcQueue;

// Memory Types
pub const MemoryPool = memory_mod.MemoryPool;
pub const MemoryPoolConfig = memory_mod.MemoryPoolConfig;
pub const MemoryPoolStats = memory_mod.MemoryPoolStats;
pub const ArenaAllocator = memory_mod.ArenaAllocator;

// ============================================================================
// Runtime Context
// ============================================================================

pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub const ContextError = error{
        RuntimeDisabled,
        AlreadyInitialized,
        NotInitialized,
        EngineCreationFailed,
    } || std.mem.Allocator.Error;

    pub fn init(_: std.mem.Allocator) ContextError!*Context {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn getEngine(_: *Context) ContextError!*Engine {
        return error.RuntimeDisabled;
    }

    pub fn createTaskGroup(_: *Context, _: TaskGroupConfig) Error!TaskGroup {
        return error.RuntimeDisabled;
    }

    pub fn createFuture(_: *Context, comptime T: type) Error!Future(T) {
        return error.RuntimeDisabled;
    }

    pub fn createCancellationSource(_: *Context) Error!CancellationSource {
        return error.RuntimeDisabled;
    }
};

// ============================================================================
// Submodule Stubs (for API compatibility with mod.zig)
// ============================================================================

pub const engine = struct {
    pub const Engine = engine_mod.Engine;
    pub const DistributedComputeEngine = engine_mod.DistributedComputeEngine;
    pub const EngineConfig = engine_mod.EngineConfig;
    pub const EngineError = types.EngineError;
    pub const TaskId = engine_mod.TaskId;
    pub const BenchmarkResult = engine_mod.BenchmarkResult;
    pub const ResultCache = engine_mod.ResultCache;
    pub const CacheConfig = engine_mod.CacheConfig;
    pub const CacheStats = engine_mod.CacheStats;
    pub const Memoize = engine_mod.Memoize;
    pub const NumaStealPolicy = engine_mod.NumaStealPolicy;
    pub const RoundRobinStealPolicy = engine_mod.RoundRobinStealPolicy;
    pub const StealPolicyConfig = engine_mod.StealPolicyConfig;
    pub const StealStats = engine_mod.StealStats;

    pub fn createEngine(allocator: std.mem.Allocator) Error!Engine {
        _ = allocator;
        return error.RuntimeDisabled;
    }

    pub fn createEngineWithConfig(allocator: std.mem.Allocator, config: EngineConfig) Error!Engine {
        _ = allocator;
        _ = config;
        return error.RuntimeDisabled;
    }

    pub fn runBenchmarks(allocator: std.mem.Allocator) Error![]BenchmarkResult {
        _ = allocator;
        return error.RuntimeDisabled;
    }
};

pub const scheduling = struct {
    pub const Future = scheduling_mod.Future;
    pub const FutureState = scheduling_mod.FutureState;
    pub const FutureResult = scheduling_mod.FutureResult;
    pub const Promise = scheduling_mod.Promise;
    pub const CancellationToken = scheduling_mod.CancellationToken;
    pub const CancellationSource = scheduling_mod.CancellationSource;
    pub const CancellationState = scheduling_mod.CancellationState;
    pub const CancellationReason = scheduling_mod.CancellationReason;
    pub const LinkedCancellation = scheduling_mod.LinkedCancellation;
    pub const ScopedCancellation = scheduling_mod.ScopedCancellation;
    pub const TaskGroup = scheduling_mod.TaskGroup;
    pub const TaskGroupConfig = scheduling_mod.TaskGroupConfig;
    pub const TaskGroupBuilder = scheduling_mod.TaskGroupBuilder;
    pub const ScopedTaskGroup = scheduling_mod.ScopedTaskGroup;
    pub const TaskContext = scheduling_mod.TaskContext;
    pub const TaskFn = scheduling_mod.TaskFn;
    pub const TaskState = scheduling_mod.TaskState;
    pub const TaskResult = scheduling_mod.TaskResult;
    pub const TaskInfo = scheduling_mod.TaskInfo;
    pub const GroupStats = scheduling_mod.GroupStats;
    pub const AsyncRuntime = scheduling_mod.AsyncRuntime;
    pub const AsyncRuntimeOptions = scheduling_mod.AsyncRuntimeOptions;
    pub const TaskHandle = scheduling_mod.TaskHandle;
    pub const AsyncTaskGroup = scheduling_mod.AsyncTaskGroup;
    pub const AsyncError = scheduling_mod.AsyncError;

    pub const all = scheduling_mod.all;
    pub const race = scheduling_mod.race;
    pub const delay = scheduling_mod.delay;
    pub const parallelForEach = scheduling_mod.parallelForEach;
};

pub const concurrency = struct {
    pub const WorkStealingQueue = concurrency_mod.WorkStealingQueue;
    pub const WorkQueue = concurrency_mod.WorkQueue;
    pub const LockFreeQueue = concurrency_mod.LockFreeQueue;
    pub const LockFreeStack = concurrency_mod.LockFreeStack;
    pub const ShardedMap = concurrency_mod.ShardedMap;
    pub const PriorityQueue = concurrency_mod.PriorityQueue;
    pub const Backoff = concurrency_mod.Backoff;
    pub const ChaseLevDeque = concurrency_mod.ChaseLevDeque;
    pub const WorkStealingScheduler = concurrency_mod.WorkStealingScheduler;
    pub const EpochReclamation = concurrency_mod.EpochReclamation;
    pub const LockFreeStackEBR = concurrency_mod.LockFreeStackEBR;
    pub const MpmcQueue = concurrency_mod.MpmcQueue;
    pub const BlockingMpmcQueue = concurrency_mod.BlockingMpmcQueue;
};

pub const memory = struct {
    pub const MemoryPool = memory_mod.MemoryPool;
    pub const ArenaAllocator = memory_mod.ArenaAllocator;
};

pub const workload = struct {
    pub const ExecutionContext = workload_mod.ExecutionContext;
    pub const WorkloadHints = workload_mod.WorkloadHints;
    pub const WorkloadVTable = workload_mod.WorkloadVTable;
    pub const GPUWorkloadVTable = workload_mod.GPUWorkloadVTable;
    pub const ResultHandle = workload_mod.ResultHandle;
    pub const ResultVTable = workload_mod.ResultVTable;
    pub const WorkItem = workload_mod.WorkItem;
    pub const runWorkItem = workload_mod.runWorkItem;
};

// ============================================================================
// Convenience Functions
// ============================================================================

pub fn createEngine(_: std.mem.Allocator, _: EngineConfig) Error!Engine {
    return error.RuntimeDisabled;
}

pub fn createDefaultEngine(_: std.mem.Allocator) Error!Engine {
    return error.RuntimeDisabled;
}

pub fn createEngineWithConfig(_: std.mem.Allocator, _: EngineConfig) Error!Engine {
    return error.RuntimeDisabled;
}

// ============================================================================
// Module-level functions
// ============================================================================

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(_: std.mem.Allocator) Error!void {
    return error.RuntimeDisabled;
}

pub fn deinit() void {}
