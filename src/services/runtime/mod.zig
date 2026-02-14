//! Runtime Module - Always-on Core Infrastructure
//!
//! This module provides the foundational runtime infrastructure that is always
//! available regardless of which features are enabled. It includes:
//!
//! - Task scheduling and execution engine
//! - Concurrency primitives (futures, task groups, cancellation)
//! - Memory management utilities
//!
//! ## Module Organization
//!
//! ```
//! runtime/
//! ├── mod.zig          # This file - unified entry point
//! ├── engine/          # Task execution engine
//! ├── scheduling/      # Futures, cancellation, task groups
//! ├── concurrency/     # Lock-free data structures
//! └── memory/          # Memory pools and allocators
//! ```
//!
//! ## Usage
//!
//! ```zig
//! const runtime = @import("runtime/mod.zig");
//!
//! // Create runtime context
//! var ctx = try runtime.Context.init(allocator);
//! defer ctx.deinit();
//!
//! // Use task groups for parallel work
//! var group = try ctx.createTaskGroup(.{});
//! defer group.deinit();
//! ```

const std = @import("std");

// Submodules - organized by domain (local implementations)
pub const engine = @import("engine/mod.zig");
pub const scheduling = @import("scheduling/mod.zig");
pub const concurrency = @import("concurrency/mod.zig");
pub const memory = @import("memory/mod.zig");

// Workload types (local)
pub const workload = @import("workload.zig");

// ============================================================================
// Engine Types (re-exported for convenience)
// ============================================================================

pub const Engine = engine.Engine;
pub const DistributedComputeEngine = engine.DistributedComputeEngine;
pub const EngineConfig = engine.EngineConfig;
pub const EngineError = engine.EngineError;
pub const TaskId = engine.TaskId;

// Workload types
pub const ExecutionContext = workload.ExecutionContext;
pub const WorkloadHints = workload.WorkloadHints;
pub const WorkloadVTable = workload.WorkloadVTable;
pub const GPUWorkloadVTable = workload.GPUWorkloadVTable;
pub const ResultHandle = workload.ResultHandle;
pub const ResultVTable = workload.ResultVTable;
pub const WorkItem = workload.WorkItem;
pub const runWorkItem = workload.runWorkItem;

// Benchmarking
pub const BenchmarkResult = engine.BenchmarkResult;
pub const runBenchmarks = engine.runBenchmarks;

// ============================================================================
// Scheduling Types (re-exported for convenience)
// ============================================================================

// Future/Promise pattern
pub const Future = scheduling.Future;
pub const FutureState = scheduling.FutureState;
pub const FutureResult = scheduling.FutureResult;
pub const Promise = scheduling.Promise;
pub const all = scheduling.all;
pub const race = scheduling.race;
pub const delay = scheduling.delay;

// Cancellation
pub const CancellationToken = scheduling.CancellationToken;
pub const CancellationSource = scheduling.CancellationSource;
pub const CancellationState = scheduling.CancellationState;
pub const CancellationReason = scheduling.CancellationReason;
pub const LinkedCancellation = scheduling.LinkedCancellation;
pub const ScopedCancellation = scheduling.ScopedCancellation;

// Task groups
pub const TaskGroup = scheduling.TaskGroup;
pub const TaskGroupConfig = scheduling.TaskGroupConfig;
pub const TaskGroupBuilder = scheduling.TaskGroupBuilder;
pub const ScopedTaskGroup = scheduling.ScopedTaskGroup;
pub const TaskContext = scheduling.TaskContext;
pub const TaskFn = scheduling.TaskFn;
pub const TaskState = scheduling.TaskState;
pub const TaskResult = scheduling.TaskResult;
pub const TaskInfo = scheduling.TaskInfo;
pub const GroupStats = scheduling.GroupStats;
pub const parallelForEach = scheduling.parallelForEach;

// Async runtime
pub const AsyncRuntime = scheduling.AsyncRuntime;
pub const AsyncRuntimeOptions = scheduling.AsyncRuntimeOptions;
pub const TaskHandle = scheduling.TaskHandle;
pub const AsyncTaskGroup = scheduling.AsyncTaskGroup;
pub const AsyncError = scheduling.AsyncError;

// v2 work-stealing thread pool (use for parallel batch work, e.g. CPU inference)
pub const ThreadPool = scheduling.ThreadPool;
pub const ThreadPoolTask = scheduling.ThreadPoolTask;
pub const parallelFor = scheduling.parallelFor;

// v2 DAG pipeline scheduler
pub const DagPipeline = scheduling.Pipeline;
pub const DagPipelineResult = scheduling.PipelineResult;
pub const DagStage = scheduling.PipelineStage;
pub const DagStageStatus = scheduling.StageStatus;
pub const createInferencePipeline = scheduling.createInferencePipeline;

// ============================================================================
// Concurrency Types (re-exported for convenience)
// ============================================================================

pub const WorkStealingQueue = concurrency.WorkStealingQueue;
pub const WorkQueue = concurrency.WorkQueue;
pub const LockFreeQueue = concurrency.LockFreeQueue;
pub const LockFreeStack = concurrency.LockFreeStack;
pub const ShardedMap = concurrency.ShardedMap;
pub const PriorityQueue = concurrency.PriorityQueue;
pub const Backoff = concurrency.Backoff;

// New lock-free concurrency primitives
pub const ChaseLevDeque = concurrency.ChaseLevDeque;
pub const WorkStealingScheduler = concurrency.WorkStealingScheduler;
pub const EpochReclamation = concurrency.EpochReclamation;
pub const LockFreeStackEBR = concurrency.LockFreeStackEBR;
pub const MpmcQueue = concurrency.MpmcQueue;
pub const BlockingMpmcQueue = concurrency.BlockingMpmcQueue;

// v2 Vyukov MPMC channel
pub const Channel = concurrency.Channel;
pub const ByteChannel = concurrency.ByteChannel;
pub const MessageChannel = concurrency.MessageChannel;
pub const ChannelMessage = concurrency.Message;

// Result caching for fast-path task completion
pub const ResultCache = engine.ResultCache;
pub const CacheConfig = engine.CacheConfig;
pub const CacheStats = engine.CacheStats;
pub const Memoize = engine.Memoize;

// Work-stealing policies
pub const NumaStealPolicy = engine.NumaStealPolicy;
pub const RoundRobinStealPolicy = engine.RoundRobinStealPolicy;
pub const StealPolicyConfig = engine.StealPolicyConfig;
pub const StealStats = engine.StealStats;

// ============================================================================
// Memory Types (re-exported for convenience)
// ============================================================================

pub const MemoryPool = memory.MemoryPool;
pub const ArenaAllocator = memory.ArenaAllocator;

// ============================================================================
// Runtime Context
// ============================================================================

/// Runtime context - the always-available infrastructure.
/// This is created automatically by the Framework and provides
/// access to scheduling, concurrency, and memory primitives.
pub const Context = struct {
    allocator: std.mem.Allocator,
    engine_ptr: ?*Engine = null,
    initialized: bool = false,

    pub const Error = error{
        AlreadyInitialized,
        NotInitialized,
        EngineCreationFailed,
    } || std.mem.Allocator.Error;

    /// Initialize the runtime context.
    pub fn init(allocator: std.mem.Allocator) Error!*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .initialized = true,
        };
        return ctx;
    }

    /// Shutdown the runtime context.
    pub fn deinit(self: *Context) void {
        const allocator = self.allocator;
        if (self.engine_ptr) |e| {
            e.deinit();
            allocator.destroy(e);
        }
        allocator.destroy(self);
    }

    /// Get or create the compute engine.
    pub fn getEngine(self: *Context) Error!*Engine {
        if (self.engine_ptr) |e| return e;

        const engine_instance = try self.allocator.create(Engine);
        engine_instance.* = engine.createEngine(self.allocator) catch {
            self.allocator.destroy(engine_instance);
            return Error.EngineCreationFailed;
        };
        self.engine_ptr = engine_instance;
        return engine_instance;
    }

    /// Create a new task group.
    pub fn createTaskGroup(self: *Context, config: TaskGroupConfig) !TaskGroup {
        return TaskGroup.init(self.allocator, config);
    }

    /// Create a new future.
    pub fn createFuture(self: *Context, comptime T: type) !Future(T) {
        return Future(T).init(self.allocator);
    }

    /// Create a cancellation source.
    pub fn createCancellationSource(self: *Context) !CancellationSource {
        return CancellationSource.init(self.allocator);
    }
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create an engine with configuration (2-arg version for compatibility).
pub fn createEngine(allocator: std.mem.Allocator, config: EngineConfig) !Engine {
    return engine.createEngineWithConfig(allocator, config);
}

/// Create an engine with default configuration.
pub fn createDefaultEngine(allocator: std.mem.Allocator) !Engine {
    return engine.createEngine(allocator);
}

/// Create an engine with custom configuration.
pub fn createEngineWithConfig(allocator: std.mem.Allocator, config: EngineConfig) !Engine {
    return engine.createEngineWithConfig(allocator, config);
}

// ============================================================================
// Tests
// ============================================================================

test "Context initialization" {
    var ctx = try Context.init(std.testing.allocator);
    defer ctx.deinit();

    try std.testing.expect(ctx.initialized);
}

test "Context.getEngine creates engine lazily" {
    var ctx = try Context.init(std.testing.allocator);
    defer ctx.deinit();

    try std.testing.expect(ctx.engine_ptr == null);
    const engine1 = try ctx.getEngine();
    try std.testing.expect(ctx.engine_ptr != null);
    const engine2 = try ctx.getEngine();
    try std.testing.expect(engine1 == engine2);
}

test "submodules are accessible" {
    // Verify submodule exports compile
    _ = engine.EngineConfig;
    _ = scheduling.TaskGroup;
    _ = concurrency.WorkStealingQueue;
    _ = memory.MemoryPool;
}

test "new concurrency types accessible" {
    _ = ChaseLevDeque;
    _ = WorkStealingScheduler;
    _ = EpochReclamation;
    _ = LockFreeStackEBR;
    _ = MpmcQueue;
    _ = BlockingMpmcQueue;
}

test "result cache types accessible" {
    _ = ResultCache;
    _ = CacheConfig;
    _ = CacheStats;
    _ = Memoize;
}

test "steal policy types accessible" {
    _ = NumaStealPolicy;
    _ = RoundRobinStealPolicy;
    _ = StealPolicyConfig;
    _ = StealStats;
}
