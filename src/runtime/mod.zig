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
//! const runtime = @import("runtime");
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

/// Core task execution engine.
pub const Engine = engine.Engine;
/// Compute engine with distributed execution capabilities.
pub const DistributedComputeEngine = engine.DistributedComputeEngine;
/// Configuration for the task engine.
pub const EngineConfig = engine.EngineConfig;
/// Error set for compute engine operations.
pub const EngineError = engine.EngineError;
/// Unique identifier for a task.
pub const TaskId = engine.TaskId;

// Workload types
/// Context providing resources for workload execution.
pub const ExecutionContext = workload.ExecutionContext;
/// Optimization hints for the workload scheduler.
pub const WorkloadHints = workload.WorkloadHints;
/// Interface for custom compute workloads.
pub const WorkloadVTable = workload.WorkloadVTable;
/// Interface for GPU-accelerated workloads.
pub const GPUWorkloadVTable = workload.GPUWorkloadVTable;
/// Handle to an asynchronous execution result.
pub const ResultHandle = workload.ResultHandle;
/// Virtual table for result polling and retrieval.
pub const ResultVTable = workload.ResultVTable;
/// A single unit of work to be executed.
pub const WorkItem = workload.WorkItem;
/// Helper to execute a work item in a specific context.
pub const runWorkItem = workload.runWorkItem;

// Benchmarking
/// Results from a benchmark run.
pub const BenchmarkResult = engine.BenchmarkResult;
/// Execute the internal benchmark suite.
pub const runBenchmarks = engine.runBenchmarks;

// ============================================================================
// Scheduling Types (re-exported for convenience)
// ============================================================================

// Future/Promise pattern
/// A handle to a future value that may not be available yet.
pub const Future = scheduling.Future;
/// Current state of a future (pending, completed, etc.).
pub const FutureState = scheduling.FutureState;
/// Result of a future completion.
pub const FutureResult = scheduling.FutureResult;
/// The producer side of a future value.
pub const Promise = scheduling.Promise;
/// Wait for all futures in a list to complete.
pub const all = scheduling.all;
/// Wait for the first future in a list to complete.
pub const race = scheduling.race;
/// Create a future that completes after a delay.
pub const delay = scheduling.delay;

// Cancellation
/// Token used to signal and check for operation cancellation.
pub const CancellationToken = scheduling.CancellationToken;
/// Source that can trigger cancellation for associated tokens.
pub const CancellationSource = scheduling.CancellationSource;
/// State of a cancellation request.
pub const CancellationState = scheduling.CancellationState;
/// Reason for an operation being cancelled.
pub const CancellationReason = scheduling.CancellationReason;
/// Cancellation token that propagates from a parent source.
pub const LinkedCancellation = scheduling.LinkedCancellation;
/// Token that automatically cancels when it goes out of scope.
pub const ScopedCancellation = scheduling.ScopedCancellation;

// Task groups
/// Group of related tasks that can be managed together.
pub const TaskGroup = scheduling.TaskGroup;
/// Configuration for task group behavior.
pub const TaskGroupConfig = scheduling.TaskGroupConfig;
/// Builder for constructing task groups.
pub const TaskGroupBuilder = scheduling.TaskGroupBuilder;
/// Task group that automatically waits/cancels on scope exit.
pub const ScopedTaskGroup = scheduling.ScopedTaskGroup;
/// Execution context for a specific task.
pub const TaskContext = scheduling.TaskContext;
/// Function signature for tasks in a group.
pub const TaskFn = scheduling.TaskFn;
/// Lifecycle state of a task in a group.
pub const TaskState = scheduling.TaskState;
/// Outcome of a task execution.
pub const TaskResult = scheduling.TaskResult;
/// Metadata about a task in a group.
pub const TaskInfo = scheduling.TaskInfo;
/// Performance metrics for a task group.
pub const GroupStats = scheduling.GroupStats;
/// Execute a function in parallel for each element in a slice.
pub const parallelForEach = scheduling.parallelForEach;

// Async runtime
/// High-level asynchronous runtime for task orchestration.
pub const AsyncRuntime = scheduling.AsyncRuntime;
/// Options for configuring the async runtime.
pub const AsyncRuntimeOptions = scheduling.AsyncRuntimeOptions;
/// Handle to a task running in the async runtime.
pub const TaskHandle = scheduling.TaskHandle;
/// Group for managing tasks within the async runtime.
pub const AsyncTaskGroup = scheduling.AsyncTaskGroup;
/// Error set for async runtime operations.
pub const AsyncError = scheduling.AsyncError;

// v2 work-stealing thread pool (use for parallel batch work, e.g. CPU inference)
/// Work-stealing thread pool for high-performance batch processing.
pub const ThreadPool = scheduling.ThreadPool;
/// A single task to be executed by the thread pool.
pub const ThreadPoolTask = scheduling.ThreadPoolTask;
/// Execute a loop in parallel using the thread pool.
pub const parallelFor = scheduling.parallelFor;

// v2 DAG pipeline scheduler
/// Directed Acyclic Graph (DAG) pipeline for complex data flows.
pub const DagPipeline = scheduling.Pipeline;
/// Collective result from a pipeline execution.
pub const DagPipelineResult = scheduling.PipelineResult;
/// A single stage within a compute pipeline.
pub const DagStage = scheduling.PipelineStage;
/// Current execution status of a pipeline stage.
pub const DagStageStatus = scheduling.StageStatus;
/// Create a standard inference pipeline for LLM processing.
pub const createInferencePipeline = scheduling.createInferencePipeline;

// ============================================================================
// Concurrency Types (re-exported for convenience)
// ============================================================================

/// Queue optimized for work-stealing schedulers.
pub const WorkStealingQueue = concurrency.WorkStealingQueue;
/// Generic thread-safe work queue.
pub const WorkQueue = concurrency.WorkQueue;
/// Multi-producer, multi-consumer lock-free queue.
pub const LockFreeQueue = concurrency.LockFreeQueue;
/// Thread-safe lock-free stack.
pub const LockFreeStack = concurrency.LockFreeStack;
/// Map partitioned into shards to reduce contention.
pub const ShardedMap = concurrency.ShardedMap;
/// Thread-safe priority queue.
pub const PriorityQueue = concurrency.PriorityQueue;
/// Exponential backoff helper for spin-loops.
pub const Backoff = concurrency.Backoff;

// New lock-free concurrency primitives
/// Double-ended queue for work-stealing schedulers.
pub const ChaseLevDeque = concurrency.ChaseLevDeque;
/// Scheduler that uses work-stealing for load balancing.
pub const WorkStealingScheduler = concurrency.WorkStealingScheduler;
/// Epoch-based memory reclamation for lock-free structures.
pub const EpochReclamation = concurrency.EpochReclamation;
/// Lock-free stack using epoch-based reclamation.
pub const LockFreeStackEBR = concurrency.LockFreeStackEBR;
/// High-performance multi-producer, multi-consumer queue.
pub const MpmcQueue = concurrency.MpmcQueue;
/// MPMC queue that blocks when empty or full.
pub const BlockingMpmcQueue = concurrency.BlockingMpmcQueue;

// v2 Vyukov MPMC channel
/// Multi-producer, multi-consumer communication channel.
pub const Channel = concurrency.Channel;
/// Channel optimized for raw byte transfers.
pub const ByteChannel = concurrency.ByteChannel;
/// Channel for typed message passing.
pub const MessageChannel = concurrency.MessageChannel;
/// Standard message structure for channels.
pub const ChannelMessage = concurrency.Message;

// Result caching for fast-path task completion
/// Cache for storing and reusing compute results.
pub const ResultCache = engine.ResultCache;
/// Configuration for the result cache.
pub const CacheConfig = engine.CacheConfig;
/// Metrics for cache hits, misses, and evictions.
pub const CacheStats = engine.CacheStats;
/// Memoization helper for function results.
pub const Memoize = engine.Memoize;

// Work-stealing policies
/// Work-stealing policy that respects NUMA boundaries.
pub const NumaStealPolicy = engine.NumaStealPolicy;
/// Simple round-robin work-stealing policy.
pub const RoundRobinStealPolicy = engine.RoundRobinStealPolicy;
/// Configuration for stealing behavior.
pub const StealPolicyConfig = engine.StealPolicyConfig;
/// Metrics for work-stealing performance.
pub const StealStats = engine.StealStats;

// ============================================================================
// Memory Types (re-exported for convenience)
// ============================================================================

/// Thread-safe object pool for reducing allocation pressure.
pub const MemoryPool = memory.MemoryPool;
/// Arena allocator for fast bulk allocations.
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

test {
    std.testing.refAllDecls(@This());
}
