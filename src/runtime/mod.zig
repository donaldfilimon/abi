//! Runtime Module - Always-on Core Infrastructure
//!
//! This module provides the foundational runtime infrastructure that is always
//! available regardless of which features are enabled. It includes:
//!
//! - Task scheduling and execution engine
//! - Concurrency primitives (futures, task groups, cancellation)
//! - Memory management utilities
//! - SIMD acceleration
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
//!
//! try group.spawn(myTask, .{arg1, arg2});
//! try group.awaitAll();
//! ```

const std = @import("std");

// Re-export from compute/runtime for now (gradual migration)
const compute_runtime = @import("../compute/runtime/mod.zig");
const compute_concurrency = @import("../compute/concurrency/mod.zig");
const compute_memory = @import("../compute/memory/mod.zig");

// Core engine types
pub const Engine = compute_runtime.DistributedComputeEngine;
pub const EngineConfig = compute_runtime.EngineConfig;
pub const EngineError = compute_runtime.EngineError;
pub const TaskId = compute_runtime.TaskId;

// Future/Promise pattern
pub const Future = compute_runtime.Future;
pub const FutureState = compute_runtime.FutureState;
pub const FutureResult = compute_runtime.FutureResult;
pub const Promise = compute_runtime.Promise;
pub const all = compute_runtime.all;
pub const race = compute_runtime.race;
pub const delay = compute_runtime.delay;

// Cancellation
pub const CancellationToken = compute_runtime.CancellationToken;
pub const CancellationSource = compute_runtime.CancellationSource;
pub const CancellationState = compute_runtime.CancellationState;
pub const CancellationReason = compute_runtime.CancellationReason;
pub const LinkedCancellation = compute_runtime.LinkedCancellation;
pub const ScopedCancellation = compute_runtime.ScopedCancellation;

// Task groups
pub const TaskGroup = compute_runtime.TaskGroup;
pub const TaskGroupConfig = compute_runtime.TaskGroupConfig;
pub const TaskGroupBuilder = compute_runtime.TaskGroupBuilder;
pub const ScopedTaskGroup = compute_runtime.ScopedTaskGroup;
pub const TaskContext = compute_runtime.TaskContext;
pub const TaskFn = compute_runtime.TaskFn;
pub const TaskState = compute_runtime.TaskState;
pub const TaskResult = compute_runtime.TaskResult;
pub const TaskInfo = compute_runtime.TaskInfo;
pub const GroupStats = compute_runtime.GroupStats;
pub const parallelForEach = compute_runtime.parallelForEach;

// Async runtime
pub const AsyncRuntime = compute_runtime.AsyncRuntime;
pub const AsyncRuntimeOptions = compute_runtime.AsyncRuntimeOptions;
pub const TaskHandle = compute_runtime.TaskHandle;
pub const AsyncTaskGroup = compute_runtime.AsyncTaskGroup;
pub const AsyncError = compute_runtime.AsyncError;

// Workload types
pub const ExecutionContext = compute_runtime.ExecutionContext;
pub const WorkloadHints = compute_runtime.WorkloadHints;
pub const WorkloadVTable = compute_runtime.WorkloadVTable;
pub const GPUWorkloadVTable = compute_runtime.GPUWorkloadVTable;
pub const ResultHandle = compute_runtime.ResultHandle;
pub const ResultVTable = compute_runtime.ResultVTable;
pub const WorkItem = compute_runtime.WorkItem;
pub const runWorkItem = compute_runtime.runWorkItem;

// Benchmarking
pub const BenchmarkResult = compute_runtime.BenchmarkResult;
pub const runBenchmarks = compute_runtime.runBenchmarks;

// Concurrency primitives
pub const WorkStealingQueue = compute_concurrency.WorkStealingQueue;
pub const LockFreeQueue = compute_concurrency.LockFreeQueue;
pub const LockFreeStack = compute_concurrency.LockFreeStack;
pub const ShardedMap = compute_concurrency.ShardedMap;

// Memory management
pub const MemoryPool = compute_memory.MemoryPool;
pub const ArenaAllocator = compute_memory.ArenaAllocator;

/// Runtime context - the always-available infrastructure.
/// This is created automatically by the Framework and provides
/// access to scheduling, concurrency, and memory primitives.
pub const Context = struct {
    allocator: std.mem.Allocator,
    engine: ?*Engine = null,
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
        if (self.engine) |e| {
            e.deinit();
        }
        const allocator = self.allocator;
        allocator.destroy(self);
    }

    /// Get or create the compute engine.
    pub fn getEngine(self: *Context) Error!*Engine {
        if (self.engine) |e| return e;

        const engine_ptr = try self.allocator.create(Engine);
        engine_ptr.* = compute_runtime.createEngine(self.allocator, .{}) catch {
            self.allocator.destroy(engine_ptr);
            return Error.EngineCreationFailed;
        };
        self.engine = engine_ptr;
        return engine_ptr;
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

    /// Run a task and wait for result.
    pub fn runTask(
        self: *Context,
        comptime ResultType: type,
        task: anytype,
        timeout_ms: u64,
    ) !ResultType {
        const engine_inst = try self.getEngine();
        return compute_runtime.runTask(engine_inst, ResultType, task, timeout_ms);
    }

    /// Submit a task for async execution.
    pub fn submitTask(
        self: *Context,
        comptime ResultType: type,
        task: anytype,
    ) !TaskId {
        const engine_inst = try self.getEngine();
        return compute_runtime.submitTask(engine_inst, ResultType, task);
    }

    /// Wait for a task result.
    pub fn waitForResult(
        self: *Context,
        comptime ResultType: type,
        task_id: TaskId,
        timeout_ms: u64,
    ) !ResultType {
        const engine_inst = try self.getEngine();
        return compute_runtime.waitForResult(engine_inst, ResultType, task_id, timeout_ms);
    }
};

// ============================================================================
// Convenience functions
// ============================================================================

/// Create an engine with default configuration.
pub fn createEngine(allocator: std.mem.Allocator) !Engine {
    return compute_runtime.createEngine(allocator, .{});
}

/// Create an engine with custom configuration.
pub fn createEngineWithConfig(allocator: std.mem.Allocator, config: EngineConfig) !Engine {
    return compute_runtime.createEngine(allocator, config);
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

    try std.testing.expect(ctx.engine == null);
    const engine1 = try ctx.getEngine();
    try std.testing.expect(ctx.engine != null);
    const engine2 = try ctx.getEngine();
    try std.testing.expect(engine1 == engine2);
}
