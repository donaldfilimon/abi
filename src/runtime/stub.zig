//! Runtime Stub Module
//!
//! Provides stub implementations when runtime is disabled at compile time.
//! All operations return error.RuntimeDisabled.

const std = @import("std");

// ============================================================================
// Errors
// ============================================================================

pub const Error = error{
    RuntimeDisabled,
    EngineCreationFailed,
    TaskCreationFailed,
    TaskGroupFailed,
    SchedulingFailed,
    ConcurrencyError,
    MemoryPoolError,
    AlreadyInitialized,
    NotInitialized,
    ModuleDisabled,
    FeatureNotAvailable,
    InvalidOperation,
};

pub const EngineError = Error;
pub const SchedulingError = Error;
pub const ConcurrencyError = Error;
pub const MemoryError = Error;

// ============================================================================
// Engine Types
// ============================================================================

pub const Engine = struct {
    pub fn init(_: std.mem.Allocator, _: EngineConfig) Error!Engine {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *Engine) void {}

    pub fn submit(_: *Engine, _: anytype) Error!TaskId {
        return error.RuntimeDisabled;
    }

    pub fn wait(_: *Engine, _: TaskId) Error!void {
        return error.RuntimeDisabled;
    }

    pub fn getStats(_: *const Engine) EngineStats {
        return .{};
    }
};

pub const DistributedComputeEngine = struct {
    pub fn init(_: std.mem.Allocator, _: EngineConfig) Error!DistributedComputeEngine {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *DistributedComputeEngine) void {}
};

pub const EngineConfig = struct {
    thread_count: ?usize = null,
    enable_work_stealing: bool = true,
    task_queue_size: usize = 1024,
};

pub const EngineStats = struct {
    tasks_completed: usize = 0,
    tasks_pending: usize = 0,
    workers_active: usize = 0,
};

pub const TaskId = u64;

// Workload types
pub const ExecutionContext = struct {};
pub const WorkloadHints = struct {
    priority: Priority = .normal,
    estimated_cycles: ?u64 = null,
};
pub const Priority = enum { low, normal, high, critical };
pub const WorkloadVTable = struct {};
pub const GPUWorkloadVTable = struct {};
pub const ResultHandle = struct {};
pub const ResultVTable = struct {};
pub const WorkItem = struct {};

pub fn runWorkItem(_: WorkItem) Error!void {
    return error.RuntimeDisabled;
}

// Benchmarking
pub const BenchmarkResult = struct {
    name: []const u8 = "",
    iterations: usize = 0,
    total_time_ns: u64 = 0,
    avg_time_ns: u64 = 0,
};

pub fn runBenchmarks(_: std.mem.Allocator) Error![]BenchmarkResult {
    return error.RuntimeDisabled;
}

// ============================================================================
// Scheduling Types
// ============================================================================

pub fn Future(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator) Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn get(_: *Self) Error!T {
            return error.RuntimeDisabled;
        }

        pub fn getState(_: *const Self) FutureState {
            return .pending;
        }

        pub fn cancel(_: *Self) void {}
    };
}

pub const FutureState = enum { pending, ready, cancelled, failed };
pub const FutureResult = struct {};

pub fn Promise(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator) Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn set(_: *Self, _: T) void {}

        pub fn setError(_: *Self, _: anyerror) void {}
    };
}

pub fn all(_: std.mem.Allocator, _: anytype) Error!void {
    return error.RuntimeDisabled;
}

pub fn race(_: std.mem.Allocator, _: anytype) Error!void {
    return error.RuntimeDisabled;
}

pub fn delay(_: u64) Error!void {
    return error.RuntimeDisabled;
}

// Cancellation
pub const CancellationToken = struct {
    pub fn isCancelled(_: *const CancellationToken) bool {
        return false;
    }

    pub fn getState(_: *const CancellationToken) CancellationState {
        return .none;
    }
};

pub const CancellationSource = struct {
    pub fn init(_: std.mem.Allocator) Error!CancellationSource {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *CancellationSource) void {}

    pub fn cancel(_: *CancellationSource) void {}

    pub fn token(_: *CancellationSource) CancellationToken {
        return .{};
    }
};

pub const CancellationState = enum { none, requested, acknowledged };
pub const CancellationReason = enum { user_requested, timeout, error_occurred };
pub const LinkedCancellation = struct {};
pub const ScopedCancellation = struct {};

// Task groups
pub const TaskGroup = struct {
    pub fn init(_: std.mem.Allocator, _: TaskGroupConfig) Error!TaskGroup {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *TaskGroup) void {}

    pub fn spawn(_: *TaskGroup, _: anytype) Error!void {
        return error.RuntimeDisabled;
    }

    pub fn wait(_: *TaskGroup) Error!void {
        return error.RuntimeDisabled;
    }

    pub fn getStats(_: *const TaskGroup) GroupStats {
        return .{};
    }
};

pub const TaskGroupConfig = struct {
    max_concurrent: ?usize = null,
    cancellation_token: ?CancellationToken = null,
};

pub const TaskGroupBuilder = struct {
    pub fn init(_: std.mem.Allocator) TaskGroupBuilder {
        return .{};
    }

    pub fn withMaxConcurrent(_: *TaskGroupBuilder, _: usize) *TaskGroupBuilder {
        return undefined;
    }

    pub fn build(_: *TaskGroupBuilder) Error!TaskGroup {
        return error.RuntimeDisabled;
    }
};

pub const ScopedTaskGroup = struct {};

pub const TaskContext = struct {
    allocator: std.mem.Allocator,
    cancellation_token: ?CancellationToken = null,
};

pub const TaskFn = *const fn (*TaskContext) anyerror!void;
pub const TaskState = enum { pending, running, completed, failed, cancelled };
pub const TaskResult = struct {};
pub const TaskInfo = struct {
    id: TaskId = 0,
    state: TaskState = .pending,
};
pub const GroupStats = struct {
    tasks_submitted: usize = 0,
    tasks_completed: usize = 0,
    tasks_failed: usize = 0,
};

pub fn parallelForEach(_: std.mem.Allocator, _: anytype, _: anytype) Error!void {
    return error.RuntimeDisabled;
}

// Async runtime
pub const AsyncRuntime = struct {
    pub fn init(_: std.mem.Allocator, _: AsyncRuntimeOptions) Error!AsyncRuntime {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *AsyncRuntime) void {}

    pub fn spawn(_: *AsyncRuntime, _: anytype) Error!TaskHandle {
        return error.RuntimeDisabled;
    }

    pub fn run(_: *AsyncRuntime) Error!void {
        return error.RuntimeDisabled;
    }
};

pub const AsyncRuntimeOptions = struct {
    thread_pool_size: ?usize = null,
};

pub const TaskHandle = struct {
    id: TaskId = 0,

    pub fn wait(_: *TaskHandle) Error!void {
        return error.RuntimeDisabled;
    }

    pub fn cancel(_: *TaskHandle) void {}
};

pub const AsyncTaskGroup = struct {};
pub const AsyncError = Error;

// ============================================================================
// Concurrency Types
// ============================================================================

pub fn WorkStealingQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: usize) Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn push(_: *Self, _: T) Error!void {
            return error.RuntimeDisabled;
        }

        pub fn pop(_: *Self) ?T {
            return null;
        }

        pub fn steal(_: *Self) ?T {
            return null;
        }
    };
}

pub fn WorkQueue(comptime T: type) type {
    return WorkStealingQueue(T);
}

pub fn LockFreeQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: usize) Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn push(_: *Self, _: T) Error!void {
            return error.RuntimeDisabled;
        }

        pub fn pop(_: *Self) ?T {
            return null;
        }
    };
}

pub fn LockFreeStack(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator) Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn push(_: *Self, _: T) Error!void {
            return error.RuntimeDisabled;
        }

        pub fn pop(_: *Self) ?T {
            return null;
        }
    };
}

pub fn ShardedMap(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: usize) Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn put(_: *Self, _: K, _: V) Error!void {
            return error.RuntimeDisabled;
        }

        pub fn get(_: *Self, _: K) ?V {
            return null;
        }
    };
}

pub fn PriorityQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator) Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn push(_: *Self, _: T) Error!void {
            return error.RuntimeDisabled;
        }

        pub fn pop(_: *Self) ?T {
            return null;
        }
    };
}

pub const Backoff = struct {
    pub fn init() Backoff {
        return .{};
    }

    pub fn spin(_: *Backoff) void {}

    pub fn reset(_: *Backoff) void {}
};

// New lock-free concurrency primitives
pub fn ChaseLevDeque(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: usize) Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn push(_: *Self, _: T) Error!void {
            return error.RuntimeDisabled;
        }

        pub fn pop(_: *Self) ?T {
            return null;
        }

        pub fn steal(_: *Self) ?T {
            return null;
        }
    };
}

pub const WorkStealingScheduler = struct {
    pub fn init(_: std.mem.Allocator, _: usize) Error!WorkStealingScheduler {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *WorkStealingScheduler) void {}

    pub fn schedule(_: *WorkStealingScheduler, _: anytype) Error!void {
        return error.RuntimeDisabled;
    }
};

pub fn EpochReclamation(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator) Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn retire(_: *Self, _: *T) void {}

        pub fn collect(_: *Self) void {}
    };
}

pub fn LockFreeStackEBR(comptime T: type) type {
    return LockFreeStack(T);
}

pub fn MpmcQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: usize) Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn push(_: *Self, _: T) Error!void {
            return error.RuntimeDisabled;
        }

        pub fn pop(_: *Self) ?T {
            return null;
        }
    };
}

pub fn BlockingMpmcQueue(comptime T: type) type {
    return MpmcQueue(T);
}

// Result caching
pub fn ResultCache(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: CacheConfig) Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn get(_: *Self, _: K) ?V {
            return null;
        }

        pub fn put(_: *Self, _: K, _: V) Error!void {
            return error.RuntimeDisabled;
        }

        pub fn getStats(_: *const Self) CacheStats {
            return .{};
        }
    };
}

pub const CacheConfig = struct {
    max_entries: usize = 1024,
    ttl_ms: ?u64 = null,
};

pub const CacheStats = struct {
    hits: usize = 0,
    misses: usize = 0,
    evictions: usize = 0,
};

pub fn Memoize(comptime F: type) type {
    _ = F;
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: anytype) Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn call(_: *Self, _: anytype) Error!void {
            return error.RuntimeDisabled;
        }
    };
}

// Work-stealing policies
pub const NumaStealPolicy = struct {
    pub fn init(_: StealPolicyConfig) NumaStealPolicy {
        return .{};
    }

    pub fn selectVictim(_: *NumaStealPolicy, _: usize) ?usize {
        return null;
    }
};

pub const RoundRobinStealPolicy = struct {
    pub fn init(_: StealPolicyConfig) RoundRobinStealPolicy {
        return .{};
    }

    pub fn selectVictim(_: *RoundRobinStealPolicy, _: usize) ?usize {
        return null;
    }
};

pub const StealPolicyConfig = struct {
    worker_count: usize = 1,
    numa_aware: bool = false,
};

pub const StealStats = struct {
    steal_attempts: usize = 0,
    steal_successes: usize = 0,
    steal_failures: usize = 0,
};

// ============================================================================
// Memory Types
// ============================================================================

pub const MemoryPool = struct {
    pub fn init(_: std.mem.Allocator, _: MemoryPoolConfig) Error!MemoryPool {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *MemoryPool) void {}

    pub fn alloc(_: *MemoryPool, _: usize) Error![]u8 {
        return error.RuntimeDisabled;
    }

    pub fn free(_: *MemoryPool, _: []u8) void {}

    pub fn getStats(_: *const MemoryPool) MemoryPoolStats {
        return .{};
    }
};

pub const MemoryPoolConfig = struct {
    block_size: usize = 4096,
    initial_blocks: usize = 16,
};

pub const MemoryPoolStats = struct {
    allocated_bytes: usize = 0,
    free_bytes: usize = 0,
    block_count: usize = 0,
};

pub const ArenaAllocator = struct {
    pub fn init(_: std.mem.Allocator) ArenaAllocator {
        return .{};
    }

    pub fn deinit(_: *ArenaAllocator) void {}

    pub fn allocator(_: *ArenaAllocator) std.mem.Allocator {
        return std.heap.page_allocator;
    }

    pub fn reset(_: *ArenaAllocator) void {}
};

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

const root = @This();

pub const engine = struct {
    pub const Engine = root.Engine;
    pub const DistributedComputeEngine = root.DistributedComputeEngine;
    pub const EngineConfig = root.EngineConfig;
    pub const EngineError = root.EngineError;
    pub const TaskId = root.TaskId;
    pub const BenchmarkResult = root.BenchmarkResult;
    pub const ResultCache = root.ResultCache;
    pub const CacheConfig = root.CacheConfig;
    pub const CacheStats = root.CacheStats;
    pub const Memoize = root.Memoize;
    pub const NumaStealPolicy = root.NumaStealPolicy;
    pub const RoundRobinStealPolicy = root.RoundRobinStealPolicy;
    pub const StealPolicyConfig = root.StealPolicyConfig;
    pub const StealStats = root.StealStats;

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
    pub const Future = root.Future;
    pub const FutureState = root.FutureState;
    pub const FutureResult = root.FutureResult;
    pub const Promise = root.Promise;
    pub const CancellationToken = root.CancellationToken;
    pub const CancellationSource = root.CancellationSource;
    pub const CancellationState = root.CancellationState;
    pub const CancellationReason = root.CancellationReason;
    pub const LinkedCancellation = root.LinkedCancellation;
    pub const ScopedCancellation = root.ScopedCancellation;
    pub const TaskGroup = root.TaskGroup;
    pub const TaskGroupConfig = root.TaskGroupConfig;
    pub const TaskGroupBuilder = root.TaskGroupBuilder;
    pub const ScopedTaskGroup = root.ScopedTaskGroup;
    pub const TaskContext = root.TaskContext;
    pub const TaskFn = root.TaskFn;
    pub const TaskState = root.TaskState;
    pub const TaskResult = root.TaskResult;
    pub const TaskInfo = root.TaskInfo;
    pub const GroupStats = root.GroupStats;
    pub const AsyncRuntime = root.AsyncRuntime;
    pub const AsyncRuntimeOptions = root.AsyncRuntimeOptions;
    pub const TaskHandle = root.TaskHandle;
    pub const AsyncTaskGroup = root.AsyncTaskGroup;
    pub const AsyncError = root.AsyncError;

    pub const all = root.all;
    pub const race = root.race;
    pub const delay = root.delay;
    pub const parallelForEach = root.parallelForEach;
};

pub const concurrency = struct {
    pub const WorkStealingQueue = root.WorkStealingQueue;
    pub const WorkQueue = root.WorkQueue;
    pub const LockFreeQueue = root.LockFreeQueue;
    pub const LockFreeStack = root.LockFreeStack;
    pub const ShardedMap = root.ShardedMap;
    pub const PriorityQueue = root.PriorityQueue;
    pub const Backoff = root.Backoff;
    pub const ChaseLevDeque = root.ChaseLevDeque;
    pub const WorkStealingScheduler = root.WorkStealingScheduler;
    pub const EpochReclamation = root.EpochReclamation;
    pub const LockFreeStackEBR = root.LockFreeStackEBR;
    pub const MpmcQueue = root.MpmcQueue;
    pub const BlockingMpmcQueue = root.BlockingMpmcQueue;
};

pub const memory = struct {
    pub const MemoryPool = root.MemoryPool;
    pub const ArenaAllocator = root.ArenaAllocator;
};

pub const workload = struct {
    pub const ExecutionContext = root.ExecutionContext;
    pub const WorkloadHints = root.WorkloadHints;
    pub const WorkloadVTable = root.WorkloadVTable;
    pub const GPUWorkloadVTable = root.GPUWorkloadVTable;
    pub const ResultHandle = root.ResultHandle;
    pub const ResultVTable = root.ResultVTable;
    pub const WorkItem = root.WorkItem;
    pub const runWorkItem = root.runWorkItem;
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
