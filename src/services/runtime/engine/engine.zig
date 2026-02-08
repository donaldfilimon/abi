//! Work-Stealing Distributed Compute Engine
//!
//! A high-performance task execution engine using work-stealing scheduling
//! for optimal CPU utilization. This is the core runtime for executing
//! asynchronous compute tasks across multiple worker threads.
//!
//! ## Architecture
//!
//! The engine uses a work-stealing scheduler where each worker thread has its
//! own local queue. When a worker runs out of tasks, it "steals" work from
//! other workers' queues, ensuring balanced load distribution.
//!
//! ```
//! ┌─────────────────────────────────────────────────────┐
//! │              DistributedComputeEngine               │
//! ├─────────────────────────────────────────────────────┤
//! │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
//! │  │ Worker 0│  │ Worker 1│  │ Worker N│   ...       │
//! │  │  Queue  │  │  Queue  │  │  Queue  │             │
//! │  └────┬────┘  └────┬────┘  └────┬────┘             │
//! │       │  ←steal→   │  ←steal→   │                  │
//! │       └────────────┴────────────┘                  │
//! │                     │                              │
//! │              ┌──────┴──────┐                       │
//! │              │ ShardedMap  │ (results)             │
//! │              └─────────────┘                       │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Features
//!
//! - **Work-stealing scheduling**: Automatic load balancing across workers
//! - **NUMA awareness**: Optional CPU affinity and topology-aware scheduling
//! - **Lock-free queues**: Minimal contention between workers
//! - **Sharded results**: Reduced lock contention for result storage
//! - **Generic tasks**: Support for any callable returning any result type
//!
//! ## Usage
//!
//! ```zig
//! const engine = @import("engine/engine.zig");
//!
//! // Initialize with configuration
//! var compute = try engine.DistributedComputeEngine.init(allocator, .{
//!     .max_tasks = 1024,
//!     .worker_count = 4,
//!     .numa_enabled = true,
//! });
//! defer compute.deinit();
//!
//! // Submit a task (function or struct with execute method)
//! const task_id = try compute.submit_task(u32, myTask);
//!
//! // Wait for result with timeout
//! const result = try compute.wait_for_result(u32, task_id, 5000);
//! ```
//!
//! ## Task Types
//!
//! Tasks can be:
//! - A function: `fn(allocator: std.mem.Allocator) !ResultType`
//! - A struct with `execute` method: `fn(self: @This(), allocator: std.mem.Allocator) !ResultType`
//!
//! ## Error Handling
//!
//! The engine provides structured error handling:
//! - `QueueFull`: Task queue is at capacity (increase `max_tasks`)
//! - `Timeout`: Result not ready within timeout period
//! - `TaskFailed`: Task execution threw an error
//! - `UnsupportedResultType`: Result type cannot be serialized
//!
//! ## Thread Safety
//!
//! The engine is fully thread-safe:
//! - `submit_task`: Safe to call from any thread
//! - `wait_for_result`: Safe to call from any thread
//! - Internal synchronization via atomics and condition variables
//!
//! ## Memory Management
//!
//! - Task payloads are owned by the engine until execution completes
//! - For `[]const u8` results, caller receives ownership and must free
//! - Other result types are copied; no ownership transfer
//!
//! ## Module Organization
//!
//! This module is split for maintainability:
//! - `engine.zig`: Core engine implementation (this file)
//! - `types.zig`: Type definitions, errors, config, and utilities
//! - `numa.zig`: NUMA topology detection and CPU affinity
//! - `steal_policy.zig`: Work-stealing victim selection policies

const std = @import("std");
const builtin = @import("builtin");
const numa = @import("numa.zig");
const concurrency = @import("../concurrency/mod.zig");

const sync = @import("../../shared/sync.zig");
const Mutex = sync.Mutex;

// Zig 0.16 compatibility: Simple Condition (busy-wait implementation)
const Condition = struct {
    waiters: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    pub fn wait(self: *Condition, mutex: anytype) void {
        _ = self.waiters.fetchAdd(1, .monotonic);
        mutex.unlock();
        while (self.waiters.load(.acquire) > 0) {
            std.atomic.spinLoopHint();
        }
        mutex.lock();
    }

    pub fn signal(self: *Condition) void {
        const current = self.waiters.load(.acquire);
        if (current > 0) {
            _ = self.waiters.fetchSub(1, .release);
        }
    }

    pub fn broadcast(self: *Condition) void {
        self.waiters.store(0, .release);
    }
};

/// Whether threading is available on this target
const is_threaded_target = builtin.target.os.tag != .freestanding and
    builtin.target.cpu.arch != .wasm32 and
    builtin.target.cpu.arch != .wasm64;

// Import types from submodule
pub const engine_types = @import("types.zig");

// Re-export types
pub const Backoff = engine_types.Backoff;
pub const nowMilliseconds = engine_types.nowMilliseconds;
pub const EngineError = engine_types.EngineError;
pub const TaskExecuteError = engine_types.TaskExecuteError;
pub const TaskExecuteFn = engine_types.TaskExecuteFn;
pub const TaskId = engine_types.TaskId;
pub const DEFAULT_MAX_TASKS = engine_types.DEFAULT_MAX_TASKS;
pub const EngineConfig = engine_types.EngineConfig;
pub const ResultKind = engine_types.ResultKind;
pub const ResultBlob = engine_types.ResultBlob;
pub const TaskNode = engine_types.TaskNode;
pub const isByteSlice = engine_types.isByteSlice;
pub const encodeResult = engine_types.encodeResult;
pub const callTask = engine_types.callTask;

/// Worker thread state.
/// Each worker has its own task queue for work-stealing.
const Worker = struct {
    /// Worker index (0 to worker_count-1)
    index: usize,
    /// Local task queue (push/pop from back, steal from front)
    queue: concurrency.WorkStealingQueue(*TaskNode),
    /// OS thread handle (null until started)
    thread: ?std.Thread = null,
};

/// Number of shards for the results map (reduces lock contention)
const RESULT_SHARD_COUNT: usize = 16;

/// Internal engine state.
///
/// This struct holds all mutable state for the engine, separated from the
/// public interface to allow the engine handle to be copied without issues.
///
/// ## Synchronization Strategy
///
/// - `next_id`: Atomic counter for task IDs
/// - `inflight_tasks`: Atomic counter for backpressure
/// - `results`: Sharded map with per-shard locking
/// - `results_cond/mutex`: Condition variable for result waiting
/// - `work_cond/mutex`: Condition variable for worker wakeup
/// - `stop_flag`: Atomic flag for graceful shutdown
const EngineState = struct {
    allocator: std.mem.Allocator,
    config: EngineConfig,
    /// Monotonically increasing task ID counter
    next_id: std.atomic.Value(TaskId) = std.atomic.Value(TaskId).init(1),
    /// Number of tasks currently in-flight (for backpressure)
    inflight_tasks: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    /// Sharded map for reduced lock contention under concurrent load
    results: concurrency.ShardedMap(TaskId, ResultBlob, RESULT_SHARD_COUNT),
    /// Condition variable for threads waiting on results
    results_cond: Condition = .{},
    /// Mutex for result condition variable (not for results map)
    results_mutex: Mutex = .{},
    /// Array of worker threads
    workers: []Worker,
    /// Number of tasks pending execution
    pending_tasks: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    /// Mutex for worker condition variable
    work_mutex: Mutex = .{},
    /// Condition variable to wake idle workers
    work_cond: Condition = .{},
    /// Flag to signal workers to shut down
    stop_flag: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Round-robin counter for task distribution
    next_worker: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    /// NUMA topology (optional, for affinity)
    topology: ?*numa.CpuTopology = null,
    /// Whether we own the topology (and must free it)
    owns_topology: bool = false,
    /// CPU IDs for worker affinity
    cpu_ids: ?[]usize = null,
};

/// Distributed Compute Engine
///
/// The main entry point for submitting and executing tasks. This struct
/// is a lightweight handle that can be copied; all state is in `EngineState`.
///
/// ## Lifecycle
///
/// ```zig
/// var engine = try DistributedComputeEngine.init(allocator, config);
/// defer engine.deinit();
///
/// // Submit tasks...
/// const id = try engine.submit_task(u32, myTask);
/// const result = try engine.wait_for_result(u32, id, 1000);
/// ```
///
/// ## Worker Threads
///
/// Workers are started during `init` and run until `deinit`. Each worker:
/// 1. Tries to pop from its local queue
/// 2. If empty, tries to steal from other workers
/// 3. If no work found, sleeps on condition variable
///
/// ## Backpressure
///
/// The engine limits in-flight tasks to `config.max_tasks`. If this limit
/// is reached, `submit_task` returns `EngineError.QueueFull`.
pub const DistributedComputeEngine = struct {
    state: *EngineState,

    /// Initialize the engine with the given configuration.
    ///
    /// This starts worker threads that will process tasks. The number of
    /// workers defaults to `CPU_count - 1` if not specified.
    ///
    /// ## Errors
    /// - `OutOfMemory`: Failed to allocate engine state or workers
    pub fn init(
        allocator: std.mem.Allocator,
        engine_config: EngineConfig,
    ) !DistributedComputeEngine {
        const state = try initState(allocator, engine_config);
        return .{ .state = state };
    }

    pub fn deinit(self: *DistributedComputeEngine) void {
        deinitState(self.state);
        self.* = undefined;
    }

    pub fn submit_task(
        self: *DistributedComputeEngine,
        comptime ResultType: type,
        task: anytype,
    ) !TaskId {
        const state = self.state;
        if (!reserveTaskSlot(state)) {
            return EngineError.QueueFull;
        }
        errdefer releaseTaskSlot(state);

        const id = state.next_id.fetchAdd(1, .acq_rel);
        const node = try createTaskNode(state, id, ResultType, task);
        errdefer destroyTaskNode(state, node);

        try enqueueTask(state, node);
        return id;
    }

    /// Wait for a task result to become available.
    /// Returns immediately if result is ready, or blocks up to timeout_ms.
    /// For ResultType = []const u8, the caller receives ownership of the slice
    /// and must free it using the allocator passed to submit_task.
    pub fn wait_for_result(
        self: *DistributedComputeEngine,
        comptime ResultType: type,
        id: TaskId,
        timeout_ms: u64,
    ) !ResultType {
        const state = self.state;
        const start_ms: i64 = nowMilliseconds();
        const deadline_ms: ?i64 = if (timeout_ms == 0)
            null
        else
            start_ms + @as(i64, @intCast(timeout_ms));

        while (true) {
            // ShardedMap handles its own locking, no need for results_mutex here
            if (state.results.fetchRemove(id)) |entry| {
                defer releaseTaskSlot(state);
                return decodeResult(state, ResultType, id, entry.value);
            }

            if (timeout_ms == 0) {
                return EngineError.Timeout;
            }
            if (deadline_ms) |deadline| {
                const now_ms = nowMilliseconds();
                if (now_ms >= deadline) {
                    return EngineError.Timeout;
                }
                const remaining_ms: u64 = @intCast(deadline - now_ms);
                const remaining_ns = remaining_ms * std.time.ns_per_ms;
                // Use mutex only for condition variable signaling
                state.results_mutex.lock();
                defer state.results_mutex.unlock();
                // Condition variable timeout is expected behavior; continue polling
                state.results_cond.timedWait(&state.results_mutex, remaining_ns) catch |err| {
                    std.log.debug("Result condition wait returned: {t}", .{err});
                };
            }
        }
    }

    pub fn config(self: *const DistributedComputeEngine) EngineConfig {
        return self.state.config;
    }

    pub fn nextId(self: *const DistributedComputeEngine) TaskId {
        return self.state.next_id.load(.acquire);
    }

    pub fn workerCount(self: *const DistributedComputeEngine) usize {
        return self.state.workers.len;
    }

    pub fn getCurrentNumaNode(self: *DistributedComputeEngine) ?*const numa.NumaNode {
        const state = self.state;
        if (state.topology) |*topo| {
            const cpu_id = numa.getCurrentCpuId() catch return null;
            return topo.getNodeForCpu(cpu_id);
        }
        return null;
    }

    pub fn setTaskAffinity(self: *DistributedComputeEngine, task_id: TaskId, cpu_id: usize) !void {
        _ = task_id;
        if (self.state.config.cpu_affinity_enabled) {
            try numa.setThreadAffinity(cpu_id);
        }
    }

    pub fn setTaskAffinityToNode(
        self: *DistributedComputeEngine,
        task_id: TaskId,
        node_id: usize,
    ) !void {
        _ = task_id;
        const state = self.state;
        if (state.config.cpu_affinity_enabled and state.topology) |*topo| {
            if (node_id < topo.nodes.len) {
                const node = &topo.nodes[node_id];
                if (node.cpus.len > 0) {
                    try numa.setThreadAffinity(node.cpus[0]);
                }
            }
        }
    }
};

/// Initialize engine state and start worker threads.
///
/// This function:
/// 1. Allocates and initializes EngineState
/// 2. Sets up NUMA topology if enabled
/// 3. Builds CPU ID list for affinity
/// 4. Creates worker queues
/// 5. Spawns worker threads
fn initState(allocator: std.mem.Allocator, config: EngineConfig) !*EngineState {
    const state = try allocator.create(EngineState);
    errdefer allocator.destroy(state);

    var topology: ?*numa.CpuTopology = config.numa_topology;
    var owns_topology = false;
    if (config.numa_enabled and topology == null) {
        const topo = try allocator.create(numa.CpuTopology);
        topo.* = try numa.CpuTopology.init(allocator);
        topology = topo;
        owns_topology = true;
    }

    var cpu_ids: ?[]usize = null;
    if (config.cpu_affinity_enabled) {
        cpu_ids = try buildCpuIds(allocator, topology);
    }

    state.* = .{
        .allocator = allocator,
        .config = config,
        .results = concurrency.ShardedMap(TaskId, ResultBlob, RESULT_SHARD_COUNT).init(allocator),
        .workers = &.{},
        .topology = topology,
        .owns_topology = owns_topology,
        .cpu_ids = cpu_ids,
    };

    if (!builtin.single_threaded) {
        const worker_count = computeWorkerCount(config);
        if (worker_count > 0) {
            state.workers = try allocator.alloc(Worker, worker_count);
            errdefer deinitWorkers(state);
            for (state.workers, 0..) |*worker, i| {
                worker.* = .{
                    .index = i,
                    .queue = concurrency.WorkStealingQueue(*TaskNode).init(allocator),
                    .thread = null,
                };
            }
            for (state.workers) |*worker| {
                worker.thread = try std.Thread.spawn(.{}, workerMain, .{ state, worker });
            }
        }
    }

    return state;
}

fn deinitState(state: *EngineState) void {
    stopWorkers(state);
    drainQueues(state);
    deinitWorkers(state);
    deinitResults(state);

    if (state.cpu_ids) |ids| {
        state.allocator.free(ids);
    }
    if (state.owns_topology) {
        if (state.topology) |topo| {
            topo.deinit(state.allocator);
            state.allocator.destroy(topo);
        }
    }

    state.allocator.destroy(state);
}

fn computeWorkerCount(config: EngineConfig) usize {
    if (config.worker_count) |count| {
        return count;
    }
    const cpu_count: usize = if (comptime is_threaded_target)
        std.Thread.getCpuCount() catch 1
    else
        1;
    if (cpu_count <= 1) {
        return 1;
    }
    return cpu_count - 1;
}

fn buildCpuIds(
    allocator: std.mem.Allocator,
    topology: ?*numa.CpuTopology,
) ![]usize {
    if (topology) |topo| {
        var list = std.ArrayListUnmanaged(usize).empty;
        errdefer list.deinit(allocator);
        for (topo.nodes) |node| {
            try list.appendSlice(allocator, node.cpus);
        }
        if (list.items.len == 0) {
            try list.ensureTotalCapacity(allocator, topo.cpu_count);
            var cpu_id: usize = 0;
            while (cpu_id < topo.cpu_count) {
                try list.append(allocator, cpu_id);
                cpu_id += 1;
            }
        }
        return list.toOwnedSlice(allocator);
    }

    const cpu_count: usize = if (comptime is_threaded_target)
        std.Thread.getCpuCount() catch 1
    else
        1;
    const ids = try allocator.alloc(usize, cpu_count);
    var i: usize = 0;
    while (i < ids.len) {
        ids[i] = i;
        i += 1;
    }
    return ids;
}

fn stopWorkers(state: *EngineState) void {
    state.stop_flag.store(true, .release);
    state.work_mutex.lock();
    state.work_cond.broadcast();
    state.work_mutex.unlock();
    for (state.workers) |*worker| {
        if (worker.thread) |thread| {
            thread.join();
        }
    }
}

fn drainQueues(state: *EngineState) void {
    for (state.workers) |*worker| {
        while (worker.queue.pop()) |node| {
            destroyTaskNode(state, node);
            releaseTaskSlot(state);
        }
    }
}

fn deinitWorkers(state: *EngineState) void {
    if (state.workers.len == 0) {
        return;
    }
    for (state.workers) |*worker| {
        worker.queue.deinit();
    }
    state.allocator.free(state.workers);
    state.workers = &.{};
}

fn deinitResults(state: *EngineState) void {
    // Clean up any remaining result blobs before deiniting the map
    for (&state.results.shards) |*shard| {
        shard.mutex.lock();
        defer shard.mutex.unlock();
        var it = shard.map.valueIterator();
        while (it.next()) |blob| {
            if (blob.kind == .value or blob.kind == .owned_slice) {
                state.allocator.free(blob.bytes);
            }
        }
    }
    state.results.deinit();
}

/// Atomically reserve a task slot for backpressure control.
///
/// Returns true if a slot was reserved, false if at capacity.
/// Uses compare-and-swap to avoid races between concurrent submitters.
///
/// This prevents the engine from being overwhelmed with more tasks
/// than it can process, allowing callers to implement flow control.
fn reserveTaskSlot(state: *EngineState) bool {
    const limit = state.config.max_tasks;
    if (limit == 0) {
        return false;
    }
    while (true) {
        const current = state.inflight_tasks.load(.acquire);
        if (current >= limit) {
            return false;
        }
        if (state.inflight_tasks.cmpxchgWeak(current, current + 1, .acq_rel, .acquire) == null) {
            return true;
        }
    }
}

fn releaseTaskSlot(state: *EngineState) void {
    _ = state.inflight_tasks.fetchSub(1, .acq_rel);
}

fn createTaskNode(
    state: *EngineState,
    id: TaskId,
    comptime ResultType: type,
    task: anytype,
) !*TaskNode {
    const TaskType = @TypeOf(task);
    const Payload = struct {
        task: TaskType,
    };

    const payload = try state.allocator.create(Payload);
    errdefer state.allocator.destroy(payload);
    payload.* = .{ .task = task };

    const NodeOps = struct {
        fn run(allocator: std.mem.Allocator, context: *anyopaque) TaskExecuteError!ResultBlob {
            const payload_ptr: *Payload = @ptrCast(@alignCast(context));
            const result = callTask(ResultType, payload_ptr.task, allocator) catch |err| {
                return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.ExecutionFailed,
                };
            };
            return encodeResult(allocator, ResultType, result) catch error.OutOfMemory;
        }

        fn destroy(allocator: std.mem.Allocator, context: *anyopaque) void {
            const payload_ptr: *Payload = @ptrCast(@alignCast(context));
            allocator.destroy(payload_ptr);
        }
    };

    const node = try state.allocator.create(TaskNode);
    errdefer state.allocator.destroy(node);
    node.* = .{
        .id = id,
        .execute = NodeOps.run,
        .destroy = NodeOps.destroy,
        .payload = payload,
    };
    return node;
}

fn destroyTaskNode(state: *EngineState, node: *TaskNode) void {
    node.destroy(state.allocator, node.payload);
    state.allocator.destroy(node);
}

fn enqueueTask(state: *EngineState, node: *TaskNode) !void {
    if (state.workers.len == 0) {
        try executeTaskInline(state, node);
        return;
    }

    const idx = state.next_worker.fetchAdd(1, .acq_rel) % state.workers.len;
    try state.workers[idx].queue.push(node);
    _ = state.pending_tasks.fetchAdd(1, .acq_rel);
    wakeWorkers(state);
}

fn wakeWorkers(state: *EngineState) void {
    state.work_mutex.lock();
    state.work_cond.broadcast();
    state.work_mutex.unlock();
}

/// Worker thread main loop.
///
/// Each worker runs this function until the engine is shut down.
/// The loop implements the work-stealing algorithm:
///
/// 1. Apply CPU affinity (if enabled)
/// 2. Try to pop a task from local queue
/// 3. If no local work, try to steal from other workers
/// 4. If no work anywhere, use exponential backoff then sleep
///
/// The backoff strategy minimizes CPU usage when idle while still
/// being responsive when new work arrives.
fn workerMain(state: *EngineState, worker: *Worker) void {
    applyAffinity(state, worker.index);
    var backoff = Backoff{};
    while (!state.stop_flag.load(.acquire)) {
        if (popTask(state, worker.index)) |task| {
            backoff.reset();
            executeTask(state, task);
            continue;
        }
        if (backoff.spins < 16) {
            backoff.spin();
            continue;
        }
        waitForWork(state);
        backoff.reset();
    }
}

fn waitForWork(state: *EngineState) void {
    state.work_mutex.lock();
    defer state.work_mutex.unlock();
    while (state.pending_tasks.load(.acquire) == 0 and !state.stop_flag.load(.acquire)) {
        state.work_cond.wait(&state.work_mutex);
    }
}

/// Try to get a task for a worker.
///
/// First attempts to pop from the worker's local queue (LIFO order).
/// If local queue is empty, attempts to steal from other workers
/// (FIFO order from victim's perspective).
///
/// Work-stealing ensures load balancing: busy workers don't starve
/// idle workers, and tasks are migrated to available capacity.
fn popTask(state: *EngineState, worker_index: usize) ?*TaskNode {
    if (state.workers[worker_index].queue.pop()) |task| {
        _ = state.pending_tasks.fetchSub(1, .acq_rel);
        return task;
    }
    if (state.workers.len <= 1) {
        return null;
    }
    var i: usize = 0;
    while (i < state.workers.len) {
        if (i != worker_index) {
            if (state.workers[i].queue.steal()) |task| {
                _ = state.pending_tasks.fetchSub(1, .acq_rel);
                return task;
            }
        }
        i += 1;
    }
    return null;
}

fn executeTask(state: *EngineState, node: *TaskNode) void {
    // Save task ID before executing - node may be destroyed after execute
    const task_id = node.id;

    const result = node.execute(state.allocator, node.payload) catch |err| {
        storeTaskError(state, task_id, err);
        destroyTaskNode(state, node);
        return;
    };

    // Destroy node BEFORE using task_id (not node.id) to avoid use-after-free
    destroyTaskNode(state, node);

    storeResultBlob(state, task_id, result) catch |err| {
        std.log.err("Failed to store result for task {d}: {t}", .{ task_id, err });
        if (result.kind == .value or result.kind == .owned_slice) {
            state.allocator.free(result.bytes);
        }
        releaseTaskSlot(state);
    };
}

fn executeTaskInline(state: *EngineState, node: *TaskNode) !void {
    // Save task ID before executing - node may be destroyed after execute
    const task_id = node.id;

    const result = node.execute(state.allocator, node.payload) catch |err| {
        destroyTaskNode(state, node);
        return err;
    };

    // Destroy node BEFORE using task_id (not node.id) to avoid use-after-free
    destroyTaskNode(state, node);

    storeResultBlob(state, task_id, result) catch |err| {
        if (result.kind == .value or result.kind == .owned_slice) {
            state.allocator.free(result.bytes);
        }
        return err;
    };
}

fn storeTaskError(state: *EngineState, id: TaskId, err: TaskExecuteError) void {
    std.log.debug("Task {d} execution failed: {t}", .{ id, err });
    const blob = ResultBlob{
        .kind = .task_error,
        .bytes = &.{},
        .size = 0,
        .error_code = @intFromError(err),
    };
    storeResultBlob(state, id, blob) catch |store_err| {
        std.log.err("Failed to store error for task {d}: {t} (original error: {t})", .{
            id,
            store_err,
            err,
        });
        releaseTaskSlot(state);
    };
}

fn storeResultBlob(state: *EngineState, id: TaskId, blob: ResultBlob) std.mem.Allocator.Error!void {
    // ShardedMap handles its own locking for the put operation
    try state.results.put(id, blob);
    // Signal condition variable for waiting threads
    state.results_mutex.lock();
    state.results_cond.broadcast();
    state.results_mutex.unlock();
}

fn applyAffinity(state: *EngineState, worker_index: usize) void {
    if (!state.config.cpu_affinity_enabled) {
        return;
    }
    const cpu_ids = state.cpu_ids orelse return;
    if (cpu_ids.len == 0) {
        return;
    }
    const cpu_id = cpu_ids[worker_index % cpu_ids.len];
    numa.setThreadAffinity(cpu_id) catch |err| {
        std.log.warn("Failed to set CPU affinity for worker {d} to CPU {d}: {t}", .{
            worker_index,
            cpu_id,
            err,
        });
    };
}

fn decodeResult(
    state: *EngineState,
    comptime ResultType: type,
    id: TaskId,
    blob: ResultBlob,
) !ResultType {
    if (blob.kind == .task_error) {
        const err = @errorFromInt(blob.error_code);
        std.log.err("Task {d} failed with error: {t} (error code: {d})", .{ id, err, blob.error_code });
        std.log.debug("Original error details preserved; use debug build for task stack traces", .{});
        return EngineError.TaskFailed;
    }

    if (comptime isByteSlice(ResultType)) {
        if (blob.kind != .owned_slice) {
            state.allocator.free(blob.bytes);
            return EngineError.UnsupportedResultType;
        }
        return @as(ResultType, blob.bytes);
    }

    if (blob.kind != .value or blob.size != @sizeOf(ResultType)) {
        state.allocator.free(blob.bytes);
        return EngineError.UnsupportedResultType;
    }

    var value: ResultType = undefined;
    std.mem.copyForwards(u8, std.mem.asBytes(&value), blob.bytes);
    state.allocator.free(blob.bytes);
    return value;
}

test "engine runs simple task" {
    const allocator = std.testing.allocator;
    var engine = try DistributedComputeEngine.init(allocator, .{
        .max_tasks = 8,
        .worker_count = 1,
    });
    defer engine.deinit();

    const task_id = try engine.submit_task(u32, sampleTask);
    const result = try engine.wait_for_result(u32, task_id, 1000);
    try std.testing.expectEqual(@as(u32, 42), result);
}

test "engine reports timeout for non-existent result with zero timeout" {
    const allocator = std.testing.allocator;
    var engine = try DistributedComputeEngine.init(allocator, .{
        .max_tasks = 4,
        .worker_count = 1,
    });
    defer engine.deinit();

    try std.testing.expectError(EngineError.Timeout, engine.wait_for_result(u32, 999, 0));
}

test "engine reports queue full when max tasks reached" {
    const allocator = std.testing.allocator;
    var engine = try DistributedComputeEngine.init(allocator, .{
        .max_tasks = 2,
        .worker_count = 0,
    });
    defer engine.deinit();

    _ = try engine.submit_task(u32, sampleTask);
    _ = try engine.submit_task(u32, sampleTask);

    try std.testing.expectError(EngineError.QueueFull, engine.submit_task(u32, sampleTask));
}

test "engine handles byte slice results" {
    const allocator = std.testing.allocator;
    var engine = try DistributedComputeEngine.init(allocator, .{
        .max_tasks = 8,
        .worker_count = 1,
    });
    defer engine.deinit();

    const task_id = try engine.submit_task([]const u8, byteSliceTask);
    const result = try engine.wait_for_result([]const u8, task_id, 100);
    try std.testing.expectEqual(@as(usize, 4), result.len);
    allocator.free(result);
}

test "engine handles slice results directly" {
    const allocator = std.testing.allocator;
    var engine = try DistributedComputeEngine.init(allocator, .{
        .max_tasks = 8,
        .worker_count = 1,
    });
    defer engine.deinit();

    const task_id = try engine.submit_task([]const u8, struct {
        pub fn execute(_: @This(), task_allocator: std.mem.Allocator) ![]const u8 {
            return task_allocator.dupe(u8, "test");
        }
    }{});
    const result = try engine.wait_for_result([]const u8, task_id, 100);
    try std.testing.expectEqual(@as(usize, 4), result.len);
    allocator.free(result);
}

fn sampleTask(_: std.mem.Allocator) !u32 {
    return 42;
}

fn byteSliceTask(allocator: std.mem.Allocator) ![]const u8 {
    return allocator.dupe(u8, "test");
}
