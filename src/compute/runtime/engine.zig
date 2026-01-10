//! Minimal distributed compute engine for running synchronous tasks.
const std = @import("std");
const builtin = @import("builtin");
const numa = @import("numa.zig");
const concurrency = @import("../concurrency/mod.zig");

const Backoff = struct {
    spins: usize = 0,

    pub fn reset(self: *Backoff) void {
        self.spins = 0;
    }

    pub fn spin(self: *Backoff) void {
        self.spins += 1;
        if (self.spins <= 16) {
            std.atomic.spinLoopHint();
            return;
        }
        // Thread yield failure is non-critical; log at debug level and continue
        std.Thread.yield() catch |err| {
            std.log.debug("Thread yield failed during engine backoff spin: {t}", .{err});
        };
    }

    pub fn wait(self: *Backoff) void {
        self.spins += 1;
        const iterations = @min(self.spins, 64);
        var i: usize = 0;
        while (i < iterations) {
            std.atomic.spinLoopHint();
            i += 1;
        }
        if (self.spins > 32) {
            // Thread yield failure is non-critical; log at debug level and continue
            std.Thread.yield() catch |err| {
                std.log.debug("Thread yield failed during engine backoff wait: {t}", .{err});
            };
        }
    }
};

var global_timer: ?std.time.Timer = null;

fn nowMilliseconds() i64 {
    if (global_timer == null) {
        global_timer = std.time.Timer.start() catch null;
    }
    if (global_timer) |*timer| {
        const ns = timer.read();
        const ms = @as(f64, @floatFromInt(ns)) / @as(f64, std.time.ns_per_ms);
        return @as(i64, @intFromFloat(ms));
    }
    return 0;
}

pub const EngineError = error{
    ResultNotFound,
    Timeout,
    UnsupportedResultType,
    QueueFull,
    TaskFailed,
};

pub const TaskId = u64;

const DEFAULT_MAX_TASKS: usize = 1024;

pub const EngineConfig = struct {
    max_tasks: usize = DEFAULT_MAX_TASKS,
    worker_count: ?usize = null,
    numa_enabled: bool = false,
    cpu_affinity_enabled: bool = false,
    numa_topology: ?*numa.CpuTopology = null,
};

const ResultKind = enum {
    value,
    owned_slice,
    task_error,
};

const ResultBlob = struct {
    kind: ResultKind,
    bytes: []u8,
    size: usize,
    error_code: u16 = 0,
};

const TaskNode = struct {
    id: TaskId,
    execute: *const fn (std.mem.Allocator, *anyopaque) anyerror!ResultBlob,
    destroy: *const fn (std.mem.Allocator, *anyopaque) void,
    payload: *anyopaque,
};

const Worker = struct {
    index: usize,
    queue: concurrency.WorkStealingQueue(*TaskNode),
    thread: ?std.Thread = null,
};

const EngineState = struct {
    allocator: std.mem.Allocator,
    config: EngineConfig,
    next_id: std.atomic.Value(TaskId) = std.atomic.Value(TaskId).init(1),
    inflight_tasks: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    results: std.AutoHashMap(TaskId, ResultBlob),
    results_mutex: std.Thread.Mutex = .{},
    results_cond: std.Thread.Condition = .{},
    workers: []Worker,
    pending_tasks: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    work_mutex: std.Thread.Mutex = .{},
    work_cond: std.Thread.Condition = .{},
    stop_flag: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    next_worker: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    topology: ?*numa.CpuTopology = null,
    owns_topology: bool = false,
    cpu_ids: ?[]usize = null,
};

pub const DistributedComputeEngine = struct {
    state: *EngineState,

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

        state.results_mutex.lock();
        defer state.results_mutex.unlock();

        while (true) {
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
        .results = std.AutoHashMap(TaskId, ResultBlob).init(allocator),
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
    const cpu_count = std.Thread.getCpuCount() catch 1;
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

    const cpu_count = std.Thread.getCpuCount() catch 1;
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
    var it = state.results.valueIterator();
    while (it.next()) |blob| {
        if (blob.kind == .value or blob.kind == .owned_slice) {
            state.allocator.free(blob.bytes);
        }
    }
    state.results.deinit();
}

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
        fn run(allocator: std.mem.Allocator, context: *anyopaque) anyerror!ResultBlob {
            const payload_ptr: *Payload = @ptrCast(@alignCast(context));
            const result = try callTask(ResultType, payload_ptr.task, allocator);
            return encodeResult(allocator, ResultType, result);
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
    const result = node.execute(state.allocator, node.payload) catch |err| {
        storeTaskError(state, node.id, err);
        destroyTaskNode(state, node);
        return;
    };
    destroyTaskNode(state, node);
    storeResultBlob(state, node.id, result) catch |err| {
        std.log.err("Failed to store result for task {d}: {t}", .{ node.id, err });
        if (result.kind == .value or result.kind == .owned_slice) {
            state.allocator.free(result.bytes);
        }
        releaseTaskSlot(state);
    };
}

fn executeTaskInline(state: *EngineState, node: *TaskNode) !void {
    const result = node.execute(state.allocator, node.payload) catch |err| {
        destroyTaskNode(state, node);
        return err;
    };
    destroyTaskNode(state, node);
    storeResultBlob(state, node.id, result) catch |err| {
        if (result.kind == .value or result.kind == .owned_slice) {
            state.allocator.free(result.bytes);
        }
        return err;
    };
}

fn storeTaskError(state: *EngineState, id: TaskId, err: anyerror) void {
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
    state.results_mutex.lock();
    defer state.results_mutex.unlock();
    try state.results.put(id, blob);
    state.results_cond.broadcast();
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

fn encodeResult(
    allocator: std.mem.Allocator,
    comptime ResultType: type,
    result: ResultType,
) !ResultBlob {
    if (comptime isByteSlice(ResultType)) {
        const slice: []const u8 = result;
        const copy = try allocator.dupe(u8, slice);
        return .{
            .kind = .owned_slice,
            .bytes = copy,
            .size = copy.len,
        };
    }

    const size = @sizeOf(ResultType);
    const copy = try allocator.alloc(u8, size);
    std.mem.copyForwards(u8, copy, std.mem.asBytes(&result));
    return .{
        .kind = .value,
        .bytes = copy,
        .size = size,
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

fn isByteSlice(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .pointer => |pointer| pointer.size == .slice and pointer.child == u8,
        else => false,
    };
}

fn callTask(comptime ResultType: type, task: anytype, allocator: std.mem.Allocator) !ResultType {
    const TaskType = @TypeOf(task);
    switch (@typeInfo(TaskType)) {
        .@"fn" => return task(allocator),
        .pointer => |pointer| {
            if (@typeInfo(pointer.child) == .@"fn") {
                return task.*(allocator);
            }
        },
        else => {},
    }

    if (@hasDecl(TaskType, "execute")) {
        return task.execute(allocator);
    }

    @compileError("Task must be a function or type with execute(allocator)");
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
