//! Compute engine - core runtime
//!
//! Main execution engine with work-stealing scheduler,
//! result caching, and shutdown semantics.

const std = @import("std");
const config = @import("config.zig");
const chase_lev_deque = @import("../concurrency/chase_lev_deque.zig");
const sharded_map = @import("../concurrency/sharded_map.zig");
const workload = @import("workload.zig");

const build_options = @import("build_options");

const ProfilingModule = if (build_options.enable_profiling)
    struct {
        pub const MetricsCollector = @import("../profiling/mod.zig").MetricsCollector;
        pub const DEFAULT_METRICS_CONFIG = @import("../profiling/mod.zig").DEFAULT_METRICS_CONFIG;
    }
else
    struct {
        pub const MetricsCollector = void;
        pub const DEFAULT_METRICS_CONFIG = void{};
    };

const GPUModule = if (build_options.enable_gpu)
    struct {
        pub const GPUBackend = @import("../gpu/mod.zig").GPUBackend;
        pub const GPUManager = @import("../gpu/mod.zig").GPUManager;
        pub const GPUExecutionContext = @import("../gpu/mod.zig").GPUExecutionContext;
        pub const GPUWorkloadVTable = @import("../gpu/mod.zig").GPUWorkloadVTable;
    }
else
    struct {
        pub const GPUBackend = void;
        pub const GPUManager = void;
        pub const GPUExecutionContext = void;
        pub const GPUWorkloadVTable = void;
    };

const ChaseLevDeque = chase_lev_deque.ChaseLevDeque;
const ShardedMap = sharded_map.ShardedMap;
const ResultHandle = workload.ResultHandle;
const WorkItem = workload.WorkItem;
const ExecutionContext = workload.ExecutionContext;
const WorkloadHints = workload.WorkloadHints;
const EMPTY: u64 = 0;

const Worker = struct {
    id: u32,
    local_deque: ChaseLevDeque,
    arena: std.heap.ArenaAllocator,
    thread: ?std.Thread,
    engine: *Engine,
    timer: std.time.Timer,
};

const ResultCache = struct {
    map: ShardedMap,
    mutex: std.Thread.Mutex,
};

const WorkItemCache = struct {
    map: std.AutoHashMap(u64, *WorkItem),
    mutex: std.Thread.Mutex,

    fn put(self: *WorkItemCache, _: std.mem.Allocator, id: u64, item: *WorkItem) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.map.put(id, item);
    }

    fn fetchRemove(self: *WorkItemCache, id: u64) ?std.AutoHashMap(u64, *WorkItem).KV {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.map.fetchRemove(id);
    }

    fn count(self: *WorkItemCache) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.map.count();
    }
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    workers: []Worker,
    running: std.atomic.Value(bool),
    config: config.EngineConfig,
    next_id: std.atomic.Value(u64),
    injection_queue: ChaseLevDeque,
    result_cache: ResultCache,
    work_items: WorkItemCache,

    metrics_collector: if (build_options.enable_profiling) ?*ProfilingModule.MetricsCollector else void,
    gpu_manager: if (build_options.enable_gpu) ?GPUModule.GPUManager else void,

    pub fn init(allocator: std.mem.Allocator, cfg: config.EngineConfig) !*Engine {
        const cpu_count = std.Thread.getCpuCount() catch 1;
        const worker_count = if (cfg.worker_count == 0) @as(u32, @intCast(cpu_count)) else cfg.worker_count;

        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);

        const workers = try allocator.alloc(Worker, worker_count);

        var i: usize = 0;
        while (i < worker_count) : (i += 1) {
            const arena_allocator = std.heap.ArenaAllocator.init(allocator);
            workers[i] = Worker{
                .id = @as(u32, @intCast(i)),
                .local_deque = try ChaseLevDeque.init(allocator, cfg.queue_capacity),
                .arena = arena_allocator,
                .thread = null,
                .engine = undefined,
                .timer = std.time.Timer.start() catch unreachable,
            };
        }

        const injection_queue = try ChaseLevDeque.init(allocator, cfg.queue_capacity);
        const result_map = try ShardedMap.init(allocator, @max(4, worker_count / 2));
        const work_item_map = std.AutoHashMap(u64, *WorkItem).init(allocator);

        const metrics_collector: if (build_options.enable_profiling) ?*ProfilingModule.MetricsCollector else void = if (build_options.enable_profiling) blk: {
            const mc = try allocator.create(ProfilingModule.MetricsCollector);
            mc.* = try ProfilingModule.MetricsCollector.init(allocator, ProfilingModule.DEFAULT_METRICS_CONFIG, worker_count);
            break :blk mc;
        } else {};

        const gpu_manager: if (build_options.enable_gpu) ?GPUModule.GPUManager else void = if (build_options.enable_gpu) blk: {
            const gm = try GPUModule.GPUManager.init(allocator, GPUModule.GPUBackend.none);
            break :blk gm;
        } else {};

        self.* = .{
            .allocator = allocator,
            .workers = workers,
            .running = std.atomic.Value(bool).init(false),
            .config = cfg,
            .next_id = std.atomic.Value(u64).init(1),
            .injection_queue = injection_queue,
            .result_cache = ResultCache{
                .map = result_map,
                .mutex = std.Thread.Mutex{},
            },
            .work_items = WorkItemCache{
                .map = work_item_map,
                .mutex = std.Thread.Mutex{},
            },
            .metrics_collector = metrics_collector,
            .gpu_manager = gpu_manager,
        };

        for (self.workers) |*worker| {
            worker.engine = self;
        }

        self.running.store(true, .release);

        i = 0;
        while (i < worker_count) : (i += 1) {
            const thread = try std.Thread.spawn(.{}, workerMain, .{&self.workers[i]});
            self.workers[i].thread = thread;
        }

        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.running.store(false, .release);

        if (self.config.drain_mode == .drain) {
            var pending: usize = 0;
            while (true) {
                pending = self.countPendingTasks();
                if (pending == 0) break;
                var i: usize = 0;
                while (i < self.config.drain_spin_iterations) : (i += 1) {
                    std.atomic.spinLoopHint();
                }
            }
        }

        for (self.workers) |*worker| {
            if (worker.thread) |thread| {
                thread.join();
            }
            worker.arena.deinit();
            worker.local_deque.deinit(self.allocator);
        }

        self.allocator.free(self.workers);
        self.injection_queue.deinit(self.allocator);

        for (self.result_cache.map.shards.items) |*shard| {
            shard.mutex.lock();
            defer shard.mutex.unlock();

            var iter = shard.map.iterator();
            while (iter.next()) |entry| {
                const entry_ptr: *ResultEntry = @ptrFromInt(entry.key_ptr.*);
                if (entry_ptr.complete.load(.acquire)) {
                    if (entry_ptr.handle.vtable.destroy) |destroy_fn| {
                        destroy_fn(entry_ptr.handle.ptr, self.allocator);
                    }
                }
                self.allocator.destroy(entry_ptr);
            }
        }
        self.result_cache.map.deinit(self.allocator);

        var iter = self.work_items.map.iterator();
        while (iter.next()) |kv| {
            self.allocator.destroy(kv.value_ptr.*);
        }
        self.work_items.map.deinit();

        if (build_options.enable_profiling) {
            if (self.metrics_collector) |mc| {
                mc.deinit();
                self.allocator.destroy(mc);
            }
        }

        if (build_options.enable_gpu) {
            if (self.gpu_manager) |*gm| {
                gm.deinit();
            }
        }

        self.allocator.destroy(self);
    }

    pub fn submit(self: *Engine, item: WorkItem) !u64 {
        const id = self.next_id.fetchAdd(1, .monotonic);
        const submit_timestamp = std.time.nanoTimestamp();

        const work_item_ptr = try self.allocator.create(WorkItem);
        work_item_ptr.* = WorkItem{
            .id = id,
            .user = item.user,
            .vtable = item.vtable,
            .priority = item.priority,
            .hints = item.hints,
        };

        const entry_ptr = try self.allocator.create(ResultEntry);
        entry_ptr.* = ResultEntry{
            .task_id = id,
            .handle = undefined,
            .complete = std.atomic.Value(bool).init(false),
            .metadata = ResultMetadata{
                .worker_id = 0,
                .submit_timestamp_ns = submit_timestamp,
                .complete_timestamp_ns = 0,
                .execution_duration_ns = 0,
            },
        };

        try self.work_items.put(self.allocator, id, work_item_ptr);

        self.result_cache.mutex.lock();
        defer self.result_cache.mutex.unlock();
        try self.result_cache.map.put(id, @intFromPtr(entry_ptr));

        const worker_index = @mod(id, self.workers.len);
        try self.workers[worker_index].local_deque.pushBottom(self.allocator, id);

        return id;
    }

    pub fn poll(self: *Engine) ?*ResultEntry {
        for (self.result_cache.map.shards.items) |*shard| {
            shard.mutex.lock();
            defer shard.mutex.unlock();

            var iter = shard.map.iterator();
            while (iter.next()) |entry| {
                const result_ptr: *ResultEntry = @ptrFromInt(entry.key_ptr.*);
                if (result_ptr.complete.load(.acquire)) {
                    return result_ptr;
                }
            }
        }

        return null;
    }

    pub fn take(self: *Engine, id: u64) ?ResultHandle {
        self.result_cache.mutex.lock();
        defer self.result_cache.mutex.unlock();

        const entry_ptr = self.result_cache.map.remove(id);
        if (entry_ptr) |ptr| {
            const entry: *ResultEntry = @ptrFromInt(ptr);
            const handle = entry.handle;

            if (entry.complete.load(.acquire)) {
                if (handle.vtable.destroy) |destroy_fn| {
                    destroy_fn(handle.ptr, self.allocator);
                }
            }
            self.allocator.destroy(entry);
            return handle;
        }

        return null;
    }

    pub fn getResultMetadata(self: *Engine, id: u64) ?ResultMetadata {
        self.result_cache.mutex.lock();
        defer self.result_cache.mutex.unlock();

        if (self.result_cache.map.get(id)) |ptr| {
            const entry: *ResultEntry = @ptrFromInt(ptr);
            return entry.metadata;
        }

        return null;
    }

    fn completeResult(self: *Engine, id: u64, handle: ResultHandle) !void {
        self.result_cache.mutex.lock();
        defer self.result_cache.mutex.unlock();

        if (self.result_cache.map.get(id)) |ptr| {
            const entry: *ResultEntry = @ptrFromInt(ptr);
            entry.handle = handle;
            entry.complete.store(true, .release);
        }
    }

    fn completeResultWithMetadata(self: *Engine, id: u64, handle: ResultHandle, worker_id: u32, start_time: u64, end_time: u64) !void {
        self.result_cache.mutex.lock();
        defer self.result_cache.mutex.unlock();

        if (self.result_cache.map.get(id)) |ptr| {
            const entry: *ResultEntry = @ptrFromInt(ptr);
            entry.handle = handle;
            entry.metadata.worker_id = worker_id;
            entry.metadata.complete_timestamp_ns = end_time;
            entry.metadata.execution_duration_ns = end_time - start_time;
            entry.complete.store(true, .release);
        }
    }

    fn countPendingTasks(self: *Engine) usize {
        var count: usize = 0;

        const work_item_count = self.work_items.count();

        var iter = self.work_items.map.iterator();
        while (iter.next()) |_| {
            count += 1;
        }

        for (self.result_cache.map.shards.items) |*shard| {
            shard.mutex.lock();
            defer shard.mutex.unlock();

            var iter2 = shard.map.iterator();
            while (iter2.next()) |entry| {
                const result_ptr: *ResultEntry = @ptrFromInt(entry.key_ptr.*);
                if (result_ptr.complete.load(.acquire)) {
                    count -= 1;
                }
            }
        }

        return count + work_item_count;
    }
};

fn workerMain(worker: *Worker) void {
    while (worker.engine.running.load(.acquire)) {
        var task_id = worker.local_deque.popBottom();

        if (task_id == null) {
            task_id = trySteal(worker);
        }

        if (task_id) |id| {
            executeTask(worker, id) catch |err| {
                if (worker.engine.config.error_callback) |cb| {
                    cb(worker.id, id, err);
                } else {
                    std.debug.print("Worker {} task {} error: {}\n", .{ worker.id, id, err });
                }
            };
        } else {
            var i: usize = 0;
            var backoff: u32 = 1;
            while (i < worker.engine.config.idle_spin_iterations) : (i += 1) {
                std.atomic.spinLoopHint();
                if (i % backoff == 0 and backoff < 100) {
                    backoff *= 2;
                }
            }
        }
    }
}

fn trySteal(worker: *Worker) ?u64 {
    const engine = worker.engine;
    const num_workers = engine.workers.len;

    var steal_attempts: usize = 0;
    while (steal_attempts < num_workers) : (steal_attempts += 1) {
        const target_index = (worker.id + @as(u32, @intCast(steal_attempts))) % @as(u32, @intCast(num_workers));
        if (target_index == worker.id) continue;

        const stolen = engine.workers[@intCast(target_index)].local_deque.steal();
        if (stolen) |id| {
            return id;
        }
    }

    return null;
}

fn executeTask(worker: *Worker, task_id: u64) !void {
    const engine = worker.engine;

    const item_ptr = engine.work_items.map.get(task_id);
    if (item_ptr == null) return;

    const item = item_ptr.?;

    var ctx = ExecutionContext{
        .worker_id = worker.id,
        .arena = &worker.arena,
    };

    const start_time = worker.timer.read();

    const result_ptr: *anyopaque = if (build_options.enable_gpu)
        try executeTaskWithGpu(engine, item, &ctx)
    else
        try item.vtable.exec(item.user, &ctx, engine.allocator);

    const end_time = worker.timer.read();
    const duration_ns = end_time - start_time;

    if (build_options.enable_profiling) {
        if (engine.metrics_collector) |mc| {
            mc.recordTaskExecution(worker.id, duration_ns);
        }
    }

    const result_vtable = workload.ResultVTable{
        .destroy = struct {
            fn destroyResult(ptr: *anyopaque, a: std.mem.Allocator) void {
                _ = a;
                _ = ptr;
            }
        }.destroyResult,
    };

    const handle = ResultHandle{
        .ptr = result_ptr,
        .vtable = &result_vtable,
        .owns_memory = false,
    };

    try engine.completeResultWithMetadata(task_id, handle, worker.id, start_time, end_time);

    if (engine.work_items.fetchRemove(task_id)) |kv| {
        engine.allocator.destroy(kv.value_ptr.*);
    }

    _ = worker.arena.reset(.retain_capacity);
}

fn executeTaskWithGpu(engine: *Engine, item: *const WorkItem, ctx: *ExecutionContext) !*anyopaque {
    if (item.gpu_vtable != null and (item.hints.prefers_gpu or item.hints.requires_gpu)) {
        const gpu_vtable: *const GPUModule.GPUWorkloadVTable = @ptrCast(@alignCast(item.gpu_vtable.?));

        if (engine.gpu_manager) |*gm| {
            const gpu_ctx = GPUModule.GPUExecutionContext{
                .backend = gm.backend,
                .device_id = 0,
                .stream_id = 0,
            };

            return gpu_vtable.gpu_exec(item.user, ctx, engine.allocator, gpu_ctx) catch |err| {
                if (item.hints.requires_gpu) {
                    return err;
                }
                return item.vtable.exec(item.user, ctx, engine.allocator);
            };
        }
    }

    return item.vtable.exec(item.user, ctx, engine.allocator);
}

const ResultMetadata = struct {
    worker_id: u32,
    submit_timestamp_ns: u64,
    complete_timestamp_ns: u64,
    execution_duration_ns: u64,
};

const ResultEntry = struct {
    task_id: u64,
    handle: ResultHandle,
    complete: std.atomic.Value(bool),
    metadata: ResultMetadata,
};
