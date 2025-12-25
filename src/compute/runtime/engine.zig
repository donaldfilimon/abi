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
const ResultVTable = workload.ResultVTable;
const EMPTY: u64 = 0;

pub const ResultMetadata = struct {
    worker_id: u32,
    submitted_at_ns: u64,
    execution_duration_ns: u64,
    completed_at_ns: u64,
};

const Worker = struct {
    id: u32,
    local_deque: ChaseLevDeque,
    arena: std.heap.ArenaAllocator,
    thread: ?std.Thread,
    engine: *Engine,
    timer: std.time.Timer,
};

const ResultEntry = struct {
    handle: ResultHandle,
    complete: std.atomic.Value(bool),
    worker_id: u32,
    submitted_at_ns: u64,
    execution_duration_ns: u64,
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
                    if (entry_ptr.handle.owns_memory) {
                        entry_ptr.handle.vtable.destroy(entry_ptr.handle.ptr, self.allocator);
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

        const result_entry = try self.allocator.create(ResultEntry);
        result_entry.* = ResultEntry{
            .handle = ResultHandle{ .ptr = null, .vtable = null },
            .complete = std.atomic.Value(bool).init(false),
            .worker_id = 0,
            .submitted_at_ns = std.time.nanoTimestamp(),
            .execution_duration_ns = 0,
        };

        const entry_key: u64 = @intFromPtr(result_entry);
        try self.result_cache.map.put(self.allocator, id, entry_key);

        const work_item_copy = try self.allocator.create(WorkItem);
        work_item_copy.* = item;
        work_item_copy.id = id;
        try self.work_items.put(self.allocator, id, work_item_copy);

        try self.injection_queue.pushBottom(id);
        return id;
    }

    pub fn poll(self: *Engine) ?ResultHandle {
        var iter = self.result_cache.map.shards.iterator();
        while (iter.next()) |shard| {
            shard.mutex.lock();
            defer shard.mutex.unlock();

            var map_iter = shard.map.iterator();
            while (map_iter.next()) |entry| {
                const entry_ptr: *ResultEntry = @ptrFromInt(entry.value_ptr.*);
                if (entry_ptr.complete.load(.acquire)) {
                    const result = ResultHandle{
                        .ptr = entry_ptr.handle.ptr,
                        .vtable = entry_ptr.handle.vtable,
                    };
                    shard.map.remove(entry.key_ptr.*);
                    self.allocator.destroy(entry_ptr);
                    return result;
                }
            }
        }
        return null;
    }

            const
 {void !)    pub fn completeResultWithMetadata(self: *Engine, task_id: u64, handle: ResultHandle, worker_id: u32, duration_ns: u64, queued_ns: u64) !void {
        const entry_ptr_u64 = self.result_cache.map.get(task_id) orelse return error.ResultNotFound;
        const entry: *ResultEntry = @ptrCast(@alignCast(@as(*anyopaque, @ptrFromInt(entry_ptr_u64))));
        entry.handle = handle;
        entry.worker_id = worker_id;
        entry.execution_duration_ns = duration_ns;
        entry.complete.store(true, .release);
        _ = queued_ns;

        if (build_options.enable_profiling) {
            if (self.metrics_collector) |*mc| {
                mc.recordTaskComplete(worker_id, duration_ns);
            }
        }
    }

}                
;        pub fn getResultMetadata(self: *Engine, task_id: u64) ?ResultMetadata {
        const entry_ptr_u64 = self.result_cache.map.get(task_id) orelse return null;
        const entry: *ResultEntry = @ptrCast(@alignCast(@as(*anyopaque, @ptrFromInt(entry_ptr_u64))));
        if (!entry.complete.load(.acquire)) return null;

        return ResultMetadata{
            .worker_id = entry.worker_id,
            .submitted_at_ns = entry.submitted_at_ns,
            .execution_duration_ns = entry.execution_duration_ns,
            .completed_at_ns = 0,
        };
    }

    pub fn countPendingTasks(self: *Engine) usize {
        return self.work_items.count();
    }

    fn workerMain(worker: *Worker) void {
        while (worker.engine.running.load(.acquire)) {
            var task_id = worker.local_deque.popBottom();
            if (task_id == null) task_id = worker.engine.injection_queue.steal();
            if (task_id == null) task_id = trySteal(worker);
            if (task_id) |id| executeTask(worker, id) catch |err| {
                std.log.err("Task {} failed: {}", .{ id, err });
            };
        }
    }

    fn trySteal(worker: *Worker) ?u64 {
        const num_workers = worker.engine.workers.len;
        var i: usize = 0;
        while (i < num_workers) : (i += 1) {
            if (i == worker.id) continue;
            const task_id = worker.engine.workers[i].local_deque.steal();
            if (task_id != null and task_id.? != EMPTY) return task_id;
        }
        return null;
    }

    fn executeTask(worker: *Worker, task_id: u64) !void {
        const kv = worker.engine.work_items.fetchRemove(task_id) orelse return;
        const item = kv.value;

        const start_time = worker.timer.read();

        var ctx = ExecutionContext{
            .worker_id = worker.id,
            .arena = &worker.arena,
        };

        const result = try item.vtable.exec(item.user, &ctx, worker.engine.allocator);
        const duration_ns = worker.timer.read() - start_time;

        const result_vtable = ResultVTable{
            .destroy = item.vtable.destroy,
        };

        const handle = ResultHandle{
            .ptr = result,
            .vtable = &result_vtable,
        };

        try worker.engine.completeResultWithMetadata(task_id, handle, worker.id, duration_ns, 0);

        _ = worker.arena.reset(.retain_capacity);
    }
};
