//! Compute engine - core runtime
//!
//! Main execution engine with work-stealing scheduler,
//! result caching, and shutdown semantics.

const std = @import("std");
const config = @import("config.zig");
const chase_lev_deque = @import("../concurrency/chase_lev_deque.zig");
const sharded_map = @import("../concurrency/sharded_map.zig");
const workload = @import("workload.zig");

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
};

const ResultCache = struct {
    map: ShardedMap,
    mutex: std.Thread.Mutex,
};

const WorkItemCache = struct {
    map: std.AutoHashMap(u64, *WorkItem),
    mutex: std.Thread.Mutex,
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

    pub fn init(allocator: std.mem.Allocator, cfg: config.EngineConfig) !*Engine {
        const cpu_count = std.Thread.getCpuCount() catch 1;
        const worker_count = if (cfg.worker_count == 0) @as(u32, @intCast(cpu_count)) else cfg.worker_count;

        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);

        const workers = try allocator.alloc(Worker, worker_count);

        const queue_capacity = 1024;

        var i: usize = 0;
        while (i < worker_count) : (i += 1) {
            const arena_allocator = std.heap.ArenaAllocator.init(allocator);
            workers[i] = Worker{
                .id = @as(u32, @intCast(i)),
                .local_deque = try ChaseLevDeque.init(allocator, queue_capacity),
                .arena = arena_allocator,
                .thread = null,
                .engine = undefined,
            };
        }

        const injection_queue = try ChaseLevDeque.init(allocator, queue_capacity);
        const result_map = try ShardedMap.init(allocator, @max(4, worker_count / 2));
        const work_item_map = std.AutoHashMap(u64, *WorkItem).init(allocator);

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
                while (i < 1000) : (i += 1) {
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
        self.result_cache.map.deinit(self.allocator);
        self.work_items.map.deinit();
        self.allocator.destroy(self);
    }

    pub fn submit(self: *Engine, item: WorkItem) !u64 {
        const id = self.next_id.fetchAdd(1, .monotonic);
        var timer = try std.time.Timer.start();
        const submit_timestamp = timer.read();

        const work_item_ptr = try self.allocator.create(WorkItem);
        work_item_ptr.* = WorkItem{
            .id = id,
            .user = item.user,
            .vtable = item.vtable,
            .priority = item.priority,
            .hints = item.hints,
        };

        const entry = try self.allocator.create(ResultEntry);
        entry.* = ResultEntry{
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

        self.work_items.mutex.lock();
        defer self.work_items.mutex.unlock();
        try self.work_items.map.put(id, work_item_ptr);

        self.result_cache.mutex.lock();
        defer self.result_cache.mutex.unlock();
        try self.result_cache.map.put(id, @intFromPtr(entry));

        const worker_index = @mod(id, self.workers.len);
        try self.workers[worker_index].local_deque.pushBottom(self.allocator, id);

        return id;
    }

    pub fn poll(self: *Engine) ?*ResultEntry {
        self.result_cache.mutex.lock();
        defer self.result_cache.mutex.unlock();

        for (self.result_cache.map.shards.items) |*shard| {
            shard.mutex.lock();
            defer shard.mutex.unlock();

            var iter = shard.map.iterator();
            while (iter.next()) |entry| {
                const result_ptr = @as(*ResultEntry, @ptrFromInt(@as(usize, @intCast(entry.key_ptr.*))));
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
            const entry = @as(*ResultEntry, @ptrFromInt(@as(usize, @intCast(ptr))));
            return entry.handle;
        }

        return null;
    }

    pub fn getResultMetadata(self: *Engine, id: u64) ?ResultMetadata {
        self.result_cache.mutex.lock();
        defer self.result_cache.mutex.unlock();

        if (self.result_cache.map.get(id)) |ptr| {
            const entry = @as(*ResultEntry, @ptrFromInt(@as(usize, @intCast(ptr))));
            return entry.metadata;
        }

        return null;
    }

    fn completeResult(self: *Engine, id: u64, handle: ResultHandle) !void {
        self.result_cache.mutex.lock();
        defer self.result_cache.mutex.unlock();

        if (self.result_cache.map.get(id)) |ptr| {
            const entry = @as(*ResultEntry, @ptrFromInt(@as(usize, @intCast(ptr))));
            entry.handle = handle;
            entry.complete.store(true, .release);
        }
    }

    fn completeResultWithMetadata(self: *Engine, id: u64, handle: ResultHandle, worker_id: u32, start_time: u64, end_time: u64) !void {
        self.result_cache.mutex.lock();
        defer self.result_cache.mutex.unlock();

        if (self.result_cache.map.get(id)) |ptr| {
            const entry = @as(*ResultEntry, @ptrFromInt(@as(usize, @intCast(ptr))));
            entry.handle = handle;
            entry.metadata.worker_id = worker_id;
            entry.metadata.complete_timestamp_ns = end_time;
            entry.metadata.execution_duration_ns = end_time - start_time;
            entry.complete.store(true, .release);
        }
    }

    fn countPendingTasks(self: *Engine) usize {
        var count: usize = 0;

        self.work_items.mutex.lock();
        defer self.work_items.mutex.unlock();

        var iter = self.work_items.map.iterator();
        while (iter.next()) |_| {
            count += 1;
        }

        self.result_cache.mutex.lock();
        defer self.result_cache.mutex.unlock();

        for (self.result_cache.map.shards.items) |*shard| {
            shard.mutex.lock();
            defer shard.mutex.unlock();

            var iter2 = shard.map.iterator();
            while (iter2.next()) |entry| {
                const result_ptr = @as(*ResultEntry, @ptrFromInt(@as(usize, @intCast(entry.key_ptr.*))));
                if (result_ptr.complete.load(.acquire)) {
                    count -= 1;
                }
            }
        }

        return count;
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
                std.debug.print("Worker {} task {} error: {}\n", .{ worker.id, id, err });
            };
        } else {
            var i: usize = 0;
            while (i < 100) : (i += 1) {
                std.atomic.spinLoopHint();
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

    engine.work_items.mutex.lock();
    const item_ptr = engine.work_items.map.get(task_id);
    engine.work_items.mutex.unlock();

    if (item_ptr == null) return;

    const item = item_ptr.?;

    var ctx = ExecutionContext{
        .worker_id = worker.id,
        .arena = &worker.arena,
    };

    var exec_timer = try std.time.Timer.start();
    const start_time = exec_timer.read();
    const result_ptr = try item.vtable.exec(item.user, &ctx, engine.allocator);
    const end_time = exec_timer.read();

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
    };

    try engine.completeResultWithMetadata(task_id, handle, worker.id, start_time, end_time);

    engine.work_items.mutex.lock();
    _ = engine.work_items.map.remove(task_id);
    engine.work_items.mutex.unlock();

    _ = worker.arena.reset(.retain_capacity);
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
