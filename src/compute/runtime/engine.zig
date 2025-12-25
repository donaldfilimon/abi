//! Compute engine - core runtime
//!
//! Main execution engine with work-stealing scheduler,
//! result caching, and shutdown semantics.

const std = @import("std");
const config = @import("config.zig");
const builtin = @import("builtin");

fn sleep(ns: u64) void {
    if (builtin.os.tag == .windows) {
        _ = std.os.windows.kernel32.SleepEx(@intCast(ns / std.time.ns_per_ms), 0);
    } else {
        const s = ns / std.time.ns_per_s;
        const n = ns % std.time.ns_per_s;
        _ = std.posix.nanosleep(&.{ .sec = @intCast(s), .nsec = @intCast(n) }, null);
    }
}

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

// Use generic ShardedMap for type safety
const ResultCache = ShardedMap(u64, *ResultEntry);
const WorkItemCache = ShardedMap(u64, *WorkItem);

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
        const shard_count = @max(4, worker_count * 2); // More shards for better concurrency
        const result_map = try ResultCache.init(allocator, shard_count);
        const work_item_map = try WorkItemCache.init(allocator, shard_count);

        const metrics_collector = if (build_options.enable_profiling) blk: {
            const mc = try allocator.create(ProfilingModule.MetricsCollector);
            mc.* = try ProfilingModule.MetricsCollector.init(allocator, ProfilingModule.DEFAULT_METRICS_CONFIG, worker_count);
            break :blk mc;
        } else {};

        const gpu_manager = if (build_options.enable_gpu) blk: {
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
            .result_cache = result_map,
            .work_items = work_item_map,
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

        // Clean up results
        for (self.result_cache.shards) |*shard| {
            shard.mutex.lock();
            defer shard.mutex.unlock();

            var iter = shard.map.iterator();
            while (iter.next()) |entry| {
                const entry_ptr = entry.value_ptr.*;
                if (entry_ptr.complete.load(.acquire)) {
                    if (entry_ptr.handle.owns_memory) {
                        entry_ptr.handle.vtable.destroy(entry_ptr.handle.ptr, self.allocator);
                    }
                }
                self.allocator.destroy(entry_ptr);
            }
        }
        self.result_cache.deinit();

        // Clean up pending work items
        for (self.work_items.shards) |*shard| {
            shard.mutex.lock();
            defer shard.mutex.unlock();
            var iter = shard.map.iterator();
            while (iter.next()) |kv| {
                self.allocator.destroy(kv.value_ptr.*);
            }
        }
        self.work_items.deinit();

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

        try self.result_cache.put(id, result_entry);

        const work_item_copy = try self.allocator.create(WorkItem);
        work_item_copy.* = item;
        work_item_copy.id = id;
        
        try self.work_items.put(id, work_item_copy);
        try self.injection_queue.pushBottom(id);
        
        return id;
    }

    pub fn take(self: *Engine, task_id: u64) ?ResultHandle {
        const Predicate = struct {
            fn check(_: void, entry: *ResultEntry) bool {
                return entry.complete.load(.acquire);
            }
        };

        if (self.result_cache.fetchRemoveIf(task_id, {}, Predicate.check)) |entry| {
            const handle = entry.handle;
            self.allocator.destroy(entry);
            return handle;
        }
        return null;
    }

    pub fn poll(self: *Engine) ?ResultHandle {
        // Iterate over shards to find a completed result
        // Note: This is inefficient for checking specific results, but fine for "any result" polling
        // For specific result checking, we should add a `getResult(id)` method.
        for (self.result_cache.shards) |*shard| {
            shard.mutex.lock();
            defer shard.mutex.unlock();

            var map_iter = shard.map.iterator();
            while (map_iter.next()) |entry| {
                const entry_ptr = entry.value_ptr.*;
                if (entry_ptr.complete.load(.acquire)) {
                    const result = ResultHandle{
                        .ptr = entry_ptr.handle.ptr,
                        .vtable = entry_ptr.handle.vtable,
                        .owns_memory = entry_ptr.handle.owns_memory,
                    };
                    _ = shard.map.remove(entry.key_ptr.*);
                    self.allocator.destroy(entry_ptr);
                    return result;
                }
            }
        }
        return null;
    }

    pub fn completeResultWithMetadata(self: *Engine, task_id: u64, handle: ResultHandle, worker_id: u32, duration_ns: u64, queued_ns: u64) !void {
        const entry = self.result_cache.get(task_id) orelse return error.ResultNotFound;
        
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

    pub fn getResultMetadata(self: *Engine, task_id: u64) ?ResultMetadata {
        const entry = self.result_cache.get(task_id) orelse return null;
        if (!entry.complete.load(.acquire)) return null;

        return ResultMetadata{
            .worker_id = entry.worker_id,
            .submitted_at_ns = entry.submitted_at_ns,
            .execution_duration_ns = entry.execution_duration_ns,
            .completed_at_ns = 0,
        };
    }

    pub fn countPendingTasks(self: *Engine) usize {
        var count: usize = 0;
        for (self.work_items.shards) |*shard| {
            shard.mutex.lock();
            defer shard.mutex.unlock();
            count += shard.map.count();
        }
        return count;
    }

    fn workerMain(worker: *Worker) void {
        var backoff: u32 = 0;
        while (worker.engine.running.load(.acquire)) {
            var task_id = worker.local_deque.popBottom();
            if (task_id == null) task_id = worker.engine.injection_queue.steal();
            if (task_id == null) task_id = trySteal(worker);
            
            if (task_id) |id| {
                // Found work, reset backoff
                backoff = 0;
                executeTask(worker, id) catch |err| {
                    std.log.err("Task {} failed: {}", .{ id, err });
                };
            } else {
                // Exponential backoff to save CPU when idle
                if (backoff < 10) {
                    std.atomic.spinLoopHint();
                } else if (backoff < 20) {
                    std.Thread.yield() catch {};
                } else {
                    // Sleep for 1ms if really idle
                    sleep(1 * std.time.ns_per_ms);
                }
                if (backoff < 100) backoff += 1;
            }
        }
    }

    fn trySteal(worker: *Worker) ?u64 {
        const num_workers = worker.engine.workers.len;
        var i: usize = 0;
        // Start stealing from a random-ish neighbor to avoid contention
        const start_idx = (worker.id + 1) % num_workers;
        var current_idx = start_idx;

        while (i < num_workers) : (i += 1) {
            if (current_idx != worker.id) {
                const task_id = worker.engine.workers[current_idx].local_deque.steal();
                if (task_id != null and task_id.? != EMPTY) return task_id;
            }
            current_idx = (current_idx + 1) % num_workers;
        }
        return null;
    }

    fn executeTask(worker: *Worker, task_id: u64) !void {
        // Remove from work items map
        const item = worker.engine.work_items.remove(task_id) orelse return;

        const start_time = worker.timer.read();

        var ctx = ExecutionContext{
            .worker_id = worker.id,
            .arena = &worker.arena,
        };

        // Execute workload
        const result = try item.vtable.exec(item.user, &ctx, worker.engine.allocator);
        const duration_ns = worker.timer.read() - start_time;

        const result_vtable = ResultVTable{
            .destroy = item.vtable.destroy,
        };

        const handle = ResultHandle{
            .ptr = result,
            .vtable = &result_vtable,
            .owns_memory = true, // Engine owns result until taken
        };

        // Complete the result
        try worker.engine.completeResultWithMetadata(task_id, handle, worker.id, duration_ns, 0);

        // Clean up the work item definition
        worker.engine.allocator.destroy(item);

        // Reset arena for next task
        _ = worker.arena.reset(.retain_capacity);
    }
};