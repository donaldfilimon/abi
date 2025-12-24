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

pub const Engine = struct {
    allocator: std.mem.Allocator,
    workers: []Worker,
    running: std.atomic.Value(bool),
    config: config.EngineConfig,
    next_id: std.atomic.Value(u64),
    injection_queue: ChaseLevDeque,
    result_cache: ResultCache,

    const Worker = struct {
        id: u32,
        local_deque: ChaseLevDeque,
    };

    const ResultCache = struct {
        map: ShardedMap,
        mutex: std.Thread.Mutex,
    };

    pub fn init(allocator: std.mem.Allocator, cfg: config.EngineConfig) !*Engine {
        const cpu_count = std.Thread.getCpuCount() catch 1;
        const worker_count = if (cfg.worker_count == 0) @as(u32, @intCast(cpu_count)) else cfg.worker_count;

        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);

        const workers = try allocator.alloc(Worker, worker_count);

        const queue_capacity = 1024;

        var i: usize = 0;
        while (i < worker_count) : (i += 1) {
            workers[i] = Worker{
                .id = @as(u32, @intCast(i)),
                .local_deque = try ChaseLevDeque.init(allocator, queue_capacity),
            };
        }

        const injection_queue = try ChaseLevDeque.init(allocator, queue_capacity);
        const result_map = try ShardedMap.init(allocator, @max(4, worker_count / 2));

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
        };

        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.running.store(false, .release);

        for (self.workers) |*worker| {
            worker.local_deque.deinit(self.allocator);
        }
        self.allocator.free(self.workers);
        self.injection_queue.deinit(self.allocator);
        self.result_cache.map.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn submit(self: *Engine, item: WorkItem) !u64 {
        _ = item;
        const id = self.next_id.fetchAdd(1, .monotonic);

        const entry = try self.allocator.create(ResultEntry);
        entry.* = ResultEntry{
            .task_id = id,
            .handle = undefined,
            .complete = std.atomic.Value(bool).init(false),
        };

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

            var iter = shard.map.map.iterator();
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

    pub fn completeResult(self: *Engine, id: u64, handle: ResultHandle) !void {
        self.result_cache.mutex.lock();
        defer self.result_cache.mutex.unlock();

        if (self.result_cache.map.get(id)) |ptr| {
            const entry = @as(*ResultEntry, @ptrFromInt(@as(usize, @intCast(ptr))));
            entry.handle = handle;
            entry.complete.store(true, .release);
        }
    }
};

const ResultEntry = struct {
    task_id: u64,
    handle: ResultHandle,
    complete: std.atomic.Value(bool),
};
