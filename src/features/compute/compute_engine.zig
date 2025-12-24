const std = @import("std");

pub const TaskFn = *const fn (?*anyopaque) void;

pub const Task = struct {
    func: TaskFn,
    ctx: ?*anyopaque = null,

    pub fn run(self: Task) void {
        self.func(self.ctx);
    }
};

const Worker = struct {
    queue: std.ArrayList(Task),
    mutex: std.Thread.Mutex,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) !Worker {
        return .{
            .queue = try std.ArrayList(Task).initCapacity(allocator, 0),
            .mutex = std.Thread.Mutex{},
            .allocator = allocator,
        };
    }

    fn deinit(self: *Worker) void {
        self.queue.deinit(self.allocator);
    }
};

pub const ComputeEngine = struct {
    allocator: std.mem.Allocator,
    workers: []Worker,
    threads: []std.Thread,
    running: std.atomic.Value(bool),
    next_worker: std.atomic.Value(usize),
    pending_tasks: std.atomic.Value(usize),
    idle_sleep_ns: u64,

    pub fn init(allocator: std.mem.Allocator, worker_count: u32) !*ComputeEngine {
        if (worker_count == 0) return error.InvalidWorkerCount;

        const self = try allocator.create(ComputeEngine);
        errdefer allocator.destroy(self);

        const workers = try allocator.alloc(Worker, worker_count);
        errdefer allocator.free(workers);

        const threads = try allocator.alloc(std.Thread, worker_count);
        errdefer allocator.free(threads);

        self.* = .{
            .allocator = allocator,
            .workers = workers,
            .threads = threads,
            .running = std.atomic.Value(bool).init(true),
            .next_worker = std.atomic.Value(usize).init(0),
            .pending_tasks = std.atomic.Value(usize).init(0),
            .idle_sleep_ns = 1_000_000,
        };

        for (workers) |*worker| {
            worker.* = try Worker.init(allocator);
        }

        for (threads, 0..) |*thread, idx| {
            thread.* = try std.Thread.spawn(.{}, workerMain, .{ self, idx });
        }

        return self;
    }

    pub fn deinit(self: *ComputeEngine) void {
        self.running.store(false, .release);
        for (self.threads) |thread| {
            thread.join();
        }
        for (self.workers) |*worker| {
            worker.deinit();
        }
        self.allocator.free(self.threads);
        self.allocator.free(self.workers);
        self.allocator.destroy(self);
    }

    pub fn submit(self: *ComputeEngine, task: Task) !void {
        if (!self.running.load(.acquire)) return error.EngineStopped;
        const idx = self.next_worker.fetchAdd(1, .monotonic) % self.workers.len;
        var worker = &self.workers[idx];
        worker.mutex.lock();
        defer worker.mutex.unlock();
        try worker.queue.append(worker.allocator, task);
        _ = self.pending_tasks.fetchAdd(1, .monotonic);
    }

    pub fn submitBatch(self: *ComputeEngine, tasks: []const Task) !void {
        for (tasks) |task| {
            try self.submit(task);
        }
    }

    pub fn waitIdle(self: *ComputeEngine) void {
        while (self.pending_tasks.load(.acquire) != 0) {
            std.Thread.yield() catch {};
        }
    }

    fn workerMain(self: *ComputeEngine, worker_index: usize) void {
        while (self.running.load(.acquire)) {
            if (self.popLocal(worker_index)) |task| {
                task.run();
                _ = self.pending_tasks.fetchSub(1, .monotonic);
                continue;
            }
            if (self.stealTask(worker_index)) |task| {
                task.run();
                _ = self.pending_tasks.fetchSub(1, .monotonic);
                continue;
            }
            std.Thread.yield() catch {};
        }
    }

    fn popLocal(self: *ComputeEngine, worker_index: usize) ?Task {
        var worker = &self.workers[worker_index];
        worker.mutex.lock();
        defer worker.mutex.unlock();

        if (worker.queue.items.len == 0) return null;
        return worker.queue.pop();
    }

    fn stealTask(self: *ComputeEngine, thief_index: usize) ?Task {
        for (self.workers, 0..) |*worker, idx| {
            if (idx == thief_index) continue;
            worker.mutex.lock();
            if (worker.queue.items.len == 0) {
                worker.mutex.unlock();
                continue;
            }
            const task = worker.queue.orderedRemove(0);
            worker.mutex.unlock();
            return task;
        }
        return null;
    }
};

test "compute engine executes tasks" {
    var counter = std.atomic.Value(u32).init(0);

    const Context = struct {
        fn work(ctx: ?*anyopaque) void {
            const ptr: *std.atomic.Value(u32) = @ptrCast(@alignCast(ctx.?));
            _ = ptr.fetchAdd(1, .monotonic);
        }
    };

    const engine = try ComputeEngine.init(std.testing.allocator, 2);
    defer engine.deinit();

    var tasks: [4]Task = undefined;
    for (&tasks) |*task| {
        task.* = .{ .func = Context.work, .ctx = &counter };
        try engine.submit(task.*);
    }

    engine.waitIdle();
    try std.testing.expectEqual(@as(u32, 4), counter.load(.acquire));
}
