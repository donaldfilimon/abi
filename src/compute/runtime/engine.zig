//! Minimal distributed compute engine for running synchronous tasks.
const std = @import("std");

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
        _ = std.Thread.yield() catch {};
    }

    pub fn wait(self: *Backoff) void {
        self.spins += 1;
        const iterations = @min(self.spins, 64);
        var i: usize = 0;
        while (i < iterations) : (i += 1) {
            std.atomic.spinLoopHint();
        }
        if (self.spins > 32) {
            _ = std.Thread.yield() catch {};
        }
    }
};

fn nowMilliseconds() i64 {
    var global_timer: ?std.time.Timer = null;
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
};

pub const TaskId = u64;

const DEFAULT_MAX_TASKS: usize = 1024;

pub const EngineConfig = struct {
    max_tasks: usize = DEFAULT_MAX_TASKS,
};

const ResultKind = enum {
    value,
    owned_slice,
};

const ResultBlob = struct {
    kind: ResultKind,
    bytes: []u8,
    size: usize,
};

pub const DistributedComputeEngine = struct {
    allocator: std.mem.Allocator,
    config: EngineConfig,
    next_id: TaskId = 1,
    results: std.AutoHashMap(TaskId, ResultBlob),

    pub fn init(allocator: std.mem.Allocator, config: EngineConfig) !DistributedComputeEngine {
        return .{
            .allocator = allocator,
            .config = config,
            .results = std.AutoHashMap(TaskId, ResultBlob).init(allocator),
        };
    }

    pub fn deinit(self: *DistributedComputeEngine) void {
        var it = self.results.valueIterator();
        while (it.next()) |blob| {
            self.allocator.free(blob.bytes);
        }
        self.results.deinit();
        self.* = undefined;
    }

    pub fn submit_task(
        self: *DistributedComputeEngine,
        comptime ResultType: type,
        task: anytype,
    ) !TaskId {
        if (self.results.count() >= self.config.max_tasks) return EngineError.QueueFull;

        const result = try callTask(ResultType, task, self.allocator);
        const id = self.next_id;
        self.next_id += 1;
        try self.storeResult(ResultType, id, result);
        return id;
    }

    pub fn wait_for_result(
        self: *DistributedComputeEngine,
        comptime ResultType: type,
        id: TaskId,
        timeout_ms: u64,
    ) !ResultType {
        const start_ms: i64 = nowMilliseconds();
        const deadline_ms: ?i64 = if (timeout_ms == 0)
            start_ms
        else
            start_ms + @as(i64, @intCast(timeout_ms));
        var backoff = Backoff{};
        while (true) {
            if (self.results.fetchRemove(id)) |entry| {
                return self.decodeResult(ResultType, entry.value);
            }

            if (timeout_ms == 0) return EngineError.Timeout;
            if (deadline_ms == null) return EngineError.ResultNotFound;
            if (nowMilliseconds() >= deadline_ms.?) return EngineError.Timeout;
            backoff.wait();
        }
    }

    fn storeResult(
        self: *DistributedComputeEngine,
        comptime ResultType: type,
        id: TaskId,
        result: ResultType,
    ) !void {
        if (comptime isByteSlice(ResultType)) {
            const slice: []const u8 = result;
            const copy = try self.allocator.dupe(u8, slice);
            errdefer self.allocator.free(copy);
            try self.results.put(id, .{
                .kind = .owned_slice,
                .bytes = copy,
                .size = copy.len,
            });
            return;
        }

        const size = @sizeOf(ResultType);
        const copy = try self.allocator.alloc(u8, size);
        errdefer self.allocator.free(copy);
        std.mem.copyForwards(u8, copy, std.mem.asBytes(&result));
        try self.results.put(id, .{
            .kind = .value,
            .bytes = copy,
            .size = size,
        });
    }

    fn decodeResult(
        self: *DistributedComputeEngine,
        comptime ResultType: type,
        blob: ResultBlob,
    ) !ResultType {
        if (comptime isByteSlice(ResultType)) {
            if (blob.kind != .owned_slice) return EngineError.UnsupportedResultType;
            return @as(ResultType, blob.bytes);
        }

        if (blob.kind != .value or blob.size != @sizeOf(ResultType)) {
            self.allocator.free(blob.bytes);
            return EngineError.UnsupportedResultType;
        }

        var value: ResultType = undefined;
        std.mem.copyForwards(u8, std.mem.asBytes(&value), blob.bytes);
        self.allocator.free(blob.bytes);
        return value;
    }
};

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
    var engine = try DistributedComputeEngine.init(allocator, .{ .max_tasks = 8 });
    defer engine.deinit();

    const task_id = try engine.submit_task(u32, sampleTask);
    const result = try engine.wait_for_result(u32, task_id, 1000);
    try std.testing.expectEqual(@as(u32, 42), result);
}

test "engine reports timeout for non-existent result with zero timeout" {
    const allocator = std.testing.allocator;
    var engine = try DistributedComputeEngine.init(allocator, .{ .max_tasks = 4 });
    defer engine.deinit();

    try std.testing.expectError(EngineError.Timeout, engine.wait_for_result(u32, 999, 0));
}

test "engine reports queue full when max tasks reached" {
    const allocator = std.testing.allocator;
    var engine = try DistributedComputeEngine.init(allocator, .{ .max_tasks = 2 });
    defer engine.deinit();

    _ = try engine.submit_task(u32, sampleTask);
    _ = try engine.submit_task(u32, sampleTask);

    try std.testing.expectError(EngineError.QueueFull, engine.submit_task(u32, sampleTask));
}

test "engine handles byte slice results" {
    const alloc = std.testing.allocator;
    var engine = try DistributedComputeEngine.init(alloc, .{ .max_tasks = 8 });
    defer alloc.deinit();

    const task_id = try engine.submit_task([]const u8, byteSliceTask);
    const result = try engine.wait_for_result([]const u8, task_id, 100);
    try std.testing.expectEqual(@as(usize, 4), result.len);
    alloc.free(result);
}

test "engine handles slice results directly" {
    const alloc = std.testing.allocator;
    var engine = try DistributedComputeEngine.init(alloc, .{ .max_tasks = 8 });
    defer alloc.deinit();

    const test_data: []const u8 = "test";
    const task_id = try engine.submit_task([]const u8, struct {
        pub fn execute(_: @This(), allocator: std.mem.Allocator) ![]const u8 {
            return allocator.dupe(u8, test_data);
        }
    }{});
    const result = try engine.wait_for_result([]const u8, task_id, 100);
    try std.testing.expectEqual(@as(usize, 4), result.len);
    alloc.free(result);
}

fn sampleTask(_: std.mem.Allocator) !u32 {
    return 42;
}

fn byteSliceTask(allocator: std.mem.Allocator) ![]const u8 {
    return allocator.dupe(u8, "test");
}

test "engine returns timeout when result not ready within time" {
    const allocator = std.testing.allocator;
    var engine = try DistributedComputeEngine.init(allocator, .{ .max_tasks = 4 });
    defer engine.deinit();

    try std.testing.expectError(EngineError.Timeout, engine.wait_for_result(u32, 999, 100));
}
