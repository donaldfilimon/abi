const std = @import("std");
const time = @import("../../shared/utils/time.zig");

pub const EngineError = error{
    ResultNotFound,
    Timeout,
    UnsupportedResultType,
    QueueFull,
};

pub const TaskId = u64;

pub const EngineConfig = struct {
    max_tasks: usize = 1024,
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

    pub fn submit_task(self: *DistributedComputeEngine, comptime ResultType: type, task: anytype) !TaskId {
        if (self.results.count() >= self.config.max_tasks) return EngineError.QueueFull;

        const result = try callTask(ResultType, task, self.allocator);
        const id = self.next_id;
        self.next_id += 1;
        try self.storeResult(ResultType, id, result);
        return id;
    }

    pub fn wait_for_result(self: *DistributedComputeEngine, comptime ResultType: type, id: TaskId, timeout_ms: u64) !ResultType {
        const start_ms: i64 = time.nowMilliseconds();
        while (true) {
            if (self.results.fetchRemove(id)) |entry| {
                return self.decodeResult(ResultType, entry.value);
            }

            if (timeout_ms == 0) return EngineError.ResultNotFound;
            const elapsed = time.nowMilliseconds() - start_ms;
            if (elapsed >= @as(i64, @intCast(timeout_ms))) return EngineError.Timeout;
            std.atomic.spinLoopHint();
        }
    }

    fn storeResult(self: *DistributedComputeEngine, comptime ResultType: type, id: TaskId, result: ResultType) !void {
        if (comptime isByteSlice(ResultType)) {
            const slice: []const u8 = result;
            const copy = try self.allocator.dupe(u8, slice);
            try self.results.put(id, .{
                .kind = .owned_slice,
                .bytes = copy,
                .size = copy.len,
            });
            return;
        }

        const size = @sizeOf(ResultType);
        const copy = try self.allocator.alloc(u8, size);
        std.mem.copyForwards(u8, copy, std.mem.asBytes(&result));
        try self.results.put(id, .{
            .kind = .value,
            .bytes = copy,
            .size = size,
        });
    }

    fn decodeResult(self: *DistributedComputeEngine, comptime ResultType: type, blob: ResultBlob) !ResultType {
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
    const result = try engine.wait_for_result(u32, task_id, 0);
    try std.testing.expectEqual(@as(u32, 42), result);
}

fn sampleTask(_: std.mem.Allocator) !u32 {
    return 42;
}
