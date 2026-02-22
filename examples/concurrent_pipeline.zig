//! Concurrent Pipeline Example
//!
//! Demonstrates a pipeline built from runtime primitives:
//! - `abi.runtime.Channel` for stage handoff/backpressure
//! - `abi.runtime.ThreadPool` for parallel stage work
//! - `abi.runtime.DagPipeline` for stage dependency orchestration
//!
//! Run with: `zig build run-concurrent-pipeline`

const std = @import("std");
const abi = @import("abi");

const U64Channel = abi.runtime.Channel(u64);
const ThreadPool = abi.runtime.ThreadPool;
const DagPipeline = abi.runtime.DagPipeline;
const primitives = abi.shared.utils.primitives;

const ProduceCtx = struct {
    pool: *ThreadPool,
    output: *U64Channel,
    source: []const u64,
};

const TransformCtx = struct {
    pool: *ThreadPool,
    input: *U64Channel,
    output: *U64Channel,
};

const ReduceCtx = struct {
    input: *U64Channel,
    sum: *u64,
};

fn asCtx(comptime T: type, ptr: ?*anyopaque) ?*T {
    const raw = ptr orelse return null;
    return @ptrCast(@alignCast(raw));
}

fn produceStage(raw_ctx: ?*anyopaque) bool {
    const ctx = asCtx(ProduceCtx, raw_ctx) orelse return false;

    const send_task = struct {
        fn run(output: *U64Channel, value: u64) void {
            output.send(value) catch {};
        }
    }.run;

    for (ctx.source) |value| {
        if (!ctx.pool.schedule(send_task, .{ ctx.output, value })) {
            ctx.output.send(value) catch return false;
        }
    }

    ctx.pool.waitIdle();
    ctx.output.close();
    return true;
}

fn transformStage(raw_ctx: ?*anyopaque) bool {
    const ctx = asCtx(TransformCtx, raw_ctx) orelse return false;

    const transform_task = struct {
        fn run(output: *U64Channel, value: u64) void {
            const squared = value * value;
            output.send(squared) catch {};
        }
    }.run;

    while (true) {
        if (ctx.input.tryRecv()) |value| {
            if (!ctx.pool.schedule(transform_task, .{ ctx.output, value })) {
                const squared = value * value;
                ctx.output.send(squared) catch return false;
            }
            continue;
        }

        if (ctx.input.isClosed() and ctx.input.isEmpty()) break;
        std.Thread.yield() catch {};
    }

    ctx.pool.waitIdle();
    ctx.output.close();
    return true;
}

fn reduceStage(raw_ctx: ?*anyopaque) bool {
    const ctx = asCtx(ReduceCtx, raw_ctx) orelse return false;
    var total: u64 = 0;

    while (true) {
        if (ctx.input.tryRecv()) |value| {
            total +%= value;
            continue;
        }

        if (ctx.input.isClosed() and ctx.input.isEmpty()) break;
        std.Thread.yield() catch {};
    }

    ctx.sum.* = total;
    return true;
}

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== ABI Concurrent Pipeline Demo ===\n", .{});
    std.debug.print("Platform: {s}\n", .{primitives.Platform.description()});

    var pool = try ThreadPool.init(allocator, .{ .thread_count = 4 });
    defer pool.deinit();

    var stage_one = try U64Channel.init(allocator, 64);
    defer stage_one.deinit();

    var stage_two = try U64Channel.init(allocator, 64);
    defer stage_two.deinit();

    const input = [_]u64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var reduced_sum: u64 = 0;

    var produce_ctx = ProduceCtx{
        .pool = pool,
        .output = &stage_one,
        .source = input[0..],
    };
    var transform_ctx = TransformCtx{
        .pool = pool,
        .input = &stage_one,
        .output = &stage_two,
    };
    var reduce_ctx = ReduceCtx{
        .input = &stage_two,
        .sum = &reduced_sum,
    };

    var pipeline = DagPipeline.init();
    const produce_id = try pipeline.addStage("produce", .input);
    const transform_id = try pipeline.addStage("transform", .compute);
    const reduce_id = try pipeline.addStage("reduce", .output);

    try pipeline.addDependency(transform_id, produce_id);
    try pipeline.addDependency(reduce_id, transform_id);

    pipeline.bindExecutor(produce_id, produceStage, &produce_ctx);
    pipeline.bindExecutor(transform_id, transformStage, &transform_ctx);
    pipeline.bindExecutor(reduce_id, reduceStage, &reduce_ctx);

    const result = try pipeline.execute();
    if (!result.success) return error.PipelineExecutionFailed;

    const expected_sum: u64 = 385; // 1^2 + ... + 10^2
    if (reduced_sum != expected_sum) {
        std.debug.print("Unexpected reduce result: {d} (expected {d})\n", .{
            reduced_sum,
            expected_sum,
        });
        return error.UnexpectedResult;
    }

    const pool_stats = pool.stats();
    const stage_one_stats = stage_one.stats();
    const stage_two_stats = stage_two.stats();

    std.debug.print("Stages run: {d}/{d}, failed: {d}\n", .{
        result.stages_run,
        result.stages_total,
        result.stages_failed,
    });
    std.debug.print("Reduce output: {d}\n", .{reduced_sum});
    std.debug.print("ThreadPool submitted/completed: {d}/{d}\n", .{
        pool_stats.tasks_submitted,
        pool_stats.tasks_completed,
    });
    std.debug.print("Channel stage_one sent/recv: {d}/{d}\n", .{
        stage_one_stats.total_sent,
        stage_one_stats.total_received,
    });
    std.debug.print("Channel stage_two sent/recv: {d}/{d}\n", .{
        stage_two_stats.total_sent,
        stage_two_stats.total_received,
    });
}
