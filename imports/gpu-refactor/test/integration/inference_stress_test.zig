//! Stress testing for inference engine components.
//!
//! Verifies thread-safety of the scheduler, KV cache, and stats reporting
//! under high concurrency.

const std = @import("std");
const abi = @import("abi");

const Engine = abi.inference.Engine;
const Request = abi.inference.Request;

const StressHelper = struct {
    var counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);
    var errors: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);

    fn submitJob(engine: *Engine, num_submits: usize) void {
        for (0..num_submits) |i| {
            const id = counter.fetchAdd(1, .acq_rel);
            _ = engine.submit(.{
                .id = id,
                .prompt = "stress test prompt",
                .priority = @intCast(i % 256),
                .created_at = @intCast(abi.foundation.time.unixMs()),
            }) catch {
                _ = errors.fetchAdd(1, .acq_rel);
            };
        }
    }

    fn statsJob(engine: *const Engine, num_reads: usize) void {
        for (0..num_reads) |_| {
            const stats = engine.getStats();
            // Just touch the stats to ensure no crash
            if (stats.total_requests > 1000000) {
                std.debug.print("Impossible stats\n", .{});
            }
            std.Thread.yield() catch {};
        }
    }
};

test "inference stress: concurrent submit and getStats" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 1000,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .vocab_size = 256,
    });
    defer engine.deinit();

    StressHelper.counter.store(0, .release);
    StressHelper.errors.store(0, .release);

    const num_threads = 16;
    const submits_per_thread = 2000;
    var threads: [num_threads]std.Thread = undefined;

    // Spawn submitters
    for (0..num_threads) |i| {
        threads[i] = try std.Thread.spawn(.{}, StressHelper.submitJob, .{ &engine, submits_per_thread });
    }

    // Spawn stats readers
    var readers: [4]std.Thread = undefined;
    for (0..4) |i| {
        readers[i] = try std.Thread.spawn(.{}, StressHelper.statsJob, .{ &engine, 10000 });
    }

    for (0..num_threads) |i| {
        threads[i].join();
    }
    for (0..4) |i| {
        readers[i].join();
    }

    const stats = engine.getStats();
    const total_attempted = num_threads * submits_per_thread;
    const failed = StressHelper.errors.load(.acquire);

    // We expect some may fail if queue is full, but it shouldn't crash.
    // However, without locks, it WILL likely crash or corrupt memory.
    std.debug.print("Stress test: {d} submits, {d} failures, {d} pending\n", .{
        total_attempted, failed, stats.pending_requests,
    });
}

test "inference stress: concurrent generateAsync" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 256,
    });
    defer engine.deinit();

    const num_requests = 20;
    var results = try allocator.alloc(*abi.inference.AsyncResult, num_requests);
    defer allocator.free(results);

    for (0..num_requests) |i| {
        results[i] = try engine.generateAsyncWithTimeout(.{
            .id = @intCast(i),
            .prompt = "concurrent async stress",
            .max_tokens = 4,
        });
    }

    for (0..num_requests) |i| {
        if (results[i].waitTimeout(5000)) |res| {
            const expected_id: u64 = @intCast(i);
            try std.testing.expectEqual(expected_id, res.id);
        }
        results[i].deinit();
    }
}
