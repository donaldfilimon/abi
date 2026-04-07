//! Integration Tests: Inference Engine Multi-Backend
//!
//! Tests the inference engine with demo, connector, and local backends,
//! along with a deep integration test simulating an end-to-end workload.

const std = @import("std");
const abi = @import("abi");
const time_mod = abi.foundation.time;

const Engine = abi.inference.Engine;
const Backend = abi.inference.Backend;
const Result = abi.inference.Result;

test "inference: demo backend generates text" {
    var engine = try Engine.init(std.testing.allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .vocab_size = 256,
        .backend = .demo,
    });
    defer engine.deinit();

    const result = try engine.generate(.{
        .id = 1,
        .prompt = "Explain HNSW indexing",
        .max_tokens = 10,
    });
    defer result.deinit(std.testing.allocator);

    try std.testing.expect(result.text.len > 0);
    try std.testing.expect(result.completion_tokens > 0);
    try std.testing.expectEqual(Backend.demo, engine.getStats().backend);
}

test "inference: connector backend returns model-tagged response" {
    var engine = try Engine.init(std.testing.allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .backend = .connector,
        .model_id = "claude-3-sonnet",
    });
    defer engine.deinit();

    const result = try engine.generate(.{
        .id = 1,
        .prompt = "What is a vector database?",
        .max_tokens = 50,
    });
    defer result.deinit(std.testing.allocator);

    try std.testing.expect(result.text.len > 0);
    // model_id "claude-3-sonnet" has no slash, so provider resolves to "echo"
    try std.testing.expect(std.mem.indexOf(u8, result.text, "[echo/claude-3-sonnet]") != null);
    try std.testing.expectEqual(Backend.connector, engine.getStats().backend);
}

test "inference: local backend falls back to demo" {
    var engine = try Engine.init(std.testing.allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .vocab_size = 256,
        .backend = .local,
    });
    defer engine.deinit();

    const result = try engine.generate(.{
        .id = 1,
        .prompt = "Test local inference",
        .max_tokens = 8,
    });
    defer result.deinit(std.testing.allocator);

    try std.testing.expect(result.text.len > 0);
}

test "inference: scheduler accepts and tracks requests" {
    var engine = try Engine.init(std.testing.allocator, .{
        .kv_cache_pages = 10,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
    });
    defer engine.deinit();

    const ok1 = try engine.submit(.{ .id = 1, .prompt = "query 1", .priority = 100 });
    const ok2 = try engine.submit(.{ .id = 2, .prompt = "query 2", .priority = 200 });

    try std.testing.expect(ok1);
    try std.testing.expect(ok2);
    try std.testing.expectEqual(@as(u32, 2), engine.getStats().pending_requests);
}

<<<<<<< Updated upstream
test "inference: sampler with deterministic seed produces consistent output" {
    const Sampler = abi.inference.Sampler;
    var s1 = Sampler.initWithSeed(.{ .temperature = 1.0 }, 42);
    var s2 = Sampler.initWithSeed(.{ .temperature = 1.0 }, 42);

    var logits1 = [_]f32{ 1.0, 2.0, 3.0, 0.5, 1.5 };
    var logits2 = [_]f32{ 1.0, 2.0, 3.0, 0.5, 1.5 };

    const token1 = s1.sample(&logits1);
    const token2 = s2.sample(&logits2);
    try std.testing.expectEqual(token1, token2);
}

test "inference: sampler argmax returns highest logit index" {
    const Sampler = abi.inference.Sampler;
    const logits = [_]f32{ 0.1, 0.3, 0.9, 0.2, 0.5 };
    try std.testing.expectEqual(@as(u32, 2), Sampler.argmax(&logits));
}

test "inference: sampler handles single-element logits" {
    const Sampler = abi.inference.Sampler;
    var s = Sampler.initWithSeed(.{}, 1);
    var logits = [_]f32{5.0};
    try std.testing.expectEqual(@as(u32, 0), s.sample(&logits));
}

test "inference: KV cache allocates and frees pages" {
    const PagedKVCache = abi.inference.PagedKVCache;
    var cache = try PagedKVCache.init(std.testing.allocator, .{
        .num_pages = 4,
        .page_size = 8,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
    });
    defer cache.deinit();

    // Allocate pages for a sequence
    const ok = try cache.allocate(1, 8);
    try std.testing.expect(ok);

    // Free the sequence — pages should return to pool
    cache.free(1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cache.getUtilization(), 1e-5);
    try std.testing.expectEqual(@as(usize, 0), cache.activeSequences());
}

test "inference: KV cache rejects allocation when full" {
    const PagedKVCache = abi.inference.PagedKVCache;
    var cache = try PagedKVCache.init(std.testing.allocator, .{
        .num_pages = 2,
        .page_size = 4,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 2,
    });
    defer cache.deinit();

    // Fill all pages
    const ok1 = try cache.allocate(1, 8);
    try std.testing.expect(ok1);

    // No pages left — should return false
    const ok2 = try cache.allocate(2, 4);
    try std.testing.expect(!ok2);
}

// Sibling test modules (pulled in via refAllDecls)
const _async = @import("inference_async_test.zig");
=======
const AsyncState = struct {
    var started: std.atomic.Value(bool) = .{ .raw = false };
    var release: std.atomic.Value(bool) = .{ .raw = false };
    var done: std.atomic.Value(bool) = .{ .raw = false };
    var result_id: std.atomic.Value(u64) = .{ .raw = 0 };

    fn reset() void {
        started.store(false, .release);
        release.store(false, .release);
        done.store(false, .release);
        result_id.store(0, .release);
    }

    fn callback(res: Result) void {
        started.store(true, .release);
        result_id.store(res.id, .release);
        done.store(true, .release);
    }

    fn blockingCallback(res: Result) void {
        started.store(true, .release);
        while (!release.load(.acquire)) {
            time_mod.sleepMs(1);
        }
        result_id.store(res.id, .release);
        done.store(true, .release);
    }
};

test "inference: async callback survives prompt reclamation" {
    const allocator = std.testing.allocator;

    AsyncState.reset();

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

    const prompt = try allocator.dupe(u8, "prompt lifetime regression");
    errdefer allocator.free(prompt);

    try engine.generateAsync(.{
        .id = 17,
        .prompt = prompt,
        .max_tokens = 12,
    }, AsyncState.callback);
    allocator.free(prompt);

    const noise = try allocator.alloc(u8, 4096);
    defer allocator.free(noise);
    @memset(noise, 0xaa);

    var spins: usize = 0;
    while (!AsyncState.done.load(.acquire) and spins < 5000) : (spins += 1) {
        time_mod.sleepMs(1);
    }

    try std.testing.expect(AsyncState.done.load(.acquire));
    try std.testing.expectEqual(@as(u64, 17), AsyncState.result_id.load(.acquire));
}

test "inference: engine deinit waits for in-flight async work" {
    const allocator = std.testing.allocator;

    AsyncState.reset();

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 256,
    });

    const releaser = try std.Thread.spawn(.{}, struct {
        fn run(_: void) void {
            while (!AsyncState.started.load(.acquire)) {
                time_mod.sleepMs(1);
            }
            time_mod.sleepMs(20);
            AsyncState.release.store(true, .release);
        }
    }.run, .{});

    try engine.generateAsync(.{
        .id = 99,
        .prompt = "hold open",
        .max_tokens = 16,
    }, AsyncState.blockingCallback);

    while (!AsyncState.started.load(.acquire)) {
        time_mod.sleepMs(1);
    }

    engine.deinit();
    releaser.join();

    try std.testing.expect(AsyncState.done.load(.acquire));
    try std.testing.expectEqual(@as(u64, 99), AsyncState.result_id.load(.acquire));
}

test "inference: average throughput is based on elapsed time" {
    var engine = try Engine.init(std.testing.allocator, .{
        .kv_cache_pages = 10,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 2,
    });
    defer engine.deinit();

    engine.total_requests = 4;
    engine.total_tokens = 240;
    engine.total_elapsed_ns = 3 * std.time.ns_per_s;

    const stats = engine.getStats();
    try std.testing.expectApproxEqAbs(@as(f32, 80.0), stats.avg_tokens_per_second, 0.001);
}

test "inference: deep end-to-end workload with gpu interaction" {
    const allocator = std.testing.allocator;

    // 1. Initialize GPU (lightweight mock interaction via simulated backend)
    var gpu = try abi.gpu.Gpu.init(allocator, .{
        .preferred_backend = .simulated,
        .allow_fallback = true,
    });
    defer gpu.deinit();

    // Verify GPU is ready
    const health = try gpu.getHealth();
    try std.testing.expect(health.status == .healthy or health.status == .degraded);

    // 2. Initialize Inference Engine
    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 200,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .vocab_size = 256,
        .backend = .demo,
    });
    defer engine.deinit();

    // 3. Submit a batch of requests representing a workload
    const workload_size = 4;
    var i: u64 = 0;
    while (i < workload_size) : (i += 1) {
        const ok = try engine.submit(.{
            .id = i + 1,
            .prompt = "Analyze this deeply simulated data.",
            .max_tokens = 16,
            .priority = @intCast(100 + i),
        });
        try std.testing.expect(ok);
    }

    try std.testing.expectEqual(@as(usize, workload_size), engine.scheduler.pendingCount());

    // 4. Process the workload
    const batch = try engine.scheduler.getBatch(workload_size);
    defer allocator.free(batch);
    try std.testing.expectEqual(@as(usize, workload_size), batch.len);

    var total_tokens_generated: u32 = 0;
    for (batch) |req| {
        const result = try engine.generate(req);
        defer result.deinit(allocator);

        try std.testing.expect(result.text.len > 0);
        try std.testing.expect(result.completion_tokens > 0);
        total_tokens_generated += result.completion_tokens;
    }

    // 5. Simulate writing results/metrics to a GPU buffer
    // Convert our metric to float for the buffer
    const metric_data = [_]f32{ @floatFromInt(total_tokens_generated), engine.getStats().avg_tokens_per_second };
    var stats_buffer = try gpu.createBufferFromSlice(f32, &metric_data, .{});
    defer gpu.destroyBuffer(stats_buffer);

    // Read it back to verify the GPU interaction worked
    var read_back: [2]f32 = [_]f32{0} ** 2;
    try stats_buffer.read(f32, &read_back);

    try std.testing.expectEqual(metric_data[0], read_back[0]);
    try std.testing.expectEqual(metric_data[1], read_back[1]);
    try std.testing.expect(read_back[0] > 0.0);
}
>>>>>>> Stashed changes

test {
    std.testing.refAllDecls(@This());
}
