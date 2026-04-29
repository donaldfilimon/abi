//! Focused inference unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const inference = @import("inference/mod.zig");

test {
    std.testing.refAllDecls(inference);
}

// ── Engine lifecycle tests ─────────────────────────────────────────────

test "engine default backend is demo" {
    const allocator = std.testing.allocator;
    var engine = try inference.Engine.init(allocator, .{
        .kv_cache_pages = 10,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 2,
    });
    defer engine.deinit();

    const stats = engine.getStats();
    try std.testing.expectEqual(inference.Backend.demo, stats.backend);
    try std.testing.expectEqual(@as(u64, 0), stats.total_requests);
    try std.testing.expectEqual(@as(u64, 0), stats.total_tokens_generated);
    try std.testing.expectEqual(@as(u32, 0), stats.pending_requests);
}

test "engine demo generation produces non-empty text with correct id" {
    const allocator = std.testing.allocator;
    var engine = try inference.Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 256,
    });
    defer engine.deinit();

    var result = try engine.generate(.{
        .id = 42,
        .prompt = "Hello inference engine",
        .max_tokens = 10,
    });
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(u64, 42), result.id);
    try std.testing.expect(result.text.len > 0);
    try std.testing.expect(result.completion_tokens > 0);
}

test "engine stats accumulate across multiple generations" {
    const allocator = std.testing.allocator;
    var engine = try inference.Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 256,
    });
    defer engine.deinit();

    var r1 = try engine.generate(.{ .id = 1, .prompt = "first", .max_tokens = 5 });
    r1.deinit(allocator);
    var r2 = try engine.generate(.{ .id = 2, .prompt = "second", .max_tokens = 5 });
    r2.deinit(allocator);

    const stats = engine.getStats();
    try std.testing.expectEqual(@as(u64, 2), stats.total_requests);
    try std.testing.expect(stats.total_tokens_generated > 0);
    try std.testing.expect(stats.avg_tokens_per_second > 0);
}

test "engine connector backend echo fallback for unknown provider" {
    const allocator = std.testing.allocator;
    var engine = try inference.Engine.init(allocator, .{
        .kv_cache_pages = 50,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .backend = .connector,
        .model_id = "echo/unknown-model",
    });
    defer engine.deinit();

    var result = try engine.generate(.{
        .id = 10,
        .prompt = "test connector",
        .max_tokens = 8,
    });
    defer result.deinit(allocator);

    try std.testing.expect(result.text.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, result.text, "[echo/unknown-model]") != null);
    try std.testing.expectEqual(inference.Backend.connector, engine.getStats().backend);
}

// ── Scheduler tests ────────────────────────────────────────────────────

test "scheduler submit and pending count" {
    const allocator = std.testing.allocator;
    var engine = try inference.Engine.init(allocator, .{
        .kv_cache_pages = 10,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
    });
    defer engine.deinit();

    const ok1 = try engine.submit(.{ .id = 1, .prompt = "a", .priority = 200 });
    const ok2 = try engine.submit(.{ .id = 2, .prompt = "b", .priority = 50 });
    try std.testing.expect(ok1);
    try std.testing.expect(ok2);
    try std.testing.expectEqual(@as(u32, 2), engine.getStats().pending_requests);
}

// ── Sampler tests ──────────────────────────────────────────────────────

test "sampler argmax returns index of largest element" {
    const logits = [_]f32{ 0.1, 0.9, 0.5, 0.2 };
    try std.testing.expectEqual(@as(u32, 1), inference.Sampler.argmax(&logits));
}

test "sampler argmax with single element" {
    const logits = [_]f32{1.0};
    try std.testing.expectEqual(@as(u32, 0), inference.Sampler.argmax(&logits));
}

test "sampler argmax with empty slice returns zero" {
    const logits = [_]f32{};
    try std.testing.expectEqual(@as(u32, 0), inference.Sampler.argmax(&logits));
}

test "sampler deterministic with same seed" {
    var logits1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var logits2 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var s1 = inference.Sampler.initWithSeed(.{ .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, 12345);
    var s2 = inference.Sampler.initWithSeed(.{ .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, 12345);
    try std.testing.expectEqual(s1.sample(&logits1), s2.sample(&logits2));
}

// ── KV Cache tests ─────────────────────────────────────────────────────

test "kv cache utilization increases after allocation" {
    const allocator = std.testing.allocator;
    var cache = try inference.PagedKVCache.init(allocator, .{
        .num_pages = 10,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
    });
    defer cache.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cache.getUtilization(), 1e-5);
    try std.testing.expectEqual(@as(usize, 0), cache.activeSequences());

    const ok = try cache.allocate(100, 16);
    try std.testing.expect(ok);
    try std.testing.expect(cache.getUtilization() > 0.0);
    try std.testing.expectEqual(@as(usize, 1), cache.activeSequences());

    cache.free(100);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cache.getUtilization(), 1e-5);
}

test "kv cache multiple sequences tracked independently" {
    const allocator = std.testing.allocator;
    var cache = try inference.PagedKVCache.init(allocator, .{
        .num_pages = 20,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
    });
    defer cache.deinit();

    _ = try cache.allocate(1, 16);
    _ = try cache.allocate(2, 16);
    try std.testing.expectEqual(@as(usize, 2), cache.activeSequences());

    cache.free(1);
    try std.testing.expectEqual(@as(usize, 1), cache.activeSequences());

    cache.free(2);
    try std.testing.expectEqual(@as(usize, 0), cache.activeSequences());
}
