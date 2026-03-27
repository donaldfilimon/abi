//! Integration Tests: Inference Engine Multi-Backend
//!
//! Tests the inference engine with demo, connector, and local backends.

const std = @import("std");
const abi = @import("abi");

const Engine = abi.inference.Engine;
const Backend = abi.inference.Backend;

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
    defer std.testing.allocator.free(result.tokens);
    defer std.testing.allocator.free(result.text);

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
    defer std.testing.allocator.free(result.text);

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
    defer std.testing.allocator.free(result.tokens);
    defer std.testing.allocator.free(result.text);

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

test {
    std.testing.refAllDecls(@This());
}
