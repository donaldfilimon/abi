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
    try std.testing.expect(std.mem.indexOf(u8, result.text, "claude-3-sonnet") != null);
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

test {
    std.testing.refAllDecls(@This());
}
