//! Integration Tests — End-to-end verification of the ABI framework

const std = @import("std");
const root = @import("../src/root.zig");

test "database insert and vector search round-trip" {
    const allocator = std.testing.allocator;

    var db = try root.database.Database.init(allocator, .{
        .num_shards = 2,
        .virtual_nodes = 4,
        .dimension = 4,
        .enable_audit = true,
    });
    defer db.deinit();

    // Insert vectors.
    const v1 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const v2 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    const v3 = [_]f32{ 0.0, 0.0, 1.0, 0.0 };
    try db.insert("doc-a", &v1, "first");
    try db.insert("doc-b", &v2, "second");
    try db.insert("doc-c", &v3, "third");

    try std.testing.expectEqual(@as(u64, 3), db.recordCount());

    // Search for nearest to v1.
    const query = [_]f32{ 0.95, 0.05, 0.0, 0.0 };
    const results = try db.search(&query, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len > 0);

    // Verify blockchain integrity.
    try std.testing.expect(db.verifyChain());
}

test "profile routing pipeline" {
    const allocator = std.testing.allocator;
    var moderator = root.AbiModerator.init(allocator);
    defer moderator.deinit();

    // Technical question → should route to Aviva or blended.
    const tech_result = try moderator.process("How do I implement a hash table with open addressing?");
    try std.testing.expect(tech_result.weights.technical_depth > 0.4);

    // Emotional content → should route to Abbey.
    const emo_result = try moderator.process("I feel really confused and need help understanding this");
    try std.testing.expect(emo_result.weights.empathy > 0.5);
}

test "inference engine generate and stats" {
    const allocator = std.testing.allocator;

    var eng = try root.engine.Engine.init(allocator, .{
        .kv_cache_pages = 50,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 128,
    });
    defer eng.deinit();

    const result = try eng.generate(.{
        .id = 42,
        .prompt = "Hello world",
        .max_tokens = 8,
        .temperature = 0.5,
        .top_p = 0.9,
        .top_k = 20,
    });
    defer allocator.free(result.tokens);

    try std.testing.expectEqual(root.engine.FinishReason.stop, result.finish_reason);
    try std.testing.expect(result.completion_tokens > 0);

    const stats = eng.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.total_requests);
}

test "server request pipeline" {
    const allocator = std.testing.allocator;
    var srv = root.Server.init(allocator, .{ .enable_auth = false });
    defer srv.deinit();

    // Health check.
    const health_resp = srv.processRequest(.{
        .method = "GET",
        .path = "/health",
        .body = "",
        .api_key = null,
        .content_type = null,
    });
    try std.testing.expectEqual(@as(u16, 200), health_resp.status);

    // Models list.
    const models_resp = srv.processRequest(.{
        .method = "GET",
        .path = "/v1/models",
        .body = "",
        .api_key = null,
        .content_type = null,
    });
    try std.testing.expectEqual(@as(u16, 200), models_resp.status);
    try std.testing.expect(std.mem.indexOf(u8, models_resp.body, "abbey-1") != null);
}

test "simd distance functions are consistent" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };

    // Cosine of a vector with itself should be 1.
    const self_sim = root.Distance.cosine(&a, &a);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), self_sim, 1e-5);

    // L2 of a vector with itself should be 0.
    const self_dist = root.Distance.l2Squared(&a, &a);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), self_dist, 1e-5);

    // Inner product should be commutative.
    const ip_ab = root.Distance.innerProduct(&a, &b);
    const ip_ba = root.Distance.innerProduct(&b, &a);
    try std.testing.expectApproxEqAbs(ip_ab, ip_ba, 1e-5);
}
