//! AI Sampling Strategy Tests — Greedy, Top-k, Top-p, Mirostat
//!
//! Tests correctness and statistical properties of token sampling methods.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const Sampler = if (build_options.enable_llm) abi.features.ai.llm.generation.sampler.Sampler else struct {};
const SamplerConfig = if (build_options.enable_llm) abi.features.ai.llm.generation.sampler.SamplerConfig else struct {};
const TopKTopP = if (build_options.enable_llm) abi.features.ai.llm.generation.sampler.TopKTopP else struct {};

// ============================================================================
// Greedy Sampling Tests
// ============================================================================

test "sampler: greedy always selects argmax" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var s = Sampler.init(allocator, .{ .temperature = 0, .repetition_penalty = 1.0 });
    defer s.deinit();

    // Run 10 times — greedy should always pick the same token
    for (0..10) |_| {
        var logits = [_]f32{ 0.1, 5.0, 0.2, 4.9, 0.3 };
        const token = s.sample(&logits);
        try std.testing.expectEqual(@as(u32, 1), token);
    }
}

test "sampler: greedy handles equal logits deterministically" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var s = Sampler.init(allocator, .{ .temperature = 0, .repetition_penalty = 1.0 });
    defer s.deinit();

    var logits = [_]f32{ 3.0, 3.0, 3.0, 3.0 };
    const token = s.sample(&logits);
    // Should pick first max (index 0 by argmax convention)
    try std.testing.expectEqual(@as(u32, 0), token);
}

test "sampler: greedy handles negative logits" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var s = Sampler.init(allocator, .{ .temperature = 0, .repetition_penalty = 1.0 });
    defer s.deinit();

    var logits = [_]f32{ -5.0, -1.0, -3.0, -10.0 };
    const token = s.sample(&logits);
    // -1.0 is the maximum
    try std.testing.expectEqual(@as(u32, 1), token);
}

// ============================================================================
// Temperature Sampling Tests
// ============================================================================

test "sampler: high temperature increases entropy" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Low temperature: should strongly prefer highest logit
    var low_temp = Sampler.init(allocator, .{
        .temperature = 0.1,
        .top_k = 0,
        .top_p = 1.0,
        .seed = 42,
    });
    defer low_temp.deinit();

    var low_counts = [_]u32{ 0, 0, 0, 0 };
    for (0..200) |_| {
        var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const token = low_temp.sample(&logits);
        low_counts[token] += 1;
    }

    // High temperature: should be more uniform
    var high_temp = Sampler.init(allocator, .{
        .temperature = 5.0,
        .top_k = 0,
        .top_p = 1.0,
        .seed = 42,
    });
    defer high_temp.deinit();

    var high_counts = [_]u32{ 0, 0, 0, 0 };
    for (0..200) |_| {
        var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const token = high_temp.sample(&logits);
        high_counts[token] += 1;
    }

    // Low temp should concentrate on token 3 more than high temp
    try std.testing.expect(low_counts[3] > high_counts[3]);
}

// ============================================================================
// Top-K / Top-P Tests
// ============================================================================

test "sampler: top-k restricts to k highest" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var s = Sampler.init(allocator, .{
        .temperature = 1.0,
        .top_k = 2,
        .top_p = 1.0,
        .repetition_penalty = 1.0,
        .seed = 42,
    });
    defer s.deinit();

    // Logits: indices 3 and 4 are highest
    for (0..100) |_| {
        var logits = [_]f32{ 0.0, 0.0, 0.0, 5.0, 4.0 };
        const token = s.sample(&logits);
        // Should only sample from top-2 (indices 3 and 4)
        try std.testing.expect(token == 3 or token == 4);
    }
}

test "sampler: top-p nucleus sampling" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // With top_p=0.5, only tokens whose cumulative prob <= 0.5 should be sampled
    var s = Sampler.init(allocator, .{
        .temperature = 1.0,
        .top_k = 0,
        .top_p = 0.5,
        .repetition_penalty = 1.0,
        .seed = 42,
    });
    defer s.deinit();

    // After softmax of [10, 0, 0, 0, 0], token 0 has ~99.99% prob
    for (0..50) |_| {
        var logits = [_]f32{ 10.0, 0.0, 0.0, 0.0, 0.0 };
        const token = s.sample(&logits);
        try std.testing.expectEqual(@as(u32, 0), token);
    }
}

// ============================================================================
// Repetition Penalty Tests
// ============================================================================

test "sampler: repetition penalty suppresses repeated tokens" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var s = Sampler.init(allocator, .{
        .temperature = 0,
        .repetition_penalty = 3.0,
        .repetition_window = 10,
    });
    defer s.deinit();

    // First sample: token 0 (highest logit 5.0)
    var logits1 = [_]f32{ 5.0, 4.5, 4.0 };
    const t1 = s.sample(&logits1);
    try std.testing.expectEqual(@as(u32, 0), t1);

    // Second sample: token 0's logit penalized from 5.0 to 5.0/3.0 ≈ 1.67
    // Token 1 (4.5) should now win
    var logits2 = [_]f32{ 5.0, 4.5, 4.0 };
    const t2 = s.sample(&logits2);
    try std.testing.expectEqual(@as(u32, 1), t2);

    // Third: tokens 0 and 1 both penalized
    var logits3 = [_]f32{ 5.0, 4.5, 4.0 };
    const t3 = s.sample(&logits3);
    try std.testing.expectEqual(@as(u32, 2), t3);
}

// ============================================================================
// Mirostat Tests
// ============================================================================

test "sampler: mirostat v2 produces valid tokens" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var s = Sampler.init(allocator, .{
        .mirostat = 2,
        .mirostat_tau = 5.0,
        .mirostat_eta = 0.1,
        .seed = 42,
    });
    defer s.deinit();

    for (0..50) |_| {
        var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        const token = s.sample(&logits);
        try std.testing.expect(token < 8);
    }
}

test "sampler: mirostat v1 updates mu adaptively" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var s = Sampler.init(allocator, .{
        .mirostat = 1,
        .mirostat_tau = 3.0,
        .mirostat_eta = 0.5,
        .seed = 42,
    });
    defer s.deinit();

    const initial_mu = s.mirostat_mu;

    // Sample several times to observe mu drift
    for (0..20) |_| {
        var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        _ = s.sample(&logits);
    }

    // Mu should have changed from initial value
    try std.testing.expect(s.mirostat_mu != initial_mu);
}

test "sampler: reset restores initial state" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var s = Sampler.init(allocator, .{
        .mirostat = 2,
        .mirostat_tau = 5.0,
        .repetition_penalty = 1.5,
        .seed = 42,
    });
    defer s.deinit();

    // Modify state by sampling
    var logits = [_]f32{ 1.0, 2.0, 3.0 };
    _ = s.sample(&logits);

    // Reset
    s.reset();

    // Mu should be back to 2 * tau = 10.0
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), s.mirostat_mu, 0.001);
    // Recent tokens should be cleared
    try std.testing.expectEqual(@as(usize, 0), s.recent_tokens.items.len);
}

// ============================================================================
// Standalone TopKTopP Tests
// ============================================================================

test "topk: filters to k highest values" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var logits = [_]f32{ 1.0, 5.0, 2.0, 4.0, 3.0 };
    TopKTopP.topK(&logits, 2, allocator);

    // Only indices 1 (5.0) and 3 (4.0) should remain
    try std.testing.expect(logits[1] == 5.0);
    try std.testing.expect(logits[3] == 4.0);
    // Others should be -inf
    try std.testing.expect(logits[0] == -std.math.inf(f32));
    try std.testing.expect(logits[2] == -std.math.inf(f32));
    try std.testing.expect(logits[4] == -std.math.inf(f32));
}
