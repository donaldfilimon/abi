//! Backward pass for Rotary Position Embeddings (RoPE).
//!
//! RoPE applies a rotation to pairs of dimensions:
//!   y[i] = x[i] * cos(θ) - x[i + half_dim] * sin(θ)
//!   y[i + half_dim] = x[i] * sin(θ) + x[i + half_dim] * cos(θ)
//!
//! Since this is an orthogonal transform, the backward pass is simply
//! the inverse rotation (transpose of rotation matrix):
//!   dx[i] = dy[i] * cos(θ) + dy[i + half_dim] * sin(θ)
//!   dx[i + half_dim] = -dy[i] * sin(θ) + dy[i + half_dim] * cos(θ)

const std = @import("std");
const rope = @import("../rope.zig");

/// Backward pass for RoPE.
/// Since RoPE is an orthogonal transform, backward = inverse rotation.
///
/// Args:
///   dy: [head_dim] - gradient from upstream
///   pos: position in sequence
///   cache: precomputed cos/sin values
///   dx: [head_dim] - gradient output for input
pub fn ropeBackward(
    dy: []f32,
    pos: u32,
    cache: *const rope.RopeCache,
) void {
    // Apply inverse rotation (same as applyInverseRope)
    const half_dim = cache.config.head_dim / 2;
    const cos = cache.getCos(pos);
    const sin = cache.getSin(pos);

    // Inverse rotation: use transpose of rotation matrix
    for (0..half_dim) |i| {
        const dy0 = dy[i];
        const dy1 = dy[i + half_dim];
        const c = cos[i];
        const s = sin[i];

        // Transpose of [[c, -s], [s, c]] is [[c, s], [-s, c]]
        dy[i] = dy0 * c + dy1 * s;
        dy[i + half_dim] = -dy0 * s + dy1 * c;
    }
}

/// Backward pass for RoPE batch (Q and K).
/// Apply inverse rotation to gradients of Q and K.
pub fn ropeBackwardBatch(
    dq: []f32, // [seq_len, num_heads * head_dim]
    dk: []f32, // [seq_len, num_kv_heads * head_dim]
    start_pos: u32,
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache: *const rope.RopeCache,
) void {
    const hidden_dim = num_heads * head_dim;
    const kv_hidden_dim = num_kv_heads * head_dim;

    for (0..seq_len) |s| {
        const pos = start_pos + @as(u32, @intCast(s));

        // Apply inverse RoPE to each Q head gradient
        for (0..num_heads) |h| {
            const offset = s * hidden_dim + h * head_dim;
            ropeBackward(dq[offset .. offset + head_dim], pos, cache);
        }

        // Apply inverse RoPE to each K head gradient
        for (0..num_kv_heads) |h| {
            const offset = s * kv_hidden_dim + h * head_dim;
            ropeBackward(dk[offset .. offset + head_dim], pos, cache);
        }
    }
}

/// Backward pass for RoPE without cache.
pub fn ropeBackwardNocache(
    dy: []f32,
    pos: u32,
    theta_base: f32,
) void {
    const head_dim = dy.len;
    const half_dim = head_dim / 2;

    for (0..half_dim) |i| {
        // Compute frequency for this dimension pair
        const exp = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim));
        const freq = 1.0 / std.math.pow(f32, theta_base, exp);
        const angle = @as(f32, @floatFromInt(pos)) * freq;

        const c = @cos(angle);
        const s = @sin(angle);

        const dy0 = dy[i];
        const dy1 = dy[i + half_dim];

        // Inverse rotation
        dy[i] = dy0 * c + dy1 * s;
        dy[i + half_dim] = -dy0 * s + dy1 * c;
    }
}

/// Verify that forward and backward are inverse operations.
pub fn verifyRopeGradient(
    allocator: std.mem.Allocator,
    head_dim: u32,
    pos: u32,
    theta_base: f32,
) !bool {
    const half_dim = head_dim / 2;

    // Create cache
    var cache = try rope.RopeCache.init(allocator, .{
        .head_dim = head_dim,
        .theta_base = theta_base,
        .max_seq_len = pos + 1,
    });
    defer cache.deinit();

    // Create test vector
    var x = try allocator.alloc(f32, head_dim);
    defer allocator.free(x);
    for (0..head_dim) |i| {
        x[i] = @as(f32, @floatFromInt(i + 1));
    }

    // Save original
    const x_orig = try allocator.alloc(f32, head_dim);
    defer allocator.free(x_orig);
    @memcpy(x_orig, x);

    // Apply forward then backward
    rope.applyRope(x, pos, &cache);
    ropeBackward(x, pos, &cache);

    // Check that we recover original
    var max_error: f32 = 0;
    for (0..head_dim) |i| {
        const err = @abs(x[i] - x_orig[i]);
        if (err > max_error) max_error = err;
    }

    _ = half_dim;
    return max_error < 1e-5;
}

test "rope backward is inverse of forward" {
    const allocator = std.testing.allocator;

    var cache = try rope.RopeCache.init(allocator, .{
        .head_dim = 4,
        .max_seq_len = 16,
    });
    defer cache.deinit();

    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const x_orig = x;

    // Apply forward
    rope.applyRope(&x, 5, &cache);

    // Apply backward (inverse)
    ropeBackward(&x, 5, &cache);

    // Should recover original
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(x_orig[i], x[i], 0.001);
    }
}

test "rope backward nocache matches cached" {
    const allocator = std.testing.allocator;

    var cache = try rope.RopeCache.init(allocator, .{
        .head_dim = 4,
        .theta_base = 10000.0,
        .max_seq_len = 16,
    });
    defer cache.deinit();

    var x_cached = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var x_nocache = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    // Apply backward with both methods
    ropeBackward(&x_cached, 5, &cache);
    ropeBackwardNocache(&x_nocache, 5, 10000.0);

    // Should match
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(x_cached[i], x_nocache[i], 0.001);
    }
}

test "rope gradient verification" {
    const allocator = std.testing.allocator;

    const is_valid = try verifyRopeGradient(allocator, 64, 100, 10000.0);
    try std.testing.expect(is_valid);
}
