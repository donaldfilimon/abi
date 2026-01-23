//! Rotary Position Embeddings (RoPE) implementation.
//!
//! RoPE encodes position information directly into attention queries and keys
//! by rotating pairs of dimensions based on their position. This allows the
//! model to learn relative positions without explicit position embeddings.

const std = @import("std");

/// RoPE configuration.
pub const RopeConfig = struct {
    /// Head dimension (must be even)
    head_dim: u32,
    /// Base for frequency computation (default: 10000)
    theta_base: f32 = 10000.0,
    /// Maximum sequence length for precomputation
    max_seq_len: u32 = 4096,
    /// Whether to scale frequencies for longer contexts
    scaling_type: ScalingType = .none,
    /// Scaling factor for extended context
    scaling_factor: f32 = 1.0,

    pub const ScalingType = enum {
        none,
        linear,
        ntk, // Neural Tangent Kernel
        yarn, // Yet Another RoPE extensioN
    };
};

/// Precomputed cosine and sine values for RoPE.
pub const RopeCache = struct {
    allocator: std.mem.Allocator,
    config: RopeConfig,
    /// Cosine values: [max_seq_len, head_dim/2]
    cos_cache: []f32,
    /// Sine values: [max_seq_len, head_dim/2]
    sin_cache: []f32,

    pub fn init(allocator: std.mem.Allocator, config: RopeConfig) !RopeCache {
        const half_dim = config.head_dim / 2;
        const cache_size = @as(usize, config.max_seq_len) * half_dim;

        const cos_cache = try allocator.alloc(f32, cache_size);
        errdefer allocator.free(cos_cache);
        const sin_cache = try allocator.alloc(f32, cache_size);
        errdefer allocator.free(sin_cache);

        // Precompute frequencies
        var freqs = try allocator.alloc(f32, half_dim);
        defer allocator.free(freqs);

        for (0..half_dim) |i| {
            const exp = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(config.head_dim));
            freqs[i] = 1.0 / std.math.pow(f32, config.theta_base, exp);
        }

        // Apply scaling if configured
        if (config.scaling_type == .linear and config.scaling_factor != 1.0) {
            for (freqs) |*f| {
                f.* /= config.scaling_factor;
            }
        }

        // Precompute cos/sin for each position
        for (0..config.max_seq_len) |pos| {
            const pos_f = @as(f32, @floatFromInt(pos));
            for (0..half_dim) |i| {
                const angle = pos_f * freqs[i];
                const idx = pos * half_dim + i;
                cos_cache[idx] = @cos(angle);
                sin_cache[idx] = @sin(angle);
            }
        }

        return .{
            .allocator = allocator,
            .config = config,
            .cos_cache = cos_cache,
            .sin_cache = sin_cache,
        };
    }

    pub fn deinit(self: *RopeCache) void {
        self.allocator.free(self.cos_cache);
        self.allocator.free(self.sin_cache);
        self.* = undefined;
    }

    /// Get cos values for a position.
    pub fn getCos(self: *const RopeCache, pos: u32) []const f32 {
        const half_dim = self.config.head_dim / 2;
        const start = @as(usize, pos) * half_dim;
        return self.cos_cache[start .. start + half_dim];
    }

    /// Get sin values for a position.
    pub fn getSin(self: *const RopeCache, pos: u32) []const f32 {
        const half_dim = self.config.head_dim / 2;
        const start = @as(usize, pos) * half_dim;
        return self.sin_cache[start .. start + half_dim];
    }
};

/// Apply RoPE to a single vector at a given position.
/// x: [head_dim], modified in-place
pub fn applyRope(x: []f32, pos: u32, cache: *const RopeCache) void {
    const half_dim = cache.config.head_dim / 2;
    const cos = cache.getCos(pos);
    const sin = cache.getSin(pos);

    // Rotate pairs of dimensions
    for (0..half_dim) |i| {
        const x0 = x[i];
        const x1 = x[i + half_dim];
        const c = cos[i];
        const s = sin[i];

        x[i] = x0 * c - x1 * s;
        x[i + half_dim] = x0 * s + x1 * c;
    }
}

/// Apply RoPE to Q and K tensors for a batch of positions.
/// q, k: [seq_len, num_heads, head_dim] or [seq_len, hidden_dim]
pub fn applyRopeBatch(
    q: []f32,
    k: []f32,
    start_pos: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    cache: *const RopeCache,
) void {
    const hidden_dim = num_heads * head_dim;

    for (0..seq_len) |s| {
        const pos = start_pos + @as(u32, @intCast(s));

        // Apply RoPE to each head
        for (0..num_heads) |h| {
            const offset = s * hidden_dim + h * head_dim;

            // Apply to Q
            applyRope(q[offset .. offset + head_dim], pos, cache);

            // Apply to K
            applyRope(k[offset .. offset + head_dim], pos, cache);
        }
    }
}

/// Apply RoPE without precomputed cache (slower, but no memory overhead).
pub fn applyRopeNocache(x: []f32, pos: u32, theta_base: f32) void {
    const head_dim = x.len;
    const half_dim = head_dim / 2;

    for (0..half_dim) |i| {
        // Compute frequency for this dimension pair
        const exp = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim));
        const freq = 1.0 / std.math.pow(f32, theta_base, exp);
        const angle = @as(f32, @floatFromInt(pos)) * freq;

        const c = @cos(angle);
        const s = @sin(angle);

        const x0 = x[i];
        const x1 = x[i + half_dim];

        x[i] = x0 * c - x1 * s;
        x[i + half_dim] = x0 * s + x1 * c;
    }
}

/// Compute inverse RoPE (for debugging/analysis).
pub fn applyInverseRope(x: []f32, pos: u32, cache: *const RopeCache) void {
    const half_dim = cache.config.head_dim / 2;
    const cos = cache.getCos(pos);
    const sin = cache.getSin(pos);

    // Inverse rotation: use -sin
    for (0..half_dim) |i| {
        const x0 = x[i];
        const x1 = x[i + half_dim];
        const c = cos[i];
        const s = sin[i];

        // Inverse: transpose of rotation matrix
        x[i] = x0 * c + x1 * s;
        x[i + half_dim] = -x0 * s + x1 * c;
    }
}

test "rope cache creation" {
    const allocator = std.testing.allocator;

    var cache = try RopeCache.init(allocator, .{
        .head_dim = 64,
        .max_seq_len = 128,
    });
    defer cache.deinit();

    const cos0 = cache.getCos(0);
    const sin0 = cache.getSin(0);

    // At position 0, sin should be 0 and cos should be 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cos0[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sin0[0], 0.001);
}

test "rope application" {
    const allocator = std.testing.allocator;

    var cache = try RopeCache.init(allocator, .{
        .head_dim = 4,
        .max_seq_len = 16,
    });
    defer cache.deinit();

    // Test vector
    var x = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const x_orig = x;

    // Apply RoPE at position 0 (should be identity-ish)
    applyRope(&x, 0, &cache);

    // At position 0, rotation is minimal
    try std.testing.expectApproxEqAbs(x_orig[0], x[0], 0.01);
}

test "rope inverse" {
    const allocator = std.testing.allocator;

    var cache = try RopeCache.init(allocator, .{
        .head_dim = 4,
        .max_seq_len = 16,
    });
    defer cache.deinit();

    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const x_orig = x;

    // Apply RoPE then inverse should recover original
    applyRope(&x, 5, &cache);
    applyInverseRope(&x, 5, &cache);

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(x_orig[i], x[i], 0.001);
    }
}
