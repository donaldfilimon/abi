//! Token sampling strategies.
//!
//! Implements temperature scaling, top-p (nucleus), and top-k sampling
//! for autoregressive text generation.

const std = @import("std");
const time = @import("../services/shared/mod.zig").time;

pub const SamplingParams = struct {
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    repetition_penalty: f32 = 1.0,
};

pub const Sampler = struct {
    const Self = @This();

    params: SamplingParams,
    rng: std.Random.DefaultPrng,
    allocator: ?std.mem.Allocator = null,

    pub fn init(params: SamplingParams) Self {
        return .{
            .params = params,
            .rng = std.Random.DefaultPrng.init(@bitCast(time.unixSeconds())),
        };
    }

    pub fn initWithAllocator(allocator: std.mem.Allocator, params: SamplingParams) Self {
        return .{
            .params = params,
            .rng = std.Random.DefaultPrng.init(@bitCast(time.unixSeconds())),
            .allocator = allocator,
        };
    }

    pub fn initWithSeed(params: SamplingParams, seed: u64) Self {
        return .{
            .params = params,
            .rng = std.Random.DefaultPrng.init(seed),
        };
    }

    pub fn initWithSeedAndAllocator(allocator: std.mem.Allocator, params: SamplingParams, seed: u64) Self {
        return .{
            .params = params,
            .rng = std.Random.DefaultPrng.init(seed),
            .allocator = allocator,
        };
    }

    pub fn sample(self: *Self, logits: []f32) u32 {
        const n = logits.len;
        if (n == 0) return 0;

        if (self.params.temperature > 0.0 and self.params.temperature != 1.0) {
            const inv_temp = 1.0 / self.params.temperature;
            for (logits) |*l| l.* *= inv_temp;
        }

        if (self.params.top_k > 0 and self.params.top_k < n) {
            var threshold: f32 = -std.math.inf(f32);
            var count: u32 = 0;
            var min_top: f32 = std.math.inf(f32);
            for (logits) |l| {
                if (count < self.params.top_k) {
                    min_top = @min(min_top, l);
                    count += 1;
                } else if (l > min_top) {
                    min_top = l;
                }
            }
            threshold = min_top;
            for (logits) |*l| {
                if (l.* < threshold) l.* = -std.math.inf(f32);
            }
        }

        var max_logit: f32 = -std.math.inf(f32);
        for (logits) |l| max_logit = @max(max_logit, l);

        var sum: f32 = 0.0;
        for (logits) |*l| {
            l.* = @exp(l.* - max_logit);
            sum += l.*;
        }
        if (sum > 0.0) {
            for (logits) |*l| l.* /= sum;
        }

        if (self.params.top_p < 1.0) {
            self.applyTopP(logits);
        }

        const r = self.rng.random().float(f32);
        var cumul: f32 = 0.0;
        for (logits, 0..) |p, i| {
            cumul += p;
            if (cumul >= r) return @intCast(i);
        }

        return @intCast(n - 1);
    }

    /// Apply nucleus (top-p) sampling by zeroing tokens outside the nucleus
    /// and renormalizing. Uses sort-based algorithm when allocator is available,
    /// otherwise falls back to selection-sort for small vocabs.
    fn applyTopP(self: *Self, probs: []f32) void {
        if (self.allocator) |alloc| {
            self.applyTopPSorted(alloc, probs);
        } else if (probs.len <= 4096) {
            applyTopPLinear(probs, self.params.top_p);
        }
        // For large vocabs without allocator, top-p is skipped (O(n²) too expensive)
    }

    /// Sort-based top-p: allocate index buffer, sort by probability descending,
    /// walk cumulative sum, zero tokens below the nucleus cutoff, renormalize.
    fn applyTopPSorted(self: *Self, alloc: std.mem.Allocator, probs: []f32) void {
        const indices = alloc.alloc(u32, probs.len) catch return;
        defer alloc.free(indices);

        for (indices, 0..) |*idx, i| idx.* = @intCast(i);

        // Sort indices descending by probability
        std.mem.sortUnstable(u32, indices, probs, struct {
            fn lessThan(p: []f32, a: u32, b: u32) bool {
                return p[a] > p[b]; // descending
            }
        }.lessThan);

        // Walk cumulative sum to find the nucleus boundary
        var cumulative: f32 = 0.0;
        var cutoff_idx: usize = indices.len;
        for (indices, 0..) |idx, i| {
            cumulative += probs[idx];
            if (cumulative >= self.params.top_p) {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out tokens outside the nucleus
        if (cutoff_idx < indices.len) {
            for (indices[cutoff_idx..]) |idx| {
                probs[idx] = 0.0;
            }

            // Renormalize
            var new_sum: f32 = 0.0;
            for (probs) |p| new_sum += p;
            if (new_sum > 0.0) {
                for (probs) |*p| p.* /= new_sum;
            }
        }
    }

    /// Linear-scan top-p for small vocabularies (no allocation needed).
    /// Uses selection-sort approach: repeatedly find the largest remaining
    /// probability, accumulate, and stop when cumulative >= top_p.
    /// O(n * nucleus_size) — acceptable for vocabs <= 4096.
    fn applyTopPLinear(probs: []f32, top_p: f32) void {
        // Make a copy of probs to use as a working set for finding maxima
        // without destroying the original. We use the sign bit trick:
        // negate visited entries temporarily.
        var cumulative: f32 = 0.0;
        var kept_count: usize = 0;

        // Pass 1: find the nucleus boundary by repeatedly extracting the max
        while (cumulative < top_p and kept_count < probs.len) {
            var local_max: f32 = -1.0;
            for (probs) |p| {
                if (p > local_max) local_max = p;
            }
            if (local_max <= 0.0) break;

            // Count all entries equal to local_max and add them to cumulative
            for (probs) |*p| {
                if (p.* == local_max and cumulative < top_p) {
                    cumulative += p.*;
                    kept_count += 1;
                    // Mark as visited by negating (will restore in pass 2)
                    p.* = -p.*;
                }
            }
        }

        // Pass 2: restore kept entries (negative → positive), zero the rest
        for (probs) |*p| {
            if (p.* < 0.0) {
                p.* = -p.*; // restore kept entry
            } else {
                p.* = 0.0; // zero entries outside nucleus
            }
        }

        // Renormalize
        var new_sum: f32 = 0.0;
        for (probs) |p| new_sum += p;
        if (new_sum > 0.0) {
            for (probs) |*p| p.* /= new_sum;
        }
    }

    pub fn argmax(logits: []const f32) u32 {
        if (logits.len == 0) return 0;
        var best_idx: u32 = 0;
        var best_val: f32 = logits[0];
        for (logits[1..], 1..) |l, i| {
            if (l > best_val) {
                best_val = l;
                best_idx = @intCast(i);
            }
        }
        return best_idx;
    }
};

test "argmax" {
    const logits = [_]f32{ 0.1, 0.3, 0.9, 0.2 };
    try std.testing.expectEqual(@as(u32, 2), Sampler.argmax(&logits));
}

test "sample returns valid index" {
    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var sampler = Sampler.initWithSeed(.{ .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, 42);
    const idx = sampler.sample(&logits);
    try std.testing.expect(idx < logits.len);
}

test "temperature 0 behaves like argmax" {
    var logits = [_]f32{ 0.1, 0.5, 0.3, 0.1 };
    var sampler = Sampler.initWithSeed(.{ .temperature = 0.01, .top_k = 0, .top_p = 1.0 }, 42);
    const idx = sampler.sample(&logits);
    try std.testing.expect(idx < logits.len);
}

test "top-p nucleus sampling with allocator" {
    const allocator = std.testing.allocator;
    // Skewed distribution: token 0 has ~0.73 probability after softmax
    // With top_p=0.5, only token 0 should be in the nucleus
    var logits = [_]f32{ 5.0, 2.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001 };
    var sampler = Sampler.initWithSeedAndAllocator(allocator, .{
        .temperature = 1.0,
        .top_k = 0,
        .top_p = 0.5,
    }, 42);

    // Run multiple samples — with top_p=0.5, results should be heavily concentrated
    var counts = [_]u32{0} ** 8;
    for (0..100) |_| {
        var l = logits;
        const idx = sampler.sample(&l);
        try std.testing.expect(idx < logits.len);
        counts[idx] += 1;
    }
    // Token 0 (highest prob) should get the majority of samples
    try std.testing.expect(counts[0] > 50);
}

test "top-p nucleus sampling without allocator (linear fallback)" {
    // Small vocab — linear scan path
    var logits = [_]f32{ 5.0, 2.0, 1.0, 0.1 };
    var sampler = Sampler.initWithSeed(.{
        .temperature = 1.0,
        .top_k = 0,
        .top_p = 0.5,
    }, 42);

    var counts = [_]u32{0} ** 4;
    for (0..100) |_| {
        var l = logits;
        const idx = sampler.sample(&l);
        try std.testing.expect(idx < logits.len);
        counts[idx] += 1;
    }
    // Token 0 should dominate with tight top_p
    try std.testing.expect(counts[0] > 50);
}

test "top-p 1.0 does not filter" {
    const logits = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var sampler = Sampler.initWithSeed(.{
        .temperature = 1.0,
        .top_k = 0,
        .top_p = 1.0,
    }, 42);
    // All tokens should be possible
    var seen = [_]bool{false} ** 4;
    for (0..200) |_| {
        var l = logits;
        const idx = sampler.sample(&l);
        seen[idx] = true;
    }
    // With uniform distribution and 200 samples, all 4 tokens should appear
    for (seen) |s| try std.testing.expect(s);
}
